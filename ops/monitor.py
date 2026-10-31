"""학습 모니터 창 (rich, 2초 갱신). 상태 파일만 읽으므로 학습에 영향이 없다.

화면: 대기열 요약, 현재 학습의 단계(항상 표시)와 진행 막대(it/s, ETA, 손실 이동평균, 학습률, 마지막 체크포인트),
생성 평가처럼 '이름 i/n'으로 진행을 남기는 단계는 두 번째 막대와 처리 속도·남은 시간,
GPU(사용률, VRAM, 온도, 전력, 학습 프로세스의 GPU 점유 여부), 하트비트 경과 시간, 최근 이벤트 5개.
하트비트: 5분 미만 정상 / 5~15분 주의 / 15분 이상 경고 + 경고음.

창을 닫아도 학습은 계속된다. 다시 열기: ops\\open_monitor.cmd

사용법:
    python -m ops.monitor
"""

import atexit
import os
import re
import time

import psutil
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

from ops.paths import (
    MONITOR_EXIT, MONITOR_PID, PAUSE_FILE, PROGRESS_DIR, STATE_FILE, SUPERVISOR_PID, read_events, read_json, write_json_atomic,
)

REFRESH_S = 2.0
GPU_EVERY_S = 5.0
WARN_S, ALERT_S = 5 * 60, 15 * 60
STATUS_STYLE = {"done": "green", "running": "bold cyan", "retrying": "yellow", "failed": "bold red",
                "blocked": "magenta", "pending": "white"}
# 학습 스크립트가 진행 파일에 남기는 단계 이름 → 화면 표시
PHASE_KO = {
    "train": "학습", "eval_generate": "inner-val 생성 평가", "predict_heldout": "held-out 생성 평가",
    "save_final": "최종 저장", "export": "탐지 결과 내보내기", "generate": "생성(프롬프트 기준선)",
    "mmad": "MMAD 객관식 평가", "fit": "기준선 학습", "load": "모델 불러오기", "normalization": "정규화 계산",
    "mahalanobis": "Mahalanobis 특징 추출", "official_test": "공식 test 실행",
}
PHASE_RE = re.compile(r"^(\S+)\s+(\d+)/(\d+)$")


class PhaseRate:
    """'이름 i/n' 단계의 처리 속도. 같은 run·시도·단계의 첫 관측부터의 진행량을 진행 파일 갱신 시각으로 나눈다."""

    def __init__(self):
        self.first = {}

    def rate(self, key: tuple, done: int, t: float) -> float | None:
        if key not in self.first or done < self.first[key][0]:
            self.first[key] = (done, t)
            return None
        d0, t0 = self.first[key]
        return (done - d0) / (t - t0) if t > t0 and done > d0 else None


phase_rate = PhaseRate()


def fmt_dur(s) -> str:
    if s is None:
        return "-"
    s = int(s)
    h, r = divmod(s, 3600)
    m, sec = divmod(r, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


def alive(rec: dict | None) -> bool:
    if not rec:
        return False
    try:
        return abs(psutil.Process(rec["pid"]).create_time() - rec.get("create_time", 0)) < 1.0
    except (psutil.NoSuchProcess, KeyError, ValueError):
        return False


class Gpu:
    def __init__(self):
        self.ok, self.last, self.info = False, 0.0, {}
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nv = pynvml
            self.h = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.ok = True
        except Exception:  # noqa: BLE001
            pass

    def read(self) -> dict:
        if not self.ok or time.time() - self.last < GPU_EVERY_S:
            return self.info
        nv, h = self.nv, self.h
        mem = nv.nvmlDeviceGetMemoryInfo(h)
        try:
            pids = {p.pid for p in nv.nvmlDeviceGetComputeRunningProcesses(h)}
        except Exception:  # noqa: BLE001
            pids = set()
        self.info = {
            "util": nv.nvmlDeviceGetUtilizationRates(h).gpu,
            "used": mem.used / 1024**3, "total": mem.total / 1024**3,
            "temp": nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU),
            "power": nv.nvmlDeviceGetPowerUsage(h) / 1000,
            "pids": pids,
        }
        self.last = time.time()
        return self.info


def queue_panel(state: dict) -> Panel:
    runs = state.get("runs", {})
    n = len(runs)
    done = sum(st["status"] == "done" for st in runs.values())
    failed = sum(st["status"] == "failed" for st in runs.values())
    t = Table(expand=True, box=None, padding=(0, 1))
    t.add_column("run", ratio=3)
    t.add_column("상태", ratio=1)
    t.add_column("시도", justify="right", ratio=1)
    t.add_column("비고", ratio=4)
    for name, st in runs.items():
        note = st.get("blocked_reason") or ""
        if not note and st.get("last_sig"):
            # 실행·재시도 중인 run의 오류 서명은 지금 오류가 아니라 직전 시도의 실패 이유다
            prefix = "직전 시도 오류: " if st["status"] in ("running", "retrying", "pending") else ""
            note = prefix + st["last_sig"]
        t.add_row(name, Text(st["status"], style=STATUS_STYLE.get(st["status"], "")),
                  str(st.get("attempts", 0)), note[:70])
    title = f"대기열  완료 {done}/{n}  실패 {failed}" + ("  [일시 정지]" if PAUSE_FILE.exists() else "")
    return Panel(t, title=title, border_style="blue")


def current_panel(state: dict, gpu: dict, beep_state: dict) -> Panel:
    running = [(n, st) for n, st in state.get("runs", {}).items() if st["status"] == "running"]
    if not running:
        return Panel(Text("실행 중인 학습 없음", style="dim"), title="현재 학습", border_style="cyan")
    name, st = running[0]
    prog = read_json(PROGRESS_DIR / f"{name}.json", {})
    fresh = prog.get("attempt") == st.get("attempts")
    step = prog.get("step") if fresh else None
    total = prog.get("total_steps") if fresh else None
    age = time.time() - (prog["updated_at"] if fresh and prog.get("updated_at") else st.get("started_at", time.time()))
    style = "green" if age < WARN_S else ("yellow" if age < ALERT_S else "bold red")
    if age >= ALERT_S and time.time() - beep_state.get("t", 0) > 60:
        print("\a", end="", flush=True)
        beep_state["t"] = time.time()

    pid = st.get("pid")
    gpu_hold = "-"
    if pid and gpu.get("pids") is not None:
        try:
            family = {pid} | {c.pid for c in psutil.Process(pid).children(recursive=True)}
            gpu_hold = "예" if family & gpu["pids"] else "아니오(또는 확인 불가)"
        except psutil.NoSuchProcess:
            gpu_hold = "프로세스 없음"

    lines = [Text.assemble(("run ", "bold"), name, f"  (시도 {st.get('attempts')}, pid {pid})")]
    phase = prog.get("phase") if fresh else None
    m = PHASE_RE.match(phase or "")
    pname = m.group(1) if m else (phase or "")
    lines.append(Text.assemble(("단계 ", "bold"), (PHASE_KO.get(pname, pname) or "시작 대기", "bold magenta"),
                               (f"  ({phase})" if phase else "", "dim")))
    if step is not None and total:
        lines.append(ProgressBar(total=total, completed=min(step, total), width=60))
        if step >= total:
            # 학습 step은 끝났고 생성 평가·저장 등 뒤 단계가 진행 중이다 (막대 100%는 멈춤이 아님)
            lines.append(Text(f"학습 step {step}/{total} 완료"))
        else:
            lines.append(Text(f"학습 step {step}/{total}  ({100 * step / total:.1f}%)  "
                              f"{(prog.get('it_s') or 0):.2f} it/s  ETA {fmt_dur(prog.get('eta_s'))}  "
                              f"epoch {prog.get('epoch') if prog.get('epoch') is not None else '-'}"))
    if m:
        done_i, n = int(m.group(2)), int(m.group(3))
        r = phase_rate.rate((name, st.get("attempts"), pname), done_i, prog.get("updated_at", time.time()))
        lines.append(ProgressBar(total=max(n, 1), completed=min(done_i, n), width=60, complete_style="magenta"))
        speed = f"{60 * r:.1f}장/분  남은 시간 {fmt_dur((n - done_i) / r)}" if r else "속도 계산 중"
        lines.append(Text(f"{PHASE_KO.get(pname, pname)} {done_i}/{n}  ({100 * done_i / max(n, 1):.1f}%)  {speed}"))
    loss = prog.get("loss_ema") if fresh else None
    lr = prog.get("lr") if fresh else None
    ck = prog.get("last_ckpt_time") if fresh else None
    lines.append(Text(f"손실(EMA) {loss:.4f}" if loss is not None else "손실(EMA) -", style="bold"))
    lines.append(Text(f"학습률 {lr:.2e}" if lr is not None else "학습률 -"))
    lines.append(Text(f"마지막 체크포인트: {time.strftime('%H:%M:%S', time.localtime(ck)) if ck else '-'}"))
    lines.append(Text.assemble("하트비트 경과 ", (fmt_dur(age), style),
                               ("  경고: 15분 이상 멈춤" if age >= ALERT_S else ""), style=""))
    lines.append(Text(f"GPU 메모리 점유: {gpu_hold}"))
    if fresh and prog.get("vram_reserved_gb") is not None:
        # 하트비트가 남긴 프로세스 VRAM과 GPU 전체 사용량: 전체가 한계에 닿으면 다른 프로세스와의 경합(생성 지연 원인)을 의심
        lines.append(Text(f"VRAM 예약 {prog['vram_reserved_gb']:.2f} GB (할당 {prog.get('vram_allocated_gb', 0):.2f})"
                          + (f"  GPU 전체 사용 {prog['gpu_used_gb']:.2f} / {prog['gpu_total_gb']:.2f} GB"
                             if prog.get("gpu_used_gb") is not None else "")))
    return Panel(Group(*lines), title="현재 학습", border_style="cyan")


def gpu_panel(g: dict) -> Panel:
    if not g:
        return Panel(Text("GPU 정보 없음", style="dim"), title="GPU", border_style="green")
    txt = (f"사용률 {g['util']}%   VRAM {g['used']:.1f}/{g['total']:.1f} GB   "
           f"온도 {g['temp']}°C   전력 {g['power']:.0f} W")
    return Panel(Text(txt), title="GPU", border_style="green")


def events_panel() -> Panel:
    t = Table(expand=True, box=None, padding=(0, 1), show_header=False)
    for ev in read_events(limit=5):
        detail = {k: v for k, v in ev.items() if k not in ("ts", "time", "kind", "run")}
        t.add_row(ev["time"][11:], ev["kind"], ev.get("run") or "", str(detail)[:80])
    return Panel(t, title="최근 이벤트", border_style="white")


def render(gpu: Gpu, beep_state: dict):
    state = read_json(STATE_FILE, {"runs": {}})
    g = gpu.read()
    sup = "실행 중" if alive(read_json(SUPERVISOR_PID)) else "꺼짐 (작업 스케줄러가 10분 안에 재시작)"
    header = Text(f"LOCO 학습 모니터   {time.strftime('%Y-%m-%d %H:%M:%S')}   감독자: {sup}", style="bold")
    return Group(header, queue_panel(state), current_panel(state, g, beep_state), gpu_panel(g), events_panel())


def main():
    me = psutil.Process()
    write_json_atomic(MONITOR_PID, {"pid": os.getpid(), "create_time": me.create_time()})

    def _cleanup():
        rec = read_json(MONITOR_PID, {})
        if rec.get("pid") == os.getpid():
            try:
                MONITOR_PID.unlink()
            except FileNotFoundError:
                pass

    atexit.register(_cleanup)
    gpu, beep_state = Gpu(), {}
    with Live(render(gpu, beep_state), refresh_per_second=1, screen=True) as live:
        while not MONITOR_EXIT.exists():
            time.sleep(REFRESH_S)
            try:
                live.update(render(gpu, beep_state))
            except Exception as e:  # noqa: BLE001 - 상태 파일 교체 중 일시 오류는 다음 갱신에서 회복
                live.update(Text(f"갱신 오류: {e}", style="red"))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
