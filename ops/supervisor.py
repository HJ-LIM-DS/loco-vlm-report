"""학습 감독자: 대기열(configs/run_queue.yaml)을 순서대로 실행하고 감시한다.

작업 스케줄러(\\LOCO-Train\\Supervisor)가 로그온 시와 10분마다 pythonw로 실행한다.
이미 다른 감독자가 돌고 있으면 바로 종료한다(잠금 파일). Claude Code 세션과 무관하게 동작한다.

run 상태: pending -> running -> done | retrying -> ... | failed, blocked(스모크·사전 점검 미통과)
- 대기열 파일은 매 루프 다시 읽으므로 학습 중에도 추가·순서 변경이 가능하다.
- ops/runtime/PAUSE가 있으면 실행 중인 run을 멈추고(체크포인트에서 재개 가능) 새 run을 시작하지 않는다.
- 재시도는 같은 설정 그대로 최신 유효 체크포인트에서 재개한다(하이퍼파라미터 자동 변경 없음).
- 같은 오류 서명(traceback 마지막 줄)으로 max_same_error회 실패하면 failed로 두고 다음 run으로 넘어간다.
- 손실 NaN/Inf(종료 코드 3)는 재시도 없이 failed.
- 하트비트가 heartbeat_timeout_min 이상 멈추면 프로세스 트리를 종료하고 재개한다.
- 감독자가 재시작되면 살아 있는 학습 프로세스를 다시 이어받는다.
- 모든 run이 done이고 결과 검증을 통과하면 auto_teardown 설정에 따라 원복(ops.envctl teardown)을 호출한다.

학습 스크립트 규약: ops.train_hooks 참고 (LOCO_* 환경 변수, 체크포인트 재개, DONE 표식, LOCO_SMOKE).

사용법:
    pythonw -m ops.supervisor          # 작업 스케줄러가 실행
    python -m ops.supervisor --status  # 상태 요약 출력
"""

import _winapi
import argparse
import ctypes
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import psutil
import yaml
from filelock import FileLock, Timeout

from ops.notify import toast
from ops.paths import (
    EVENTS_FILE, EXIT_NONFINITE, MONITOR_PID, PAUSE_FILE, PROGRESS_DIR, QUEUE_FILE, ROOT, RUNS, RUNTIME,
    SMOKE_DIR, STATE_FILE, SUPERVISOR_LOCK, SUPERVISOR_PID, append_event, read_json, write_json_atomic,
)
from ops.train_hooks import DONE_MARKER

DEFAULTS = {
    "poll_s": 10,
    "heartbeat_timeout_min": 15,
    "startup_grace_min": 15,
    "min_free_vram_gb": 10,
    "min_free_disk_gb": 50,
    "max_same_error": 2,
    "max_attempts": 6,
    "auto_teardown": False,
    "open_monitor": True,
}
ACTIVE = ("pending", "retrying", "running", "blocked")
# 재부팅 직후 학습 시작을 미루는 시간 (Smart App Control 평판 확인 대기). 운영 장치 시험은 0으로 끈다
BOOT_GRACE_S = float(os.environ.get("LOCO_BOOT_GRACE_S", 300))
CREATE_NO_WINDOW = 0x08000000
CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_BREAKAWAY_FROM_JOB = 0x01000000
DETACHED_PROCESS = 0x00000008
# 절전 방지 (프로세스 범위, 종료 시 자동 해제)
ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
SYNCHRONIZE, PROCESS_QUERY_LIMITED_INFORMATION = 0x00100000, 0x1000
SIG_RE = re.compile(r"^([A-Za-z_][\w.]*(Error|Exception|Interrupt|Exit|Fault)\b.*)$")


# --- 대기열·상태 ---

def load_queue(include_disabled: bool = False) -> tuple[dict, list]:
    q = yaml.safe_load(QUEUE_FILE.read_text(encoding="utf-8")) or {}
    settings = {**DEFAULTS, **(q.get("settings") or {})}
    runs = [r for r in (q.get("runs") or []) if include_disabled or r.get("enabled", True)]
    names = [r["name"] for r in runs]
    if len(names) != len(set(names)):
        raise ValueError("대기열에 같은 run 이름이 있음")
    return settings, runs


def config_hash(entry: dict) -> str:
    keys = {k: entry.get(k) for k in ("cmd", "env", "smoke_cmd")}
    return hashlib.sha1(json.dumps(keys, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:12]


def load_state() -> dict:
    return read_json(STATE_FILE, {"runs": {}})


def save_state(state: dict) -> None:
    state["updated_at"] = time.time()
    write_json_atomic(STATE_FILE, state)


def sync_state(state: dict, runs: list) -> None:
    for r in runs:
        st = state["runs"].setdefault(r["name"], {"status": "pending", "attempts": 0, "sig_counts": {}})
        st["config_hash"] = config_hash(r)


# --- 사전 점검 ---

def smoke_ok(entry: dict, all_runs: list | None = None) -> bool:
    """스모크 통과 기록 확인. smoke_as: 설정 계열이 같은 대표 run의 기록을 공유한다
    (예: seed·fold·데이터만 다른 4B 주력 run들은 대표 run 하나만 스모크)."""
    if not entry.get("smoke", True):
        return True
    ref = entry
    if entry.get("smoke_as"):
        ref = next((r for r in (all_runs or []) if r["name"] == entry["smoke_as"]), None)
        if ref is None:
            return False
    rec = read_json(SMOKE_DIR / f"{ref['name']}.json")
    return bool(rec and rec.get("passed") and rec.get("config_hash") == config_hash(ref))


def free_vram_gb() -> float | None:
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        return pynvml.nvmlDeviceGetMemoryInfo(h).free / 1024**3
    except Exception:  # noqa: BLE001
        return None


def precheck(entry: dict, settings: dict, all_runs: list | None = None) -> str | None:
    """실행 가능하면 None, 아니면 막힌 이유."""
    if not smoke_ok(entry, all_runs):
        return "스모크 테스트 미통과 (python -m ops.smoke " + entry.get("smoke_as", entry["name"]) + ")"
    for p in entry.get("requires", []):
        if not (ROOT / p).exists():
            return f"데이터 경로 없음: {p}"
    need = entry.get("min_free_vram_gb", settings["min_free_vram_gb"])
    vram = free_vram_gb()
    if vram is not None and vram < need:
        return f"남은 VRAM {vram:.1f}GB < {need}GB"
    disk = shutil.disk_usage(ROOT).free / 1024**3
    need_disk = entry.get("min_free_disk_gb", settings["min_free_disk_gb"])
    if disk < need_disk:
        return f"남은 디스크 {disk:.0f}GB < {need_disk}GB"
    return None


# --- 프로세스 ---

def python_exe() -> str:
    # 감독자는 pythonw로 돌지만 학습은 출력을 로그로 받기 위해 python.exe(창 없이)로 실행한다
    exe = Path(sys.executable)
    cand = exe.with_name("python.exe")
    return str(cand if cand.exists() else exe)


def build_cmd(cmd: list) -> list:
    return [python_exe() if c == "python" else str(c) for c in cmd]


def launch(entry: dict, st: dict, watched: dict) -> None:
    name = entry["name"]
    attempt = st["attempts"] + 1
    run_dir = RUNS / name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / f"attempt_{attempt}.log"
    env = {**os.environ, **{k: str(v) for k, v in (entry.get("env") or {}).items()},
           "LOCO_RUN_NAME": name, "LOCO_RUN_DIR": str(run_dir), "LOCO_ATTEMPT": str(attempt),
           "LOCO_RUNTIME_DIR": str(RUNTIME), "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1",
           "PYTHONUNBUFFERED": "1"}
    env.pop("LOCO_SMOKE", None)
    logf = open(log_path, "ab")
    flags = CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP
    try:
        # 작업 스케줄러의 job object에서 떼어 내 감독자가 죽어도 학습이 계속되게 한다
        p = subprocess.Popen(build_cmd(entry["cmd"]), cwd=ROOT, env=env, stdout=logf, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, creationflags=flags | CREATE_BREAKAWAY_FROM_JOB)
    except OSError:
        p = subprocess.Popen(build_cmd(entry["cmd"]), cwd=ROOT, env=env, stdout=logf, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, creationflags=flags)
    logf.close()
    watched[name] = Watched(p.pid)
    st.update(status="running", attempts=attempt, pid=p.pid, started_at=time.time(),
              create_time=psutil.Process(p.pid).create_time(), log=str(log_path))
    append_event("run_start", name, attempt=attempt, pid=p.pid, log=str(log_path))


class Watched:
    """프로세스 핸들을 잡아 두어, 끝난 뒤에도 종료 코드를 읽을 수 있게 한다 (감독자가 재시작돼 이어받은 경우 포함)."""

    def __init__(self, pid: int):
        self.pid = pid
        self.h = _winapi.OpenProcess(SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION, False, pid)

    def poll(self) -> int | None:
        if _winapi.WaitForSingleObject(self.h, 0) == _winapi.WAIT_OBJECT_0:
            return _winapi.GetExitCodeProcess(self.h)
        return None

    def close(self) -> None:
        _winapi.CloseHandle(self.h)


def proc_of(st: dict) -> psutil.Process | None:
    pid = st.get("pid")
    if not pid:
        return None
    try:
        p = psutil.Process(pid)
        if abs(p.create_time() - st.get("create_time", 0)) > 1.0:
            return None  # PID 재사용
        return p
    except psutil.NoSuchProcess:
        return None


def kill_tree(p: psutil.Process) -> None:
    procs = [p] + p.children(recursive=True)
    for q in procs:
        try:
            q.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(procs, timeout=30)


def error_signature(log_path: str | None, code: int | None) -> str:
    """traceback 마지막 줄. 없으면 종료 코드."""
    if log_path and Path(log_path).exists():
        tail = Path(log_path).read_bytes()[-20000:].decode("utf-8", errors="replace").splitlines()
        for ln in reversed(tail):
            m = SIG_RE.match(ln.strip())
            if m:
                # step 번호·경로 같은 가변 부분은 숫자를 지워 같은 오류로 묶는다
                return re.sub(r"\d+", "N", m.group(1))[:200]
    return f"exit code {code}"


def outputs_ok(entry: dict) -> str | None:
    """결과 검증. 학습(kind: train)은 DONE 표식 필수, 데이터 작업(kind: task)은 종료 코드 + outputs만 본다."""
    run_dir = RUNS / entry["name"]
    if entry.get("kind", "train") == "train" and not (run_dir / DONE_MARKER).is_file():
        return "DONE 표식 없음"
    for p in entry.get("outputs", []):
        path = ROOT / p
        if not path.exists():
            return f"결과 파일 없음: {p}"
        if path.suffix == ".json":
            try:
                json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return f"결과 파일 파싱 실패: {p}"
    return None


def heartbeat_age(st: dict) -> float:
    prog = read_json(PROGRESS_DIR / f"{st['name']}.json", {})
    t0 = st.get("started_at", time.time())
    if prog.get("attempt") == st["attempts"] and prog.get("updated_at", 0) >= t0:
        return time.time() - prog["updated_at"]
    return time.time() - t0  # 아직 첫 하트비트 전 (모델 로딩 등)


# --- 판정 ---

def finish(entry: dict, st: dict, code: int | None, settings: dict) -> None:
    name = entry["name"]
    st["pid"] = None
    st["finished_at"] = time.time()
    if code == 0:
        bad = outputs_ok(entry)
        if bad is None:
            st["status"] = "done"
            append_event("run_done", name, attempt=st["attempts"])
            return
        sig = f"출력 검증 실패: {bad}"
    elif code == EXIT_NONFINITE:
        st.update(status="failed", last_sig="손실 NaN/Inf")
        append_event("run_failed", name, attempt=st["attempts"], reason="손실 NaN/Inf")
        toast("LOCO 학습 실패", f"{name}: 손실 NaN/Inf, 다음 run으로 넘어감")
        return
    else:
        sig = st.pop("forced_sig", None) or error_signature(st.get("log"), code)
    counts = st.setdefault("sig_counts", {})
    counts[sig] = counts.get(sig, 0) + 1
    st["last_sig"] = sig
    if counts[sig] >= settings["max_same_error"] or st["attempts"] >= settings["max_attempts"]:
        st["status"] = "failed"
        append_event("run_failed", name, attempt=st["attempts"], reason=sig, count=counts[sig])
        toast("LOCO 학습 실패", f"{name}: {sig[:80]} ({counts[sig]}회), 다음 run으로 넘어감")
    else:
        st["status"] = "retrying"
        append_event("run_retry", name, attempt=st["attempts"], reason=sig, code=code)


def check_running(entry: dict, st: dict, settings: dict, watched: dict) -> None:
    name = entry["name"]
    w = watched.get(name)
    if w is None:
        p = proc_of(st)
        if p is None:
            # 감독자가 내려가 있는 동안 끝났다: 종료 코드를 알 수 없으므로 DONE 표식으로 판정
            code = 0 if (RUNS / name / DONE_MARKER).is_file() else None
            if code is None:
                st["forced_sig"] = "감독자 부재 중 종료(종료 코드 미상)"
            finish(entry, st, code, settings)
            return
        w = watched[name] = Watched(p.pid)
        append_event("run_adopted", name, attempt=st["attempts"], pid=p.pid)
    code = w.poll()
    if code is not None:
        w.close()
        watched.pop(name, None)
        finish(entry, st, code, settings)
        return
    age = heartbeat_age(st)
    first = read_json(PROGRESS_DIR / f"{name}.json", {}).get("attempt") != st["attempts"]
    key = "startup_grace_min" if first else "heartbeat_timeout_min"
    limit = (entry.get(key) or settings[key]) * 60  # run별 덮어쓰기 (고장 주입 시험 F4는 1분)
    if entry.get("kind", "train") == "task":
        limit = max(limit, (entry.get("timeout_min") or 600) * 60)  # 하트비트 없는 데이터 작업
    if age > limit:
        append_event("heartbeat_timeout", name, attempt=st["attempts"], age_s=round(age))
        p = proc_of(st)
        if p is not None:
            kill_tree(p)
        w.close()
        watched.pop(name, None)
        st["forced_sig"] = "하트비트 정지"
        finish(entry, st, None, settings)


def pause_running(entry: dict, st: dict, watched: dict) -> None:
    p = proc_of(st)
    if p is not None:
        kill_tree(p)
    w = watched.pop(entry["name"], None)
    if w is not None:
        w.close()
    st.update(status="pending", pid=None)
    append_event("run_paused", entry["name"], attempt=st["attempts"])
    # 일시 정지로 끊긴 시도는 실패 횟수에 넣지 않는다
    st["attempts"] = max(0, st["attempts"] - 1)


# --- 모니터 창·원복 ---

def monitor_alive() -> bool:
    rec = read_json(MONITOR_PID, {})
    try:
        p = psutil.Process(rec.get("pid", -1))
        return abs(p.create_time() - rec.get("create_time", 0)) < 1.0
    except (psutil.NoSuchProcess, ValueError):
        return False


def open_monitor() -> None:
    if monitor_alive():
        return
    wt = shutil.which("wt.exe")
    cmd = ([wt, "-w", "new", "--title", "LOCO 학습 모니터", "-d", str(ROOT), python_exe(), "-m", "ops.monitor"]
           if wt else [python_exe(), "-m", "ops.monitor"])
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    flags = 0 if wt else subprocess.CREATE_NEW_CONSOLE
    try:
        subprocess.Popen(cmd, cwd=ROOT, env=env, creationflags=flags | CREATE_BREAKAWAY_FROM_JOB)
    except OSError:
        subprocess.Popen(cmd, cwd=ROOT, env=env, creationflags=flags)
    append_event("monitor_open", None)


def spawn_teardown() -> None:
    cmd = [python_exe(), "-m", "ops.envctl", "teardown", "--auto"]
    RUNS.mkdir(parents=True, exist_ok=True)
    log = open(RUNS / "_teardown.log", "ab")
    flags = CREATE_NO_WINDOW | DETACHED_PROCESS
    try:
        subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                         creationflags=flags | CREATE_BREAKAWAY_FROM_JOB)
    except OSError:
        subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, creationflags=flags)


# --- 루프 ---

def step(settings: dict, runs: list, state: dict, watched: dict) -> str:
    """루프 1회. 반환: 'busy' | 'idle' | 'finished'."""
    sync_state(state, runs)
    by_name = {r["name"]: r for r in runs}
    running = [n for n, st in state["runs"].items() if st["status"] == "running" and n in by_name]
    paused = PAUSE_FILE.exists()

    for n in running:
        st = state["runs"][n]
        st["name"] = n
        if paused:
            pause_running(by_name[n], st, watched)
        else:
            check_running(by_name[n], st, settings, watched)
    if any(state["runs"][n]["status"] == "running" for n in by_name):
        return "busy"
    if paused:
        return "idle"

    for r in runs:  # 대기열 순서대로
        st = state["runs"][r["name"]]
        st["name"] = r["name"]
        if st["status"] not in ("pending", "retrying", "blocked"):
            continue
        why = precheck(r, settings, load_queue(include_disabled=True)[1])
        if why:
            if st.get("blocked_reason") != why:
                append_event("run_blocked", r["name"], reason=why)
            st.update(status="blocked", blocked_reason=why)
            continue
        st.pop("blocked_reason", None)
        launch(r, st, watched)
        return "busy"

    statuses = [state["runs"][r["name"]]["status"] for r in runs]
    if runs and all(s == "done" for s in statuses):
        return "finished"
    return "idle"


def status_text() -> str:
    state = load_state()
    lines = []
    for n, st in state.get("runs", {}).items():
        sig = st.get("last_sig") or ""
        if sig and st["status"] in ("running", "retrying", "pending"):
            sig = "직전 시도 오류: " + sig  # 지금 오류가 아니라 이전 시도의 실패 이유
        lines.append(f"{n:40s} {st['status']:9s} 시도 {st.get('attempts', 0)} {sig}"
                     f"{' / ' + st['blocked_reason'] if st.get('blocked_reason') else ''}")
    return "\n".join(lines) or "(상태 없음)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--max-loops", type=int, default=None, help="시험용: 루프 횟수 제한")
    args = ap.parse_args()
    if args.status:
        print(status_text())
        return

    RUNTIME.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(SUPERVISOR_LOCK))
    try:
        lock.acquire(timeout=0)
    except Timeout:
        return  # 다른 감독자가 실행 중
    try:
        write_json_atomic(SUPERVISOR_PID, {"pid": os.getpid(), "create_time": psutil.Process().create_time()})
        settings, runs = load_queue()
        state = load_state()
        sync_state(state, runs)
        if not any(state["runs"][r["name"]]["status"] in ACTIVE for r in runs):
            return  # 할 일 없음: 조용히 종료 (작업 스케줄러가 10분 뒤 다시 깨움)
        # 재부팅 직후에는 Smart App Control이 서명 없는 DLL(.pyd)의 평판을 아직 확인하지 못해
        # 일시적으로 로드를 막는다 (F8 시험에서 확인). 부팅 후 BOOT_GRACE_S가 지날 때까지 기다린다.
        wait = BOOT_GRACE_S - (time.time() - psutil.boot_time())
        if wait > 0:
            append_event("boot_grace_wait", None, seconds=round(wait))
            time.sleep(wait)
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        watched, loops, announced = {}, 0, False
        while True:
            try:
                settings, runs = load_queue()
            except Exception as e:  # noqa: BLE001 - 편집 중인 대기열 파일
                append_event("queue_read_error", None, error=str(e)[:200])
                time.sleep(DEFAULTS["poll_s"])
                continue
            state = load_state()
            result = step(settings, runs, state, watched)
            save_state(state)
            if result == "busy" and not announced:
                # 실제로 학습을 돌리거나 이어받을 때만 시작을 기록하고 모니터 창을 연다
                announced = True
                append_event("supervisor_start", None, pid=os.getpid())
                if settings["open_monitor"]:
                    open_monitor()
            if result == "finished":
                append_event("queue_finished", None)
                toast("LOCO 학습 완료", "대기열의 모든 run이 끝났습니다")
                if settings["auto_teardown"]:
                    append_event("auto_teardown_start", None)
                    spawn_teardown()
                break
            if result == "idle" and not any(
                    state["runs"][r["name"]]["status"] in ("pending", "retrying", "running") for r in runs):
                # 남은 것이 failed/blocked뿐이면 종료한다 (작업 스케줄러가 10분마다 다시 확인)
                if announced:
                    append_event("supervisor_idle_exit", None)
                break
            loops += 1
            if args.max_loops and loops >= args.max_loops:
                break
            time.sleep(settings["poll_s"])
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
        try:
            SUPERVISOR_PID.unlink()
        except FileNotFoundError:
            pass
        lock.release()


if __name__ == "__main__":
    main()
