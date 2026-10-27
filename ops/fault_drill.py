"""고장 주입 시험 F1~F9 구동과 판정 (실제 2B 모델, 작업 스케줄러 감독자 사용).

drive: 대기열의 f1~f5 run이 진행되는 동안 정해진 시점에 고장을 가한다 (학습은 하지 않고 관찰·주입만).
    F1  f1_kill이 step 25에 도달하면 학습 프로세스 강제 종료
    F2  f2_supervisor가 step 50에 도달하면 감독자 프로세스 강제 종료 (작업 스케줄러가 10분 안에 재시작해야 함)
    F6  f2_supervisor 진행 중 모니터 창을 닫고 30초 뒤 다시 연다
    F5  f5_corrupt의 checkpoint-50이 완료되면 프로세스를 멈추고 어댑터 파일을 잘라 손상시킨 뒤 강제 종료
    (F3·F4는 대기열 env의 고장 주입으로 자동 발생)
check: 이벤트 로그·상태·요약 파일로 판정해 reports/ops/fault_tests.md를 쓴다.
    F7(Claude Code 재시작), F8(로그오프·로그온)은 사용자 동작이 필요해 이벤트 증거만 확인한다.
    F9(원복)는 ops/env.ps1 -Action teardown 이후 verify 결과로 판정한다.

사용법:
    python -m ops.fault_drill drive
    python -m ops.fault_drill check
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import psutil

from ops.paths import (
    MONITOR_PID, PROGRESS_DIR, ROOT, RUNS, STATE_FILE, SUPERVISOR_PID, append_event, read_events, read_json,
)
from ops.train_hooks import is_complete

REPORT = ROOT / "reports" / "ops" / "fault_tests.md"
LOSS_TOL = 0.05  # F1 손실 곡선 허용 오차 (상대). GPU 비결정 연산(sdpa 역전파)만큼의 차이는 허용


def _state(name):
    return read_json(STATE_FILE, {"runs": {}}).get("runs", {}).get(name, {})


def _step(name):
    st = _state(name)
    prog = read_json(PROGRESS_DIR / f"{name}.json", {})
    if st.get("status") == "running" and prog.get("attempt") == st.get("attempts"):
        return prog.get("step") or 0
    return -1


def _proc(rec):
    try:
        p = psutil.Process(rec["pid"])
        return p if abs(p.create_time() - rec.get("create_time", 0)) < 1.0 else None
    except (psutil.NoSuchProcess, KeyError, TypeError):
        return None


def _kill_training(name):
    st = _state(name)
    p = _proc({"pid": st.get("pid"), "create_time": st.get("create_time")})
    if p:
        p.kill()
        append_event("drill_kill_training", name, pid=p.pid)
        return True
    return False


def drive(timeout_h: float = 3.0):
    done = set()
    t0 = time.time()
    while time.time() - t0 < timeout_h * 3600:
        if "F1" not in done and _step("f1_kill") >= 25 and _kill_training("f1_kill"):
            done.add("F1")
        if "F2" not in done and _step("f2_supervisor") >= 50:
            p = _proc(read_json(SUPERVISOR_PID))
            if p:
                p.kill()
                append_event("drill_kill_supervisor", None, pid=p.pid)
                done.add("F2")
        if "F2" in done and "F6" not in done and _step("f2_supervisor") >= 150:
            m = _proc(read_json(MONITOR_PID))
            if m:
                # 창 닫기 = 모니터 프로세스 종료 (Windows Terminal 프로세스는 다른 창도 품고 있어 건드리지 않음)
                m.kill()
                append_event("drill_close_monitor", None, pid=m.pid)
                time.sleep(30)
                subprocess.Popen(["cmd", "/c", str(ROOT / "ops" / "open_monitor.cmd")], cwd=ROOT)
                append_event("drill_reopen_monitor", None)
                done.add("F6")
        ck = RUNS / "f5_corrupt" / "checkpoints" / "checkpoint-50"
        if "F5" not in done and ck.is_dir() and is_complete(ck):
            st = _state("f5_corrupt")
            p = _proc({"pid": st.get("pid"), "create_time": st.get("create_time")})
            if p:
                p.suspend()
                f = ck / "adapter_model.safetensors"
                data = f.read_bytes()
                f.write_bytes(data[: len(data) // 2])  # 파일 절반만 남겨 손상
                p.kill()
                append_event("drill_corrupt_checkpoint", "f5_corrupt", path=str(f))
                done.add("F5")
        states = read_json(STATE_FILE, {"runs": {}}).get("runs", {})
        names = ["f1_ref", "f1_kill", "f2_supervisor", "f3_fault", "f4_hang", "f5_corrupt"]
        if all(states.get(n, {}).get("status") in ("done", "failed") for n in names):
            print("시험 run 종료:", {n: states[n]["status"] for n in names}, "주입:", sorted(done))
            return
        time.sleep(2)
    print("시간 초과. 주입:", sorted(done))


def _loss(name):
    s = read_json(RUNS / name / "final" / "summary.json", {})
    return {int(k): v for k, v in (s.get("train_loss_by_step") or {}).items()}


def check() -> int:
    ev = read_events()
    st = read_json(STATE_FILE, {"runs": {}}).get("runs", {})

    def evs(kind, run=None):
        return [e for e in ev if e["kind"] == kind and (run is None or e.get("run") == run)]

    rows = []

    # F1
    kill = evs("drill_kill_training", "f1_kill")
    starts = [e for e in evs("run_start", "f1_kill") if e.get("attempt") == 2]
    resumed = [e for e in evs("resume", "f1_kill") if e.get("attempt") == 2]
    a, b = _loss("f1_ref"), _loss("f1_kill")
    diff = max((abs(a[k] - b[k]) / max(abs(a[k]), 1e-6) for k in a if k in b), default=None)
    gap = starts[0]["ts"] - kill[0]["ts"] if kill and starts else None
    ok = bool(kill and resumed and gap is not None and gap < 60 and st.get("f1_kill", {}).get("status") == "done"
              and diff is not None and diff <= LOSS_TOL and a.keys() == b.keys())
    rows.append(("F1", "학습 프로세스 강제 종료", ok,
                 f"재시작까지 {gap:.0f}초, 재개 {resumed[0]['path'].split(chr(92))[-1] if resumed else '-'}, "
                 f"손실 최대 상대 차이 {diff:.4f}" if gap is not None and diff is not None else "증거 부족"))
    # F2
    ks = evs("drill_kill_supervisor")
    after = [e for e in evs("supervisor_start") if ks and e["ts"] > ks[0]["ts"]]
    adopted = evs("run_adopted", "f2_supervisor")
    lag = after[0]["ts"] - ks[0]["ts"] if ks and after else None
    ok = bool(ks and after and lag < 11 * 60 and st.get("f2_supervisor", {}).get("status") == "done")
    rows.append(("F2", "감독자 강제 종료", ok, f"재시작까지 {lag / 60:.1f}분, 이어받기 {len(adopted)}건"
                 if lag is not None else "증거 부족"))
    # F3
    s3 = st.get("f3_fault", {})
    nxt = [e for e in evs("run_start", "f4_hang")]
    fail = evs("run_failed", "f3_fault")
    ok = bool(s3.get("status") == "failed" and s3.get("attempts") == 2 and "InjectedFault" in (s3.get("last_sig") or "")
              and nxt and fail and nxt[0]["ts"] > fail[0]["ts"])
    rows.append(("F3", "step 5 예외 주입", ok, f"상태 {s3.get('status')}, 시도 {s3.get('attempts')}, "
                 f"서명 {s3.get('last_sig')}, 다음 run 시작 {'예' if nxt else '아니오'} (토스트는 화면 확인)"))
    # F4
    hb = evs("heartbeat_timeout", "f4_hang")
    r4 = [e for e in evs("resume", "f4_hang") if e.get("attempt") == 2]
    ok = bool(hb and r4 and st.get("f4_hang", {}).get("status") == "done")
    rows.append(("F4", "하트비트 중단(임계값 1분)", ok, f"정지 감지 {hb[0]['age_s'] if hb else '-'}초 후 종료, "
                 f"재개 {r4[0]['path'].split(chr(92))[-1] if r4 else '-'}"))
    # F5
    cor = evs("drill_corrupt_checkpoint", "f5_corrupt")
    inv = evs("checkpoint_invalid")
    r5 = [e for e in evs("resume", "f5_corrupt") if e.get("attempt") == 2]
    ok = bool(cor and inv and r5 and r5[0]["path"].endswith("checkpoint-40")
              and st.get("f5_corrupt", {}).get("status") == "done")
    rows.append(("F5", "마지막 체크포인트 손상", ok, f"손상 감지 {len(inv)}건, 재개 "
                 f"{r5[0]['path'].split(chr(92))[-1] if r5 else '-'}"))
    # F6
    cl, ro = evs("drill_close_monitor"), evs("drill_reopen_monitor")
    retries = [e for e in evs("run_retry", "f2_supervisor") if cl and e["ts"] > cl[0]["ts"]]
    ok = bool(cl and ro and not retries and st.get("f2_supervisor", {}).get("status") == "done")
    rows.append(("F6", "모니터 창 닫기·다시 열기", ok, f"창 닫은 뒤 학습 재시도 {len(retries)}건"))
    # F7, F8: 사용자 동작 증거
    f7 = evs("drill_claude_restart")
    rows.append(("F7", "Claude Code 종료 후 재시작", bool(f7), "사용자 동작 필요" if not f7 else f7[-1].get("note", "")))
    f8 = evs("drill_logon")
    rows.append(("F8", "로그오프·로그온", bool(f8), "사용자 동작 필요" if not f8 else f8[-1].get("note", "")))
    # F9
    f9 = evs("drill_teardown_verify")
    rows.append(("F9", "원복 실행", bool(f9 and f9[-1].get("residual") == 0),
                 "원복 후 판정" if not f9 else f"잔여 변경 {f9[-1].get('residual')}건"))

    lines = [f"# 고장 주입 시험 결과 ({time.strftime('%Y-%m-%d %H:%M:%S')})", "",
             "| 시험 | 주입 | 통과 | 근거 |", "|---|---|---|---|"]
    lines += [f"| {k} | {d} | {'통과' if ok else '미통과'} | {why} |" for k, d, ok, why in rows]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0 if all(ok for _, _, ok, _ in rows) else 1


def f9() -> int:
    """원복 후 판정: verify 잔여 변경 0건 + teardown 보고서의 사전 스냅샷 비교 결과로 F9 줄을 갱신한다."""
    from ops.envctl import verify

    residual = verify()
    rep = ROOT / "reports" / "ops" / "teardown_report.md"
    text = rep.read_text(encoding="utf-8") if rep.exists() else ""
    clean = "**결과: 잔여 변경 없음**" in text
    ok = residual == 0 and clean
    why = (f"verify 잔여 변경 {'0건' if residual == 0 else '있음'}, 사전 스냅샷 비교 "
           f"{'차이 없음(보존 대상 제외)' if clean else '차이 있음 - teardown_report.md 확인'}")
    lines = REPORT.read_text(encoding="utf-8").splitlines()
    lines = [f"| F9 | 원복 실행 | {'통과' if ok else '미통과'} | {why} |" if ln.startswith("| F9 |") else ln
             for ln in lines]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0 if ok else 1


def mark(test: str, ok: bool, why: str, status: str | None = None) -> None:
    """사용자 동작 시험(F7·F8) 판정을 보고서의 해당 줄에 기록하고 이벤트로도 남긴다.

    원복 때 이전 이벤트 로그는 reports/ops/로 보관되므로, check()를 다시 돌리지 않고 줄만 갱신한다.
    """
    kinds = {"F7": "drill_claude_restart", "F8": "drill_logon"}
    append_event(kinds.get(test, "drill_note"), None, note=why, ok=ok)
    lines = REPORT.read_text(encoding="utf-8").splitlines()
    out = []
    for ln in lines:
        if ln.startswith(f"| {test} |"):
            cells = ln.split("|")
            ln = f"| {test} |{cells[2]}| {status or ('통과' if ok else '미통과')} | {why} |"
        out.append(ln)
    REPORT.write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))


def note(kind: str, text: str):
    append_event(kind, None, note=text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["drive", "check", "note", "f9", "mark"])
    ap.add_argument("--kind", default=None, help="note: drill_claude_restart | drill_logon | drill_teardown_verify")
    ap.add_argument("--text", default="")
    ap.add_argument("--residual", type=int, default=None)
    ap.add_argument("--test", default=None, help="mark: F7 | F8")
    ap.add_argument("--fail", action="store_true", help="mark: 미통과로 기록")
    ap.add_argument("--status", default=None, help="mark: 통과 칸 문구 직접 지정 (예: 조건부 통과)")
    args = ap.parse_args()
    if args.action == "drive":
        drive()
    elif args.action == "check":
        sys.exit(check())
    elif args.action == "f9":
        sys.exit(f9())
    elif args.action == "mark":
        mark(args.test, not args.fail, args.text, args.status)
    else:
        if args.residual is not None:
            append_event(args.kind, None, note=args.text, residual=args.residual)
        else:
            note(args.kind, args.text)


if __name__ == "__main__":
    main()
