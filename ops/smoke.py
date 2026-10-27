"""스모크 테스트 관문: 대기열 run을 20 step으로 짧게 돌려 체크포인트 저장·재개를 확인한다.

사람이 있는 시간에 Claude나 사용자가 실행한다(몇 분). 통과 기록이 있어야 감독자가 그 run을 시작한다.
야간에 몇 분 만에 죽는 설정 오류(경로, OOM, 형식)를 미리 걸러내는 장치다.

1차: LOCO_SMOKE=1, step 15에서 고장 주입 -> 비정상 종료 + checkpoint-10 완료 표식 확인
2차: 같은 설정으로 다시 실행 -> checkpoint-10에서 재개 + 정상 종료 + DONE 확인

사용법:
    python -m ops.smoke <run 이름>
"""

import argparse
import os
import shutil
import subprocess
import sys
import time

from ops.paths import RUNS, SMOKE_DIR, read_events, read_json, write_json_atomic
from ops.supervisor import build_cmd, config_hash, load_queue
from ops.train_hooks import DONE_MARKER, is_complete

SMOKE_STEPS = 20
FAULT_STEP = 15


def run_once(cmd: list, env: dict, log_path, timeout_s: int) -> int:
    with open(log_path, "wb") as f:
        p = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=timeout_s)
    return p.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--timeout-min", type=float, default=30)
    args = ap.parse_args()

    _, runs = load_queue(include_disabled=True)  # 비활성 항목도 미리 스모크 가능
    entry = next((r for r in runs if r["name"] == args.name), None)
    if entry is None:
        sys.exit(f"대기열에 없는 run: {args.name}")
    cmd = build_cmd(entry.get("smoke_cmd") or entry["cmd"])
    sname = f"_smoke_{args.name}"
    run_dir = RUNS / "_smoke" / args.name
    shutil.rmtree(run_dir, ignore_errors=True)
    run_dir.mkdir(parents=True)
    base_env = {**os.environ, **{k: str(v) for k, v in (entry.get("env") or {}).items()},
                "LOCO_SMOKE": "1", "LOCO_RUN_NAME": sname, "LOCO_RUN_DIR": str(run_dir),
                "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1"}
    for k in ("LOCO_FAULT_AT_STEP", "LOCO_FAULT_HANG_AT_STEP", "LOCO_FAULT_ATTEMPT"):
        base_env.pop(k, None)
    timeout = int(args.timeout_min * 60)
    t0 = time.time()
    problems = []

    code1 = run_once(cmd, {**base_env, "LOCO_ATTEMPT": "1", "LOCO_FAULT_AT_STEP": str(FAULT_STEP)},
                     run_dir / "smoke_1.log", timeout)
    ck = run_dir / "checkpoints" / "checkpoint-10"
    if code1 == 0:
        problems.append("1차 실행이 고장 주입에도 정상 종료함")
    if not (ck.is_dir() and is_complete(ck)):
        problems.append("1차 실행 후 checkpoint-10 완료 표식 없음")

    t1 = time.time()
    code2 = run_once(cmd, {**base_env, "LOCO_ATTEMPT": "2"}, run_dir / "smoke_2.log", timeout)
    resumed = [e for e in read_events() if e["kind"] == "resume" and e.get("run") == sname
               and e.get("attempt") == 2 and e.get("ts", 0) >= t1]
    if code2 != 0:
        problems.append(f"2차 실행 종료 코드 {code2}")
    if not resumed:
        problems.append("2차 실행이 체크포인트에서 재개하지 않음")
    done = read_json(run_dir / DONE_MARKER)
    if done is None:
        problems.append("2차 실행 후 DONE 표식 없음")

    rec = {"name": args.name, "passed": not problems, "problems": problems, "config_hash": config_hash(entry),
           "time": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds": round(time.time() - t0),
           "summary": done, "logs": [str(run_dir / "smoke_1.log"), str(run_dir / "smoke_2.log")]}
    write_json_atomic(SMOKE_DIR / f"{args.name}.json", rec)
    shutil.rmtree(run_dir / "checkpoints", ignore_errors=True)
    shutil.rmtree(run_dir / "final", ignore_errors=True)
    print(("통과" if not problems else "실패: " + "; ".join(problems)) + f" ({rec['seconds']}s)")
    if done:
        print("요약:", done)
    sys.exit(0 if not problems else 1)


if __name__ == "__main__":
    main()
