"""학습 운영 스크립트의 공용 경로와 원자적 파일 쓰기(임시 이름으로 쓴 뒤 교체)."""

import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = Path(os.environ.get("LOCO_RUNTIME_DIR", ROOT / "ops" / "runtime"))
# 시험(tests/test_ops.py)에서는 환경 변수로 실제 대기열·runs와 분리한다
RUNS = Path(os.environ.get("LOCO_RUNS_DIR", ROOT / "runs"))
QUEUE_FILE = Path(os.environ.get("LOCO_QUEUE_FILE", ROOT / "configs" / "run_queue.yaml"))

STATE_FILE = RUNTIME / "queue_state.json"
EVENTS_FILE = RUNTIME / "events.jsonl"
PROGRESS_DIR = RUNTIME / "progress"
SMOKE_DIR = RUNTIME / "smoke"
PAUSE_FILE = RUNTIME / "PAUSE"
SUPERVISOR_LOCK = RUNTIME / "supervisor.lock"
SUPERVISOR_PID = RUNTIME / "supervisor.pid"
MONITOR_PID = RUNTIME / "monitor.pid"
MONITOR_EXIT = RUNTIME / "MONITOR_EXIT"  # 원복 시 모니터 정상 종료 신호
JOURNAL_FILE = RUNTIME / "journal.jsonl"

# 학습 프로세스 종료 코드 규약
EXIT_OK = 0
EXIT_ERROR = 1
EXIT_NONFINITE = 3  # 손실 NaN/Inf: 재시도 없이 즉시 failed


def write_json_atomic(path: Path, obj) -> None:
    """임시 파일에 쓴 뒤 교체해, 읽는 쪽이 반쯤 쓴 파일을 보지 않게 한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    for _ in range(20):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            # Windows에서 읽는 쪽이 잠깐 파일을 열고 있으면 교체가 실패할 수 있다
            time.sleep(0.05)
    os.replace(tmp, path)


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, PermissionError):
        return default


def append_event(kind: str, run: str | None = None, **detail) -> dict:
    """events.jsonl에 사건 한 줄을 덧붙인다 (감독자·학습·원복 공용)."""
    ev = {"ts": time.time(), "time": time.strftime("%Y-%m-%d %H:%M:%S"), "kind": kind, "run": run, **detail}
    EVENTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(EVENTS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(ev, ensure_ascii=False) + "\n")
    return ev


def read_events(limit: int | None = None) -> list:
    try:
        lines = EVENTS_FILE.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return []
    if limit:
        lines = lines[-limit:]
    out = []
    for ln in lines:
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError:
            continue
    return out
