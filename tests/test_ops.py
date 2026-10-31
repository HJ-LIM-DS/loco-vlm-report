"""학습 운영 스크립트 통합 시험 (가짜 학습, GPU 없음).

실제 대기열·runs·런타임과 분리된 임시 폴더에서 감독자를 띄워 고장 주입 시험 F1·F3·F4·F5의
판정 논리와 NaN 즉시 실패, 스모크 관문을 확인한다. 실제 2B 모델 고장 주입 시험은 M1 절차서를 따른다.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest
import yaml

# 감독자·환경 제어는 winreg·_winapi·PowerShell·작업 스케줄러를 쓰는 Windows 전용이다(Linux에서는 건너뜀)
pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="학습 운영 스크립트는 Windows 전용")

ROOT = Path(__file__).resolve().parents[1]
FAST = {"poll_s": 0.2, "heartbeat_timeout_min": 0.05, "startup_grace_min": 0.5, "min_free_vram_gb": 0,
        "min_free_disk_gb": 0, "max_same_error": 2, "max_attempts": 6, "open_monitor": False}


def dummy(steps=30, step_s=0.02, save_every=10, extra=()):
    return ["python", "-m", "ops.dummy_train", "--steps", str(steps), "--step-s", str(step_s),
            "--save-every", str(save_every), *extra]


class Sandbox:
    def __init__(self, tmp: Path):
        self.tmp = tmp
        self.runtime, self.runs, self.queue = tmp / "runtime", tmp / "runs", tmp / "run_queue.yaml"
        self.env = {**os.environ, "LOCO_RUNTIME_DIR": str(self.runtime), "LOCO_RUNS_DIR": str(self.runs),
                    "LOCO_QUEUE_FILE": str(self.queue), "LOCO_NO_TOAST": "1", "PYTHONIOENCODING": "utf-8",
                    "LOCO_BOOT_GRACE_S": "0"}

    def write_queue(self, runs, **settings):
        for r in runs:
            r.setdefault("smoke", False)
        self.queue.write_text(yaml.safe_dump({"settings": {**FAST, **settings}, "runs": runs},
                                             allow_unicode=True), encoding="utf-8")

    def supervisor(self, wait=True, timeout=120):
        cmd = [sys.executable, "-m", "ops.supervisor"]
        if wait:
            return subprocess.run(cmd, cwd=ROOT, env=self.env, timeout=timeout, capture_output=True)
        return subprocess.Popen(cmd, cwd=ROOT, env=self.env)

    def state(self):
        p = self.runtime / "queue_state.json"
        return json.loads(p.read_text(encoding="utf-8"))["runs"] if p.exists() else {}

    def events(self, kind=None):
        p = self.runtime / "events.jsonl"
        evs = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()] if p.exists() else []
        return [e for e in evs if kind is None or e["kind"] == kind]

    def losses(self, name):
        rows = [ln.split(",") for ln in (self.runs / name / "loss.csv").read_text().splitlines()]
        return {int(s): float(v) for _, s, v in rows}  # 같은 step은 마지막 시도 값

    def wait_step(self, name, step, timeout=60):
        t0 = time.time()
        while time.time() - t0 < timeout:
            prog = self.runtime / "progress" / f"{name}.json"
            try:
                if json.loads(prog.read_text(encoding="utf-8")).get("step", 0) >= step:
                    return
            except (FileNotFoundError, json.JSONDecodeError, PermissionError):
                pass
            time.sleep(0.05)
        raise TimeoutError(f"{name} step {step} 도달 안 함")


@pytest.fixture
def sb(tmp_path):
    return Sandbox(tmp_path)


def test_normal_run_done(sb):
    sb.write_queue([{"name": "a", "cmd": dummy(), "outputs": []}])
    sb.supervisor()
    st = sb.state()["a"]
    assert st["status"] == "done" and st["attempts"] == 1
    assert (sb.runs / "a" / "DONE").is_file()


def test_f1_kill_resumes_with_same_loss_curve(sb):
    """F1: 학습 프로세스 강제 종료 → 최신 체크포인트에서 재개, 손실 곡선이 끊김 없는 실행과 일치."""
    sb.write_queue([{"name": "ref", "cmd": dummy(steps=40, step_s=0.05)},
                    {"name": "f1", "cmd": dummy(steps=40, step_s=0.05)}])
    p = sb.supervisor(wait=False)
    sb.wait_step("f1", 25)
    pid = sb.state()["f1"]["pid"]
    psutil.Process(pid).kill()
    p.wait(timeout=120)
    st = sb.state()["f1"]
    assert st["status"] == "done" and st["attempts"] == 2
    assert any(e["run"] == "f1" and e["attempt"] == 2 for e in sb.events("resume"))
    ref, f1 = sb.losses("ref"), sb.losses("f1")
    assert ref.keys() == f1.keys()
    assert max(abs(ref[k] - f1[k]) for k in ref) < 1e-9


def test_f3_same_error_twice_fails_and_moves_on(sb):
    """F3: step k 예외 주입 → 같은 서명 2회 실패 후 failed, 다음 run 시작."""
    sb.write_queue([{"name": "bad", "cmd": dummy(), "env": {"LOCO_FAULT_AT_STEP": 5}},
                    {"name": "next", "cmd": dummy()}])
    sb.supervisor()
    s = sb.state()
    assert s["bad"]["status"] == "failed" and s["bad"]["attempts"] == 2
    assert "InjectedFault" in s["bad"]["last_sig"]
    assert s["next"]["status"] == "done"


def test_f4_heartbeat_stall_killed_and_resumed(sb):
    """F4: 하트비트 중단(시험 임계값 3초) → 강제 종료 후 재개."""
    sb.write_queue([{"name": "hang", "cmd": dummy(steps=30),
                     "env": {"LOCO_FAULT_HANG_AT_STEP": 15, "LOCO_FAULT_ATTEMPT": 1}}])
    sb.supervisor()
    st = sb.state()["hang"]
    assert st["status"] == "done" and st["attempts"] == 2
    assert sb.events("heartbeat_timeout")
    assert any(e["attempt"] == 2 for e in sb.events("resume"))


def test_f5_corrupt_checkpoint_falls_back(sb):
    """F5: 마지막 체크포인트 손상 → 이전 체크포인트에서 재개."""
    sb.write_queue([{"name": "c", "cmd": dummy(steps=40, step_s=0.2)}])
    p = sb.supervisor(wait=False)
    marker = sb.runs / "c" / "checkpoints" / "checkpoint-30" / "COMPLETE"
    t0 = time.time()
    while not marker.exists():  # 하트비트는 5초 간격이라 완료 표식을 직접 기다린다
        assert time.time() - t0 < 60
        time.sleep(0.02)
    pid = sb.state()["c"]["pid"]
    proc = psutil.Process(pid)
    proc.suspend()
    (sb.runs / "c" / "checkpoints" / "checkpoint-30" / "state.json").write_text("{깨진", encoding="utf-8")
    proc.kill()
    p.wait(timeout=120)
    assert sb.state()["c"]["status"] == "done"
    resumes = [e for e in sb.events("resume") if e["attempt"] == 2]
    assert resumes and resumes[0]["path"].endswith("checkpoint-20")
    assert sb.events("checkpoint_invalid")


def test_nan_fails_immediately(sb):
    sb.write_queue([{"name": "nan", "cmd": dummy(extra=["--nan-at", "5"])}])
    sb.supervisor()
    st = sb.state()["nan"]
    assert st["status"] == "failed" and st["attempts"] == 1


def test_smoke_gate_blocks(sb):
    sb.write_queue([{"name": "gated", "cmd": dummy(), "smoke": True}])
    sb.supervisor()
    st = sb.state()["gated"]
    assert st["status"] == "blocked" and "스모크" in st["blocked_reason"]


def test_smoke_as_shares_record(sb):
    """smoke_as: 대표 run의 스모크 기록이 있으면 같은 계열 run도 시작한다."""
    sb.write_queue([{"name": "rep", "cmd": dummy(), "smoke": True, "enabled": False},
                    {"name": "member", "cmd": dummy(steps=10), "smoke": True, "smoke_as": "rep"}])
    from ops.supervisor import config_hash
    rep = {"name": "rep", "cmd": dummy()}
    (sb.runtime / "smoke").mkdir(parents=True)
    (sb.runtime / "smoke" / "rep.json").write_text(
        json.dumps({"passed": True, "config_hash": config_hash(rep)}), encoding="utf-8")
    sb.supervisor()
    assert sb.state()["member"]["status"] == "done"


@pytest.mark.parametrize("module", ["src.vlm.build_sft", "src.vlm.train_lora"])
def test_queue_commands_parse(module):
    """실제 대기열의 명령이 대상 스크립트 파서를 통과하는지 확인한다.
    분석 단계 대기열에서 YAML 흐름 시퀀스 안의 쉼표가 항목 구분자로 해석돼 --exclude-groups 값이 쪼개진 채
    학습 데이터 생성이 실패한 일이 있었다 (따옴표 누락)."""
    import importlib
    parser = importlib.import_module(module).build_parser()
    runs = yaml.safe_load((ROOT / "configs" / "run_queue.yaml").read_text(encoding="utf-8"))["runs"]
    cmds = [r["cmd"] for r in runs if r["cmd"][:3] == ["python", "-m", module]]
    assert cmds
    for cmd in cmds:
        parser.parse_args(cmd[3:])  # 인식하지 못한 인자가 있으면 SystemExit
