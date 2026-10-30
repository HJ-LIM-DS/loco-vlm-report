"""학습 스크립트 공용 훅: 진행률·하트비트, 원자적 체크포인트 + 완료 표식, NaN 감지, 재개, 고장 주입.

감독자(ops.supervisor)가 넘기는 환경 변수
    LOCO_RUN_NAME, LOCO_RUN_DIR, LOCO_ATTEMPT, LOCO_RUNTIME_DIR
감독자 없이 직접 실행하면 기본값(스크립트가 준 이름, runs/<이름>, 시도 1)을 쓴다.

고장 주입 (고장 주입 시험 F3·F4 전용)
    LOCO_FAULT_AT_STEP=k        step k에서 예외 발생
    LOCO_FAULT_HANG_AT_STEP=k   step k에서 하트비트를 멈추고 무한 대기
    LOCO_FAULT_ATTEMPT=n        n번째 시도에서만 고장 주입 (없으면 모든 시도)

스모크 테스트 (ops.smoke가 설정): LOCO_SMOKE=1이면 학습 스크립트는 20 step, 10 step마다 저장,
가장 긴 샘플을 첫 배치에 넣는 설정으로 실행해야 한다.
"""

import math
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

from ops.paths import EXIT_ERROR, EXIT_NONFINITE, PROGRESS_DIR, RUNS, append_event, write_json_atomic

COMPLETE_MARKER = "COMPLETE"
DONE_MARKER = "DONE"
HEARTBEAT_MIN_INTERVAL_S = 5.0
GPU_SAMPLE_EVERY_S = 60.0  # nvidia-smi(GPU 전체 사용 메모리)는 1분마다만 묻는다
LOSS_EMA_DECAY = 0.98


class NonFiniteLossError(RuntimeError):
    pass


class InjectedFault(RuntimeError):
    pass


class RunContext:
    """한 번의 학습 시도(attempt)의 진행 상태를 기록한다."""

    def __init__(self, name: str, run_dir: Path, attempt: int, total_steps: int | None = None):
        self.name = name
        self.run_dir = Path(run_dir)
        self.attempt = attempt
        self.total_steps = total_steps
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.final_dir = self.run_dir / "final"
        self.progress_path = PROGRESS_DIR / f"{name}.json"
        self.started = time.time()
        self.loss_ema = None
        self.last_ckpt_time = None
        self._last_beat = 0.0
        self._first_step = None
        self._first_step_time = None
        self._last_gpu_sample = 0.0
        self._gpu_used = {}
        fault_attempt = _env_int("LOCO_FAULT_ATTEMPT")
        armed = fault_attempt is None or fault_attempt == attempt
        self.fault_step = _env_int("LOCO_FAULT_AT_STEP") if armed else None
        self.hang_step = _env_int("LOCO_FAULT_HANG_AT_STEP") if armed else None
        self.smoke = os.environ.get("LOCO_SMOKE") == "1"
        self.run_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls, default_name: str, total_steps: int | None = None) -> "RunContext":
        name = os.environ.get("LOCO_RUN_NAME", default_name)
        run_dir = Path(os.environ.get("LOCO_RUN_DIR", RUNS / name))
        attempt = int(os.environ.get("LOCO_ATTEMPT", "1"))
        return cls(name, run_dir, attempt, total_steps)

    # --- 진행률·하트비트 ---

    def gpu_fields(self) -> dict:
        """프로세스 VRAM(torch 예약·할당)과 GPU 전체 사용 메모리(다른 프로세스 포함). 8B 생성 지연이 낮 시간대에
        2.6배 느렸던 원인을 run 기록만으로 가릴 수 있게 하트비트에 남긴다(2차 개선). 실패하면 빈 dict."""
        out = {}
        torch = sys.modules.get("torch")
        if torch is not None:
            try:
                if torch.cuda.is_available():
                    out["vram_reserved_gb"] = round(torch.cuda.memory_reserved() / 1024**3, 2)
                    out["vram_allocated_gb"] = round(torch.cuda.memory_allocated() / 1024**3, 2)
            except Exception:  # noqa: BLE001
                pass
        now = time.time()
        if now - self._last_gpu_sample >= GPU_SAMPLE_EVERY_S:
            self._last_gpu_sample = now
            try:
                q = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                                   capture_output=True, text=True, timeout=5)
                used, total = (float(x) for x in q.stdout.strip().split(","))
                self._gpu_used = {"gpu_used_gb": round(used / 1024, 2), "gpu_total_gb": round(total / 1024, 2)}
            except Exception:  # noqa: BLE001
                pass
        out.update(self._gpu_used)
        return out

    def heartbeat(self, step: int | None = None, force: bool = False, **fields) -> None:
        now = time.time()
        if not force and now - self._last_beat < HEARTBEAT_MIN_INTERVAL_S:
            return
        self._last_beat = now
        fields = {**self.gpu_fields(), **fields}
        it_s = eta_s = None
        if step is not None:
            if self._first_step is None:
                self._first_step, self._first_step_time = step, now
            elif step > self._first_step:
                it_s = (step - self._first_step) / (now - self._first_step_time)
                if self.total_steps and it_s > 0:
                    eta_s = (self.total_steps - step) / it_s
        write_json_atomic(self.progress_path, {
            "run": self.name,
            "attempt": self.attempt,
            "pid": os.getpid(),
            "step": step,
            "total_steps": self.total_steps,
            "it_s": it_s,
            "eta_s": eta_s,
            "loss_ema": self.loss_ema,
            "last_ckpt_time": self.last_ckpt_time,
            "started_at": self.started,
            "updated_at": now,
            **fields,
        })

    def on_step(self, step: int, loss: float | None = None, **fields) -> None:
        """매 step 호출: 손실 검사, 고장 주입, 하트비트(5초 간격)."""
        if loss is not None:
            self.check_loss(loss, step)
            self.loss_ema = loss if self.loss_ema is None else (
                LOSS_EMA_DECAY * self.loss_ema + (1 - LOSS_EMA_DECAY) * loss)
        self.maybe_inject_fault(step)
        self.heartbeat(step=step, **fields)

    def check_loss(self, loss: float, step: int) -> None:
        if not math.isfinite(loss):
            raise NonFiniteLossError(f"step {step}에서 손실이 {loss}")

    def maybe_inject_fault(self, step: int) -> None:
        if self.fault_step is not None and step == self.fault_step:
            raise InjectedFault(f"고장 주입: step {step}")
        if self.hang_step is not None and step == self.hang_step:
            append_event("fault_hang", self.name, step=step, attempt=self.attempt)
            while True:  # 하트비트 없이 멈춘 상태를 흉내 낸다
                time.sleep(60)

    def note_resume(self, path: Path | None) -> None:
        """재개 여부를 로그와 이벤트에 남긴다 (스모크 테스트·F1 판정에 쓰임)."""
        if path is None:
            print(f"[{self.name}] 처음부터 시작 (시도 {self.attempt})", flush=True)
            return
        print(f"[{self.name}] 재개: {path} (시도 {self.attempt})", flush=True)
        append_event("resume", self.name, path=str(path), attempt=self.attempt)

    def checkpoint_saved(self, path: Path, step: int) -> None:
        mark_checkpoint_complete(path)
        self.last_ckpt_time = time.time()
        append_event("checkpoint", self.name, step=step, path=str(path), attempt=self.attempt)
        self.heartbeat(step=step, force=True)

    def mark_done(self, **summary) -> None:
        write_json_atomic(self.run_dir / DONE_MARKER, {"finished_at": time.time(), "attempt": self.attempt, **summary})
        self.heartbeat(step=self.total_steps, force=True, done=True)


# --- 체크포인트 ---

def mark_checkpoint_complete(path: Path) -> None:
    (Path(path) / COMPLETE_MARKER).write_text(str(time.time()), encoding="utf-8")


def is_complete(path: Path) -> bool:
    return (Path(path) / COMPLETE_MARKER).is_file()


def _ckpt_step(p: Path) -> int:
    try:
        return int(p.name.rsplit("-", 1)[1])
    except (IndexError, ValueError):
        return -1


def list_checkpoints(ckpt_dir: Path) -> list:
    """checkpoint-<step> 폴더를 step 오름차순으로 반환한다."""
    if not Path(ckpt_dir).is_dir():
        return []
    cks = [p for p in Path(ckpt_dir).iterdir() if p.is_dir() and _ckpt_step(p) >= 0]
    return sorted(cks, key=_ckpt_step)


def latest_valid_checkpoint(ckpt_dir: Path, validate=None) -> Path | None:
    """완료 표식이 있고 (주어지면) validate를 통과하는 가장 최근 체크포인트.

    저장 도중 끊겼거나 파일이 손상된 체크포인트는 건너뛰고 그 이전 것으로 재개한다(F5).
    """
    for p in reversed(list_checkpoints(ckpt_dir)):
        if not is_complete(p):
            continue
        if validate is not None:
            try:
                validate(p)
            except Exception as e:  # noqa: BLE001 - 손상 유형이 다양함
                append_event("checkpoint_invalid", None, path=str(p), error=f"{type(e).__name__}: {e}")
                continue
        return p
    return None


def prune_checkpoints(ckpt_dir: Path, keep: int = 2) -> None:
    """완료된 체크포인트를 최근 keep개만 남긴다. 미완료 폴더는 최신 완료본보다 오래됐으면 지운다."""
    cks = list_checkpoints(ckpt_dir)
    done = [p for p in cks if is_complete(p)]
    for p in done[:-keep]:
        shutil.rmtree(p, ignore_errors=True)
    if done:
        for p in cks:
            if not is_complete(p) and _ckpt_step(p) < _ckpt_step(done[-1]):
                shutil.rmtree(p, ignore_errors=True)


# --- 실행 래퍼 ---

def run_main(fn, *args, **kwargs):
    """학습 진입점 래퍼: 종료 코드 규약을 지키고, 마지막 줄에 오류 서명이 남도록 traceback을 출력한다."""
    try:
        fn(*args, **kwargs)
    except NonFiniteLossError:
        traceback.print_exc()
        sys.stdout.flush()
        sys.exit(EXIT_NONFINITE)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        sys.stdout.flush()
        sys.exit(EXIT_ERROR)


def _env_int(key: str) -> int | None:
    v = os.environ.get(key)
    return int(v) if v not in (None, "") else None


# --- 프레임워크별 콜백 ---

def hf_callback(ctx: RunContext, save_every_minutes: float | None = None):
    """transformers Trainer용 콜백. 저장이 끝나면 완료 표식, 시간 기준 저장 트리거."""
    from transformers import TrainerCallback

    class _CB(TrainerCallback):
        def __init__(self):
            self.last_save = time.time()

        def on_train_begin(self, args, state, control, **kw):
            ctx.total_steps = state.max_steps
            ctx.heartbeat(step=state.global_step, force=True, phase="train")

        def on_log(self, args, state, control, logs=None, **kw):
            if logs and "loss" in logs:
                ctx.check_loss(float(logs["loss"]), state.global_step)
                loss = float(logs["loss"])
                ctx.loss_ema = loss if ctx.loss_ema is None else (
                    LOSS_EMA_DECAY * ctx.loss_ema + (1 - LOSS_EMA_DECAY) * loss)
            if logs and "learning_rate" in logs:
                ctx.heartbeat(step=state.global_step, force=True, lr=float(logs["learning_rate"]),
                              epoch=state.epoch, phase="train")

        def on_step_end(self, args, state, control, **kw):
            ctx.maybe_inject_fault(state.global_step)
            ctx.heartbeat(step=state.global_step, epoch=state.epoch, phase="train")
            if save_every_minutes and time.time() - self.last_save >= save_every_minutes * 60:
                control.should_save = True
            return control

        def on_save(self, args, state, control, **kw):
            self.last_save = time.time()
            path = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            if path.is_dir():
                ctx.checkpoint_saved(path, state.global_step)

    return _CB()


def lightning_callback(ctx: RunContext, every_n_steps: int, keep: int = 2):
    """Lightning용 콜백: 진행률·하트비트·고장 주입과 원자적 체크포인트(임시 이름 -> 교체 -> 완료 표식)."""
    from lightning.pytorch.callbacks import Callback

    class _CB(Callback):
        def on_train_start(self, trainer, pl_module):
            ctx.total_steps = trainer.max_steps if trainer.max_steps and trainer.max_steps > 0 else (
                trainer.max_epochs * trainer.num_training_batches)
            ctx.heartbeat(step=trainer.global_step, force=True, phase="train")

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            step = trainer.global_step
            loss = None
            if isinstance(outputs, dict) and "loss" in outputs:
                loss = float(outputs["loss"])
            elif outputs is not None and hasattr(outputs, "item"):
                loss = float(outputs.item())
            ctx.on_step(step, loss, epoch=trainer.current_epoch, phase="train")
            if step > 0 and step % every_n_steps == 0:
                self._save(trainer, step)

        def _save(self, trainer, step):
            final = ctx.ckpt_dir / f"checkpoint-{step}"
            if final.is_dir():
                return
            tmp = ctx.ckpt_dir / f".tmp-checkpoint-{step}"
            shutil.rmtree(tmp, ignore_errors=True)
            tmp.mkdir(parents=True)
            trainer.save_checkpoint(str(tmp / "model.ckpt"))
            os.replace(tmp, final)
            ctx.checkpoint_saved(final, step)
            prune_checkpoints(ctx.ckpt_dir, keep)

    return _CB()
