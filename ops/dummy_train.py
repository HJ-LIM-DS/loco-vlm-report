"""운영 스크립트 시험용 가짜 학습 (GPU 없음). ops.train_hooks 규약을 그대로 따른다.

step마다 시드 RNG로 손실을 만들어 loss.csv에 기록하고, 체크포인트에 RNG 상태를 저장한다.
재개한 실행과 끊김 없는 실행의 손실 곡선이 같아야 한다(F1 판정 방식 검증용).

사용법:
    python -m ops.dummy_train --steps 60 --step-s 0.5 --save-every 10
"""

import argparse
import json
import random
import time

from ops.train_hooks import RunContext, latest_valid_checkpoint, prune_checkpoints, run_main


def validate(path):
    json.loads((path / "state.json").read_text(encoding="utf-8"))


def train(args):
    ctx = RunContext.from_env("dummy", total_steps=args.steps)
    steps = 20 if ctx.smoke else args.steps
    save_every = 10 if ctx.smoke else args.save_every
    ctx.total_steps = steps
    ck = latest_valid_checkpoint(ctx.ckpt_dir, validate)
    ctx.note_resume(ck)
    rng = random.Random(args.seed)
    start = 0
    if ck is not None:
        st = json.loads((ck / "state.json").read_text(encoding="utf-8"))
        start = st["step"]
        rng.setstate((st["rng"][0], tuple(st["rng"][1]), st["rng"][2]))
        ctx.loss_ema = st["loss_ema"]
    log = open(ctx.run_dir / "loss.csv", "a", encoding="utf-8")
    for step in range(start + 1, steps + 1):
        time.sleep(args.step_s)
        loss = 1.0 / (1 + 0.05 * step) + 0.01 * rng.random()
        log.write(f"{ctx.attempt},{step},{loss:.6f}\n")
        log.flush()
        if args.nan_at is not None and step == args.nan_at:
            loss = float("nan")
        ctx.on_step(step, loss, phase="train")
        if step % save_every == 0:
            final = ctx.ckpt_dir / f"checkpoint-{step}"
            final.mkdir(parents=True, exist_ok=True)
            state = {"step": step, "rng": rng.getstate(), "loss_ema": ctx.loss_ema}
            (final / "state.json").write_text(json.dumps(state), encoding="utf-8")
            ctx.checkpoint_saved(final, step)
            prune_checkpoints(ctx.ckpt_dir, keep=2)
    ctx.final_dir.mkdir(exist_ok=True)
    (ctx.final_dir / "result.json").write_text(json.dumps({"steps": steps}), encoding="utf-8")
    ctx.mark_done(steps=steps)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--step-s", type=float, default=0.5)
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nan-at", type=int, default=None, help="시험용: 이 step의 손실을 NaN으로")
    run_main(train, ap.parse_args())
