"""EfficientAD-M (anomalib 2.7) 학습과 탐지 캐시 내보내기. 감독자(ops.supervisor) 규약을 따른다.

- 학습: train/good, 256x256 입력, batch 1, 70k step (원 논문 설정). validation/good은 마지막 epoch에서
  맵 정규화 분위수 계산에만 쓰인다(그래서 validation 점수는 정규화에 이미 쓰인 표본의 값(in-sample)이라 임계값 추정에서 제외).
- 체크포인트: every_n_steps마다 원자적 저장 + 완료 표식, 최근 2개 유지, 재개는 Lightning ckpt_path.
- 내보내기: 모델 원출력(이미지 점수 = 맵 최댓값, 맵 256x256)을 outputs/detector/efficientad/<category>/에
  저장한다 (src.detector.cache 형식). anomalib 후처리 정규화는 쓰지 않는다(보정은 fold별로 따로 함).
- anomalib 2.7의 LOCO 로더는 pandas 3과 맞지 않아 future.infer_string을 끈다.
- LOCO_SMOKE=1: 20 step, 10 step마다 저장, 이미지 4장만 내보내기.

사용법:
    python -m src.detector.efficientad --category breakfast_box
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option("future.infer_string", False)

import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torchvision.transforms.v2 import functional as TF  # noqa: E402

from ops.train_hooks import RunContext, latest_valid_checkpoint, lightning_callback, run_main  # noqa: E402
from src.detector.cache import det_dir, map_key  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "MVTec-LOCO"
DETECTOR = "efficientad"
IMAGE_SIZE = 256


def validate_ckpt(path: Path) -> None:
    torch.load(path / "model.ckpt", map_location="cpu", weights_only=False)


def auroc(y, s) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


@torch.no_grad()
def export(model, category: str, out: Path, limit: int | None = None) -> dict:
    """test 전체 + validation/good의 점수와 맵을 저장하고 image AUROC와 지연을 잰다."""
    df = pd.read_parquet(ROOT / "data" / "index.parquet")
    df = df[(df["category"] == category) & (df["split"].isin(["test", "validation"]))]
    if limit:
        df = df.head(limit)
    (out / "maps").mkdir(parents=True, exist_ok=True)
    net = model.model.eval().cuda()
    rows, times = [], []
    for r in df.itertuples():
        img = TF.to_image(Image.open(DATA / r.image_path).convert("RGB"))
        x = TF.to_dtype(TF.resize(img, [IMAGE_SIZE, IMAGE_SIZE], antialias=True), torch.float32, scale=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        res = net(x[None].cuda())
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        amap = res.anomaly_map[0].squeeze().float().cpu().numpy()
        np.save(out / "maps" / f"{map_key(r.image_id)}.npy", amap.astype(np.float16))
        rows.append({"image_id": r.image_id, "split": r.split, "label": r.label, "score": float(res.pred_score[0])})
    sc = pd.DataFrame(rows)
    sc[["image_id", "score"]].to_csv(out / "scores.csv", index=False)
    test = sc[sc["split"] == "test"]
    meta = {"n": len(sc), "latency_ms_median": round(1000 * float(np.median(times[1:] or times)), 2)}
    if test["label"].nunique() > 1:
        y = (test["label"] != "good").astype(int)
        meta["auroc_all"] = auroc(y, test["score"])
        for lab in ("logical", "structural"):
            t = test[test["label"].isin(["good", lab])]
            meta[f"auroc_{lab}"] = auroc((t["label"] != "good").astype(int), t["score"])
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return meta


def train(args):
    from anomalib.callbacks.checkpoint import ModelCheckpoint
    from anomalib.data import MVTecLOCO
    from anomalib.engine import Engine
    from anomalib.models import EfficientAd
    from anomalib.utils.path import get_datasets_dir

    ctx = RunContext.from_env(f"effad_{args.category}")
    max_steps = 20 if ctx.smoke else args.steps
    torch.manual_seed(args.seed)
    dm = MVTecLOCO(root=DATA, category=args.category, train_batch_size=1, eval_batch_size=16,
                   num_workers=0 if ctx.smoke else args.workers, seed=args.seed)
    model = EfficientAd(imagenet_dir=get_datasets_dir() / "imagenette", model_size="medium",
                        evaluator=False, visualizer=False)
    n_train = len(list((DATA / args.category / "train" / "good").glob("*.png")))
    epochs = max(1, -(-max_steps // n_train))
    ck = latest_valid_checkpoint(ctx.ckpt_dir, validate_ckpt)
    ctx.note_resume(ck)
    engine = Engine(
        max_steps=max_steps, max_epochs=epochs, check_val_every_n_epoch=epochs,
        num_sanity_val_steps=0, logger=False, default_root_dir=ctx.run_dir / "lightning",
        callbacks=[
            # Engine이 기본으로 붙이는 매 epoch 저장을 막으려고 저장을 끈 ModelCheckpoint를 넘긴다
            ModelCheckpoint(dirpath=ctx.run_dir / "lightning", save_top_k=0, save_last=False),
            lightning_callback(ctx, every_n_steps=10 if ctx.smoke else args.save_every_steps),
        ],
        accelerator="gpu", devices=1,
    )
    t0 = time.time()
    engine.fit(model=model, datamodule=dm, ckpt_path=str(ck / "model.ckpt") if ck else None)
    if ctx.smoke or engine.trainer.global_step < max_steps or not model.model.is_set(model.model.quantiles):
        # 마지막 epoch 검증(맵 분위수 계산)이 돌지 않았으면 직접 실행한다
        engine.validate(model=model, datamodule=dm)
    train_s = time.time() - t0

    ctx.heartbeat(step=engine.trainer.global_step, force=True, phase="export")
    ctx.final_dir.mkdir(parents=True, exist_ok=True)
    engine.trainer.save_checkpoint(str(ctx.final_dir / "model.ckpt"))
    out = det_dir(DETECTOR, args.category) if not ctx.smoke else ctx.run_dir / "smoke_export"
    meta = export(model, args.category, out, limit=4 if ctx.smoke else None)
    meta.update(steps=engine.trainer.global_step, train_seconds_this_attempt=round(train_s, 1),
                peak_vram_gb=round(torch.cuda.max_memory_allocated() / 1024**3, 2))
    (ctx.final_dir / "summary.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    ctx.mark_done(**meta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", required=True)
    ap.add_argument("--steps", type=int, default=70000)
    ap.add_argument("--save-every-steps", type=int, default=5000)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    run_main(train, ap.parse_args())


if __name__ == "__main__":
    main()
