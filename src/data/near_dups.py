"""DINOv2 특징으로 test 이미지의 근접 중복 쌍을 찾는다.

처음 계획한 "전역 특징(CLS) 코사인 0.98 이상" 기준은 이 데이터에서 쓸 수 없다. 카메라와 배치가
고정된 장면이라 서로 다른 이미지도 CLS 유사도가 매우 높다(pushpins는 test 쌍의 중앙값이
0.98을 넘어 카테고리 전체가 한 성분으로 묶임). 그래서 다음 기준을 쓴다.

- 지표: 같은 위치 패치 토큰의 평균 코사인 (배치가 다르면 부품이 놓인 패치에서 값이 떨어짐)
- 임계값: 카테고리마다 각 이미지의 최근접 유사도 분포에서 중앙값 + k * 1.4826 * MAD(중앙값 절대편차) (k=6)
  (카테고리마다 기준선이 다름: juice_bottle 최근접 중앙값 약 0.90, pushpins 약 0.73)

같은 카테고리의 test 이미지끼리만 비교하고, splits.py가 결과로 근접 중복을 같은 fold에 모은다.
모든 분할(train/validation/test)의 CLS 임베딩도 저장해 Phase 2 정상 참조 검색에 재사용한다.
이미지는 중앙 crop 없이 종횡비를 유지해 긴 변 448(14의 배수)로 줄인다
(기본 전처리의 중앙 crop은 가로로 긴 splicing_connectors의 양쪽 커넥터를 잘라낸다).

사용법:
    python -m src.data.near_dups --k 6
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

MODEL_ID = "facebook/dinov2-base"
LONG_SIDE = 448
PATCH = 14
# ImageNet 정규화 (DINOv2 전처리 기본값)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
MAD_SCALE = 1.4826


def load_image(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    s = LONG_SIDE / max(w, h)
    nw, nh = (max(PATCH, round(w * s / PATCH) * PATCH), max(PATCH, round(h * s / PATCH) * PATCH))
    arr = np.asarray(img.resize((nw, nh), Image.BICUBIC), dtype=np.float32) / 255.0
    return ((arr - MEAN) / STD).transpose(2, 0, 1)


@torch.no_grad()
def encode(paths: list, keep_patches: np.ndarray, model, device, batch: int = 16):
    """L2 정규화한 CLS 임베딩(np)과, keep_patches가 True인 이미지의 패치 토큰(fp16 tensor).

    같은 카테고리는 해상도가 같아 한 배치로 묶인다.
    """
    cls_out, patch_out = [], []
    for i in range(0, len(paths), batch):
        x = torch.from_numpy(np.stack([load_image(p) for p in paths[i:i + batch]])).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            h = model(pixel_values=x).last_hidden_state
        h = torch.nn.functional.normalize(h.float(), dim=-1)
        cls_out.append(h[:, 0].cpu().numpy())
        keep = torch.tensor(keep_patches[i:i + batch], device=device)
        if keep.any():
            patch_out.append(h[keep, 1:].half())
    patches = torch.cat(patch_out) if patch_out else None
    return np.concatenate(cls_out), patches


@torch.no_grad()
def aligned_patch_similarity(p: torch.Tensor) -> np.ndarray:
    """N x L x D 패치 토큰 → N x N 같은 위치 패치 평균 코사인 (대각선은 -1)."""
    n = len(p)
    sim = torch.empty(n, n, device=p.device)
    for i in range(n):
        sim[i] = (p[i][None] * p).sum(-1).float().mean(-1)
    sim.fill_diagonal_(-1.0)
    return sim.cpu().numpy()


def robust_threshold(sim: np.ndarray, k: float) -> dict:
    nn = sim.max(1)
    med = float(np.median(nn))
    mad = float(MAD_SCALE * np.median(np.abs(nn - med)))
    return {"nn_median": med, "nn_mad": mad, "threshold": med + k * mad}


def main():
    root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=root / "MVTec-LOCO")
    ap.add_argument("--index", type=Path, default=root / "data" / "index.parquet")
    ap.add_argument("--splits", type=Path, default=root / "data" / "splits.csv")
    ap.add_argument("--emb-out", type=Path, default=root / "data" / "emb_dinov2_base_cls.npz")
    ap.add_argument("--out", type=Path, default=root / "data" / "near_dups.csv")
    ap.add_argument("--k", type=float, default=6.0)
    args = ap.parse_args()

    from transformers import AutoModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(MODEL_ID).to(device).eval()

    df = pd.read_parquet(args.index)
    fold = pd.read_csv(args.splits).set_index("image_id")["fold"] if args.splits.exists() else None
    cls_all, pairs, summary = {}, [], []
    for cat, g in df.groupby("category"):
        is_test = (g["split"] == "test").to_numpy()
        cls, patches = encode([args.data_root / p for p in g["image_path"]], is_test, model, device)
        cls_all.update(zip(g["image_id"], cls))

        ids = g["image_id"].to_numpy()[is_test]
        iu = np.triu_indices(len(ids), k=1)
        cls_sim = (cls[is_test] @ cls[is_test].T)[iu]
        sim = aligned_patch_similarity(patches)
        th = robust_threshold(sim, args.k)
        s = sim[iu]
        hit = s >= th["threshold"]
        for a, b, v in zip(ids[iu[0][hit]], ids[iu[1][hit]], s[hit]):
            pairs.append({"category": cat, "image_a": a, "image_b": b, "patch_cos": round(float(v), 5)})
        row = {"category": cat, "n_test": len(ids), "cls_pair_median": float(np.median(cls_sim)),
               "cls_pairs_ge_0.98": int((cls_sim >= 0.98).sum()), **th,
               "patch_pair_max": float(s.max()), "pairs": int(hit.sum())}
        if fold is not None:
            f = fold.loc[ids].to_numpy()
            row["cross_fold_pairs"] = int((hit & (f[iu[0]] != f[iu[1]])).sum())
        summary.append(row)
        del patches
        print(f"{cat}: {len(g)}장 인코딩, test 근접 중복 쌍 {int(hit.sum())}개")

    ids = df["image_id"].to_numpy()
    np.savez(args.emb_out, image_id=ids.astype(str), emb=np.stack([cls_all[i] for i in ids]))
    print(f"저장: {args.emb_out}")
    out = pd.DataFrame(pairs, columns=["category", "image_a", "image_b", "patch_cos"])
    out.to_csv(args.out, index=False, encoding="utf-8")
    print(f"저장: {args.out} (쌍 {len(out)}개, k={args.k})")
    with pd.option_context("display.width", 220):
        print(pd.DataFrame(summary).round(4).to_string(index=False))
    if len(out):
        print(out.sort_values("patch_cos", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
