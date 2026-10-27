"""B1 비-VLM 진단기: DINOv2 전역 특징 + crop DINOv2 특징 + SALAD 점수 3항 -> 로지스틱 회귀 -> KB 2줄.

- 카테고리별 다항 로지스틱 회귀. 클래스 = 세부 결함 조합(예: "1_additional_nut") + "good".
  학습 데이터는 fold의 학습 측 전체(train + inner-val), 평가는 held-out fold.
- 위치: 예측한 세부 결함의 loc_mode에 따라 point는 SALAD 맵 최대 응답 칸, global은 "전체", instances는 생략.
- 입력 이미지(검사 이미지 전역 임베딩, crop)는 주력 모델과 같은 SFT 레코드에서 가져온다.

사용법:
    python -m src.baselines.crop_classifier --sft data/sft/m2_salad_f0 --out outputs/baselines/b1_f0_heldout.jsonl
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.kb import defect_table, load_kb
from src.output_spec import LOC_GLOBAL, NORMAL, format_report, gt_report, resolve_loc_mode

ROOT = Path(__file__).resolve().parents[2]


@torch.no_grad()
def crop_embeddings(paths: list, device) -> np.ndarray:
    from transformers import AutoModel

    from src.data.near_dups import MODEL_ID, load_image

    model = AutoModel.from_pretrained(MODEL_ID).to(device).eval()
    out = []
    for p in paths:
        x = torch.from_numpy(load_image(ROOT / p))[None].to(device)
        h = model(pixel_values=x).last_hidden_state[:, 0]
        out.append(torch.nn.functional.normalize(h.float(), dim=-1).cpu().numpy()[0])
    return np.stack(out)


def features(rows: list, emb: dict, scores: pd.DataFrame, crop_emb: np.ndarray) -> np.ndarray:
    s = scores.set_index("image_id")
    glob = np.stack([emb[r["id"]] for r in rows])
    terms = s.loc[[r["id"] for r in rows], ["s_local", "s_mah", "s_comp"]].to_numpy()
    return np.hstack([glob, crop_emb, terms])


def run(args):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    kb = load_kb()
    sft = Path(args.sft)
    tr = [json.loads(x) for f in ("train", "val") for x in (sft / f"{f}.jsonl").read_text(encoding="utf-8").splitlines()]
    tr = [r for r in tr if not r.get("dropped")]  # 목록 드롭아웃 정답(미분류)은 B1 학습에 쓰지 않음
    te = [json.loads(x) for x in (sft / "heldout.jsonl").read_text(encoding="utf-8").splitlines()]
    idx = pd.read_parquet(ROOT / "data" / "index.parquet").set_index("image_id")
    z = np.load(ROOT / "data" / "emb_dinov2_base_cls.npz")
    emb = dict(zip(z["image_id"], z["emb"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = []
    for cat in sorted({r["category"] for r in te}):
        a = [r for r in tr if r["category"] == cat]
        b = [r for r in te if r["category"] == cat]
        scores = pd.read_csv(ROOT / "outputs" / "detector" / "salad" / cat / "scores.csv")
        Xa = features(a, emb, scores, crop_embeddings([r["images"][1] for r in a], device))
        Xb = features(b, emb, scores, crop_embeddings([r["images"][1] for r in b], device))
        ya = ["good" if r["label"] == "good" else "+".join(idx.loc[r["id"], "defects"]) for r in a]
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=args.C, max_iter=3000, class_weight="balanced"))
        clf.fit(Xa, ya)
        table = defect_table(kb, cat)
        for r, y in zip(b, clf.predict(Xb)):
            if y == "good":
                pred = NORMAL
            else:
                names = y.split("+")
                mode = resolve_loc_mode(table[n]["loc_mode"] for n in names)
                loc = {"point": (r.get("detector") or {}).get("location"), "global": LOC_GLOBAL,
                       "instances": None}[mode]
                pred = format_report(gt_report(kb, cat, names, loc))
            out.append({"id": r["id"], "target": r["target"], "pred": pred})
        print(f"{cat}: 학습 {len(a)} / 평가 {len(b)}, 클래스 {len(set(ya))}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")


def main():
    from ops.train_hooks import RunContext, run_main

    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--C", type=float, default=1.0)
    args = ap.parse_args()

    def task(a):
        ctx = RunContext.from_env("b1")
        ctx.heartbeat(step=0, force=True, phase="fit")
        run(a)
        ctx.mark_done()

    run_main(task, args)


if __name__ == "__main__":
    main()
