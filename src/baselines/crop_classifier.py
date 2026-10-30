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


def make_clf(C: float):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000, class_weight="balanced"))


def class_probs(clf, X: np.ndarray) -> list:
    """행별 {클래스: 확률} (소수 4자리)."""
    P = clf.predict_proba(X)
    return [{c: round(float(p), 4) for c, p in zip(clf.classes_, row)} for row in P]


def oof_probs(Xa: np.ndarray, ya: list, C: float, n_splits: int = 5, seed: int = 0) -> list:
    """학습 측 out-of-fold 확률: 층화 K-fold로 각 행을 자기를 학습하지 않은 모델이 예측한다.
    (2차 개선: 캐스케이드 B1 심판 J2의 임계값과 P3 변형 B의 학습 측 힌트는 in-sample 확률을 쓰면 안 되므로)"""
    import warnings

    from sklearn.model_selection import StratifiedKFold

    out = [None] * len(ya)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # 표본 5장 미만 클래스 경고 (층화는 가능한 범위에서만)
        for tr_i, te_i in StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(Xa, ya):
            clf = make_clf(C)
            clf.fit(Xa[tr_i], [ya[i] for i in tr_i])
            for i, p in zip(te_i, class_probs(clf, Xa[te_i])):
                out[i] = p
    return out


def run(args):
    kb = load_kb()
    sft = Path(args.sft)
    tr_all = [json.loads(x) for f in ("train", "val") for x in (sft / f"{f}.jsonl").read_text(encoding="utf-8").splitlines()]
    tr = [r for r in tr_all if not r.get("dropped")]  # 목록 드롭아웃 정답(미분류)은 B1 학습에 쓰지 않음
    dropped = [r for r in tr_all if r.get("dropped")]
    te = [json.loads(x) for x in (sft / "heldout.jsonl").read_text(encoding="utf-8").splitlines()]
    idx = pd.read_parquet(ROOT / "data" / "index.parquet").set_index("image_id")
    z = np.load(ROOT / "data" / "emb_dinov2_base_cls.npz")
    emb = dict(zip(z["image_id"], z["emb"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out, probs_out, oof_out = [], [], []
    for cat in sorted({r["category"] for r in te}):
        a = [r for r in tr if r["category"] == cat]
        b = [r for r in te if r["category"] == cat]
        d = [r for r in dropped if r["category"] == cat]
        scores = pd.read_csv(ROOT / "outputs" / "detector" / "salad" / cat / "scores.csv")
        Xa = features(a, emb, scores, crop_embeddings([r["images"][1] for r in a], device))
        Xb = features(b, emb, scores, crop_embeddings([r["images"][1] for r in b], device))
        ya = ["good" if r["label"] == "good" else "+".join(idx.loc[r["id"], "defects"]) for r in a]
        clf = make_clf(args.C)
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
        if args.probs_out:
            # held-out: 학습 측 전체로 학습한 모델의 확률. 학습 측: OOF 확률 (드롭아웃 행은 학습에 없으므로 전체 모델로)
            for r, p in zip(b, class_probs(clf, Xb)):
                probs_out.append({"id": r["id"], "role": "heldout", "p_good": p.get("good", 0.0),
                                  "argmax": max(p, key=p.get), "probs": p})
            for r, p in zip(a, oof_probs(Xa, ya, args.C)):
                oof_out.append({"id": r["id"], "role": "oof", "p_good": p.get("good", 0.0), "argmax": max(p, key=p.get), "probs": p})
            if d:
                Xd = features(d, emb, scores, crop_embeddings([r["images"][1] for r in d], device))
                for r, p in zip(d, class_probs(clf, Xd)):
                    oof_out.append({"id": r["id"], "role": "dropped", "p_good": p.get("good", 0.0),
                                    "argmax": max(p, key=p.get), "probs": p})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    if args.probs_out:
        po = Path(args.probs_out)
        po.parent.mkdir(parents=True, exist_ok=True)
        po.write_text("".join(json.dumps(o, ensure_ascii=False) + "\n" for o in probs_out), encoding="utf-8")
        po.with_name(po.stem + ".oof.jsonl").write_text("".join(json.dumps(o, ensure_ascii=False) + "\n" for o in oof_out),
                                                         encoding="utf-8")
        print(f"확률 저장: {po} ({len(probs_out)}장), OOF {len(oof_out)}장")


def main():
    from ops.train_hooks import RunContext, run_main

    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--probs-out", default=None,
                    help="클래스 확률 jsonl (held-out) + 같은 이름 .oof.jsonl (학습 측 5-fold OOF). 2차 개선용")
    args = ap.parse_args()

    def task(a):
        ctx = RunContext.from_env("b1")
        ctx.heartbeat(step=0, force=True, phase="fit")
        run(a)
        ctx.mark_done()

    run_main(task, args)


if __name__ == "__main__":
    main()
