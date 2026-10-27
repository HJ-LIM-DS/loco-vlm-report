"""M3 실패 분석: 개수 계열 논리 결함 혼동을 해상도·crop·참조 이미지 관점에서 분해 (계획 v5 Phase 3).

개수 계열 = 부품 수가 규격과 다른 그룹 3개 (과일 구성 수량, 칸별 푸시핀 수량, 부품 구성 수량; test 169장).
모든 수치는 held-out 예측(각 이미지는 자기를 학습하지 않은 모델의 예측)의 원출력 기준이다.

1. 그룹별 정확도: Ours 4B(seed 3개 평균) vs B1, 차이가 큰 순. 주요 혼동 쌍(정답 그룹 -> 예측).
2. 해상도: 결함 성분 크기를 검사 이미지 토큰 수로 환산(성분 면적 중앙값 / (이미지 면적 / 256)).
   수량 초과 계열은 GT가 같은 종류 부품을 모두 칠하므로 성분 1개 ~ 부품 1개 크기다.
3. crop: 그 이미지를 예측한 모델 fold의 탐지기 기준으로 crop 상자를 다시 계산해 GT 마스크가 상자 안에
   들어간 비율(포함률)로 나눈다. 탐지 영역이 없으면 전체 이미지가 crop 자리에 들어간다.
4. 참조: 검사 이미지와 정상 참조의 DINOv2 CLS 코사인 유사도 3분위. 2B fold 0 ablation(참조 제거·crop 제거·
   비전 인코더 LoRA)의 개수 계열 효과(fold 1 이미지).
5. 사례 후보: 학습하지 않은 held-out 이미지에서 실패 유형별로 고른 id (PDF 그림4용, M4에서 사용).

    python -m src.eval.failure
"""

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.data.masks import load_defect_masks
from src.detector.cache import FoldCalibration
from src.eval.explanation import load_preds
from src.kb import load_kb
from src.output_spec import UNKNOWN_LABEL, is_normal, parse_report
from src.vlm.build_sft import expand_box

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "MVTec-LOCO"
OUT = ROOT / "reports" / "tables" / "m3_failure"
COUNT_GROUPS = {("breakfast_box", "fruit_count"), ("pushpins", "pin_count"), ("screw_bag", "part_count")}
TOKENS = 256
SIZE_BINS = [(0, 1, "<1"), (1, 4, "1-4"), (4, 16, "4-16"), (16, 1e9, ">=16")]
COVER_BINS = [(-1, 0, "탐지 영역 없음(전체 이미지)"), (0, 0.5, "<50%"), (0.5, 0.999, "50-99%"), (0.999, 2, "100%")]


def heldout(run: str, fold: int) -> Path:
    return ROOT / "runs" / run.format(f=fold) / "final" / "heldout_predictions.jsonl"


def pred_label(text: str, kb_cat: dict) -> str:
    """예측을 그룹ID / '정상' / '미분류' / '파싱 실패'로."""
    if is_normal(text or ""):
        return "정상"
    r = parse_report(text or "")
    if r is None:
        return "파싱 실패"
    if r.group == UNKNOWN_LABEL:
        return "미분류"
    by_ko = {g["ko_label"]: gid for gid, g in kb_cat["groups"].items()}
    return by_ko.get(r.group, "KB 외 그룹")


def predictions(run: str, kb: dict, idx: pd.DataFrame, folds=(0, 1)) -> pd.Series:
    """이상 이미지별 예측 그룹ID (run 이름의 {f}는 학습 fold)."""
    pr = load_preds([heldout(run, f) for f in folds]).set_index("id")
    pr = pr[idx.loc[pr.index, "label"] != "good"]
    return pd.Series({i: pred_label(t, kb[idx.loc[i, "category"]]) for i, t in pr["pred"].items()})


def image_features(idx: pd.DataFrame, anom: pd.Index, splits: pd.DataFrame) -> pd.DataFrame:
    """결함 성분 크기(토큰), crop 포함률. 탐지 맵 기준은 이미지를 예측한 모델의 학습 fold로 정한다."""
    fold_of = splits.set_index("image_id")["fold"]
    calibs = {}
    rows = []
    for iid in anom:
        r = idx.loc[iid]
        masks, _ = load_defect_masks(DATA / r["mask_dir"])
        m = np.zeros_like(next(iter(masks.values())))
        for v in masks.values():
            m |= v
        h, w = m.shape
        n, _, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
        areas = stats[1:, cv2.CC_STAT_AREA] if n > 1 else np.array([0])
        tok_area = h * w / TOKENS
        model_fold = 1 - int(fold_of[iid])
        key = (r["category"], model_fold)
        if key not in calibs:
            side = splits.merge(idx[["category", "label", "split"]], left_on="image_id", right_index=True)
            normal = side[(side["category"] == r["category"]) & (side["split"] == "test") & (side["fold"] == model_fold)
                          & (side["label"] == "good")]["image_id"].tolist()
            calibs[key] = FoldCalibration("salad", r["category"], normal)
        region, _ = calibs[key].region(iid, (h, w))
        if region is None:
            cover, box_frac = -0.5, 1.0
        else:
            ys, xs = np.nonzero(region)
            x0, y0, x1, y1 = expand_box(xs.min(), ys.min(), xs.max() + 1, ys.max() + 1, w, h)
            cover = float(m[y0:y1, x0:x1].sum() / max(m.sum(), 1))
            box_frac = (x1 - x0) * (y1 - y0) / (w * h)
        rows.append({"id": iid, "comp_tokens": float(np.median(areas) / tok_area), "n_components": int(n - 1),
                     "mask_tokens": float(m.sum() / tok_area), "crop_cover": cover, "crop_area_frac": box_frac})
    return pd.DataFrame(rows).set_index("id")


def reference_similarity(idx: pd.DataFrame, anom: pd.Index) -> pd.Series:
    z = np.load(ROOT / "data" / "emb_dinov2_base_cls.npz")
    emb = dict(zip(z["image_id"], z["emb"]))
    ref = {}
    for f in (0, 1):
        for line in (ROOT / "data" / "sft" / f"m2_salad_f{f}" / "heldout.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            ref[r["id"]] = r["reference_id"]
    return pd.Series({i: float(emb[i] @ emb[ref[i]] / (np.linalg.norm(emb[i]) * np.linalg.norm(emb[ref[i]])))
                      for i in anom})


def binned(acc: pd.DataFrame, col: str, bins, by: pd.Series) -> list:
    """구간 [lo, hi)별 결함 종류별 정확도. crop 포함률의 '탐지 영역 없음'은 -0.5로 표시돼 첫 구간에 들어간다."""
    out = []
    for lo, hi, name in bins:
        m = (acc[col] >= lo) & (acc[col] < hi)
        row = {"bin": name}
        for grp in ("개수 계열", "기타 논리", "구조"):
            g = acc[m & (by == grp)]
            row[grp] = {"n": len(g), "ours": float(g["ours"].mean()) if len(g) else None,
                        "b1": float(g["b1"].mean()) if len(g) else None}
        out.append(row)
    return out


def main():
    kb = load_kb()
    idx = pd.read_parquet(ROOT / "data" / "index.parquet").set_index("image_id")
    splits = pd.read_csv(ROOT / "data" / "splits.csv")
    anom_idx = idx[(idx["split"] == "test") & (idx["label"] != "good")]
    gt = anom_idx["groups"].map(lambda g: g[0])
    kind = pd.Series(np.where([(c, g) in COUNT_GROUPS for c, g in zip(anom_idx["category"], gt)], "개수 계열",
                              np.where(anom_idx["label"] == "logical", "기타 논리", "구조")), index=anom_idx.index)

    seeds = {s: predictions(f"m2_4b_f{{f}}_s{s}", kb, idx) for s in (0, 1, 2)}
    b1 = pd.Series({i: pred_label(t, kb[idx.loc[i, "category"]]) for i, t in load_preds(
        [ROOT / "outputs" / "baselines" / f"b1_f{f}_heldout.jsonl" for f in (0, 1)]).set_index("id")["pred"].items()
        if idx.loc[i, "label"] != "good"})
    ids = anom_idx.index
    ok = {s: (p.reindex(ids) == gt) for s, p in seeds.items()}
    acc = pd.DataFrame({"ours": np.mean([ok[s].to_numpy() for s in seeds], axis=0), "ours_s0": ok[0],
                        "b1": (b1.reindex(ids) == gt), "kind": kind, "category": anom_idx["category"], "group": gt},
                       index=ids)
    res, lines = {}, ["# M3 실패 분석 (held-out 원출력, Ours 4B는 seed 3개 평균 정답률)", ""]

    # 1. 결함 종류별·그룹별
    lines += ["## 1. 결함 종류별 정확도", "", "| 종류 | n | Ours 4B | B1 | 차이(Ours - B1, %p) |", "|---|---|---|---|---|"]
    res["by_kind"] = {}
    for k in ("개수 계열", "기타 논리", "구조"):
        g = acc[acc["kind"] == k]
        res["by_kind"][k] = {"n": len(g), "ours": float(g["ours"].mean()), "b1": float(g["b1"].mean())}
        lines.append(f"| {k} | {len(g)} | {100 * g['ours'].mean():.1f} | {100 * g['b1'].mean():.1f} | "
                     f"{100 * (g['ours'].mean() - g['b1'].mean()):+.1f} |")
    gacc = acc.groupby(["category", "group"]).agg(n=("ours", "size"), ours=("ours", "mean"), b1=("b1", "mean"))
    gacc["gap"] = gacc["ours"] - gacc["b1"]
    gacc = gacc[gacc["n"] >= 10].sort_values("gap")
    res["by_group"] = gacc.reset_index().to_dict("records")
    lines += ["", "### 그룹별 (표본 10장 이상, B1 대비 격차가 큰 순 상위 10)", "",
              "| 카테고리 | 그룹 | n | Ours 4B | B1 | 차이(%p) |", "|---|---|---|---|---|---|"]
    for (c, g), r in gacc.head(10).iterrows():
        lines.append(f"| {c} | {kb[c]['groups'][g]['ko_label']} | {int(r['n'])} | {100 * r['ours']:.1f} | "
                     f"{100 * r['b1']:.1f} | {100 * r['gap']:+.1f} |")

    # 혼동 쌍 (seed 0)
    p0 = seeds[0].reindex(ids)
    wrong = pd.DataFrame({"category": anom_idx["category"], "gt": gt, "pred": p0})[p0 != gt]
    conf = wrong.groupby(["category", "gt", "pred"]).size().sort_values(ascending=False)
    name = lambda c, g: kb[c]["groups"][g]["ko_label"] if g in kb[c]["groups"] else g  # noqa: E731
    res["confusions_s0"] = [{"category": c, "gt": g, "pred": p, "n": int(n)} for (c, g, p), n in conf.head(15).items()]
    lines += ["", "### 주요 혼동 (seed 0, 상위 12)", "", "| 카테고리 | 정답 그룹 | 예측 | 장수 |", "|---|---|---|---|"]
    for (c, g, p), n in conf.head(12).items():
        lines.append(f"| {c} | {name(c, g)} | {name(c, p)} | {n} |")
    cnt = wrong[[(c, g) in COUNT_GROUPS for c, g in zip(wrong["category"], wrong["gt"])]]
    res["count_errors_s0"] = cnt["pred"].map(lambda p: p if p in ("정상", "미분류", "파싱 실패") else "다른 그룹").value_counts().to_dict()
    lines += ["", "- 개수 계열 오답(seed 0)의 내용: " + ", ".join(f"{k} {v}" for k, v in res["count_errors_s0"].items())]

    # 2·3. 해상도, crop
    feat = image_features(idx, ids, splits)
    acc = acc.join(feat)
    res["size"] = binned(acc, "comp_tokens", SIZE_BINS, acc["kind"])
    res["crop"] = binned(acc, "crop_cover", COVER_BINS, acc["kind"])
    for title, key, note in (
            ("2. 해상도: 결함 성분 크기(검사 이미지 토큰 수 기준)", "size",
             "성분 크기 = GT 연결 성분 면적 중앙값 / (이미지 면적 / 256). 1 미만이면 부품 하나가 토큰 1개보다 작다."),
            ("3. crop: GT 마스크가 crop 상자에 들어간 비율", "crop",
             "그 이미지를 예측한 모델 fold의 탐지 맵 기준으로 crop 상자를 다시 계산.")):
        lines += ["", f"## {title}", "", note, "",
                  "| 구간 | 개수 계열 n / Ours / B1 | 기타 논리 n / Ours / B1 | 구조 n / Ours / B1 |", "|---|---|---|---|"]
        for r in res[key]:
            cells = []
            for grp in ("개수 계열", "기타 논리", "구조"):
                v = r[grp]
                cells.append("0" if not v["n"] else f"{v['n']} / {100 * v['ours']:.1f} / {100 * v['b1']:.1f}")
            lines.append(f"| {r['bin']} | " + " | ".join(cells) + " |")
    cg = acc[acc["kind"] == "개수 계열"]
    res["count_by_category"] = {c: {"n": len(g), "comp_tokens_median": float(g["comp_tokens"].median()),
                                    "crop_cover_median": float(g["crop_cover"].median()), "ours": float(g["ours"].mean())}
                                for c, g in cg.groupby("category")}
    lines += ["", "- 개수 계열 카테고리별 성분 크기 중앙값(토큰) / crop 포함률 중앙값 / Ours: " + "; ".join(
        f"{c} {v['comp_tokens_median']:.2f} / {100 * v['crop_cover_median']:.0f}% / {100 * v['ours']:.1f}"
        for c, v in res["count_by_category"].items())]

    # 4. 참조 유사도와 ablation
    acc["ref_sim"] = reference_similarity(idx, ids)
    # 유사도 수준이 카테고리마다 달라(고정 배치 장면일수록 높음) 전체 분위는 카테고리 효과와 섞인다 -> 카테고리 안 분위
    rank = acc.groupby("category")["ref_sim"].rank(pct=True)
    tert = pd.cut(rank, [0, 1 / 3, 2 / 3, 1], labels=["하", "중", "상"], include_lowest=True)
    res["ref_sim"] = {str(t): {k: float(acc[(tert == t) & (acc["kind"] == k)]["ours"].mean()) for k in ("개수 계열", "기타 논리", "구조")}
                      for t in ("하", "중", "상")}
    lines += ["", "## 4. 정상 참조", "", "검사 이미지-참조 DINOv2 유사도의 카테고리 내 3분위별 Ours 정확도", "",
              "| 유사도 | 개수 계열 | 기타 논리 | 구조 |", "|---|---|---|---|"]
    for t, v in res["ref_sim"].items():
        lines.append(f"| {t} | " + " | ".join(f"{100 * v[k]:.1f}" for k in ("개수 계열", "기타 논리", "구조")) + " |")

    abl = {"2B 기본": "m2_2b_f{f}_s0", "정상 참조 제거": "m2_2b_f{f}_noref", "crop 제거": "m2_2b_f{f}_nocrop",
           "비전 인코더 LoRA": "m3_2b_f{f}_vislora"}
    f1_ids = pd.Index([i for i in ids if splits.set_index("image_id").loc[i, "fold"] == 1])
    res["ablation_2b_fold1"] = {}
    lines += ["", "2B fold 0 모델 -> fold 1 이미지, 결함 종류별 정확도", "", "| 설정 | 개수 계열 | 기타 논리 | 구조 |", "|---|---|---|---|"]
    for nm, run in abl.items():
        if not heldout(run, 0).exists():
            continue
        p = predictions(run, kb, idx, folds=(0,)).reindex(f1_ids)
        okv = p == gt.reindex(f1_ids)
        row = {k: float(okv[kind.reindex(f1_ids) == k].mean()) for k in ("개수 계열", "기타 논리", "구조")}
        res["ablation_2b_fold1"][nm] = row
        lines.append(f"| {nm} | " + " | ".join(f"{100 * row[k]:.1f}" for k in ("개수 계열", "기타 논리", "구조")) + " |")
    nk = (kind.reindex(f1_ids) == "개수 계열").sum()
    lines.append(f"\n(fold 1 개수 계열 {nk}장)")

    # 5. 사례 후보 (seed 0, 각 이미지는 자기를 학습하지 않은 모델의 예측)
    def pick(mask, n=3):
        """조건에 맞는 이미지를 카테고리가 겹치지 않게 id 순으로 n장 (세 seed 모두 같은 정오인 것 우선)."""
        g = acc[mask].assign(stable=lambda d: d["ours"].isin([0.0, 1.0])).sort_values(["stable", "category"],
                                                                                      ascending=[False, True])
        return g.groupby("category").head(1).head(n).index.tolist()

    known = p0.reindex(acc.index).isin([g for c in kb for g in kb[c]["groups"]])
    cases = {"성공_논리": pick(acc["ours_s0"] & (acc["kind"] != "구조")),
             "성공_구조": pick(acc["ours_s0"] & (acc["kind"] == "구조")),
             "실패_개수_다른그룹": pick(~acc["ours_s0"] & (acc["kind"] == "개수 계열") & known),
             "실패_작은결함": pick(~acc["ours_s0"] & (acc["comp_tokens"] < 1)),
             "실패_crop_놓침": pick(~acc["ours_s0"] & (acc["crop_cover"] == 0))}
    res["case_candidates"] = cases
    lines += ["", "## 5. 사례 후보 (PDF 그림4, held-out)", ""] + [f"- {k}: {', '.join(v)}" for k, v in cases.items()]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    acc.to_csv(OUT.with_suffix(".csv"), encoding="utf-8")
    OUT.with_suffix(".json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    OUT.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"저장: {OUT.with_suffix('.md')}")


if __name__ == "__main__":
    main()
