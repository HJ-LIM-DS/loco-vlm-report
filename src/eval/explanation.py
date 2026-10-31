"""2줄 리포트 채점. 주력 모델과 기준선 B0~B3이 같은 함수로 채점된다.

입력: 예측 jsonl 1개 이상 ({"id", "pred"} 필수). 두 held-out fold의 예측을 합쳐 넣으면
각 이미지는 자기를 학습하지 않은 모델의 예측으로 채점된다.

이상 이미지(oracle 게이팅, 정답 이상만 입력한 것과 같음)
- format_ok: 파서 통과율 / normal_miss: "정상"이라 답한 비율 / unknown_rate: 미분류 응답 비율
- group_acc, macro_f1(표본 10장 이상 그룹만), type_acc, detail_exact(세부 집합 일치)
- detail_exact_by_freq: 세부 결함 빈도 구간별 (전체 test 기준 <10, 10~29, 30 이상)
- location (point 결함만): exact(칸 일치), hit(예측 칸에 GT 마스크 10% 이상)
- action_consistency: 원출력 조치가 예측 그룹의 KB 조치와 같은 비율 (1 - 조치 환각률)
- severity_miss: GT가 라인정지 결함인데 라인정지가 아닌 그룹(또는 정상·미분류)으로 답한 비율
정상 이미지: normal_correct ("정상" 응답 비율, 오탐 거부 능력)
모든 지표는 원출력(raw)과 Guard 후(guarded) 두 가지로 낸다.

사용법:
    python -m src.eval.explanation --pred runs/m2_4b_f0_s0/final/heldout_predictions.jsonl \
        runs/m2_4b_f1_s0/final/heldout_predictions.jsonl --out reports/tables/m2_4b_s0.json
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from src.kb import defect_table, load_kb
from src.output_spec import CELLS, UNKNOWN_ACTION, UNKNOWN_LABEL, is_normal, parse_report
from src.vlm.guard import guard

ROOT = Path(__file__).resolve().parents[2]
HIT_FRAC = 0.10
FREQ_BINS = [(0, 10, "<10"), (10, 30, "10-29"), (30, 10**9, ">=30")]
MIN_GROUP_F1 = 10


def load_preds(paths) -> pd.DataFrame:
    rows = []
    for p in paths:
        rows += [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]
    df = pd.DataFrame(rows)
    if df["id"].duplicated().any():
        raise ValueError("같은 이미지 예측이 중복됨 (fold 예측 파일이 겹침)")
    return df


def _macro_f1(gt: list, pr: list, groups: list) -> float | None:
    if not groups:
        return None
    f1s = []
    for g in groups:
        tp = sum(a == g and b == g for a, b in zip(gt, pr))
        fp = sum(a != g and b == g for a, b in zip(gt, pr))
        fn = sum(a == g and b != g for a, b in zip(gt, pr))
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s))


def score_texts(df: pd.DataFrame, texts: pd.Series, kb: dict) -> dict:
    """df: 인덱스와 합친 예측 표(이미지당 1행), texts: 채점할 출력(원출력 또는 Guard 후)."""
    anom = df["label"] != "good"
    out = {"n_anomaly": int(anom.sum()), "n_normal": int((~anom).sum())}
    good = texts[~anom]
    out["normal_correct"] = float(good.map(is_normal).mean()) if len(good) else None

    a = df[anom]
    t = texts[anom]
    parsed = t.map(lambda x: None if is_normal(x) else parse_report(x))
    gtp = a["gt_report"].map(parse_report)
    out["format_ok"] = float((t.map(is_normal) | parsed.notna()).mean())
    out["normal_miss"] = float(t.map(is_normal).mean())
    out["unknown_rate"] = float(parsed.map(lambda r: r is not None and r.group == UNKNOWN_LABEL).mean())

    def eq(f):
        return [p is not None and f(p, g) for p, g in zip(parsed, gtp)]

    grp = eq(lambda p, g: p.group == g.group)
    out["group_acc"] = float(np.mean(grp))
    out["type_acc"] = float(np.mean(eq(lambda p, g: p.type == g.type)))
    det = eq(lambda p, g: set(p.details) == set(g.details))
    out["detail_exact"] = float(np.mean(det))
    gt_groups = [g.group for g in gtp]
    pr_groups = [p.group if p is not None else "<none>" for p in parsed]
    big = [g for g, c in Counter(gt_groups).items() if c >= MIN_GROUP_F1]
    out["macro_f1"] = _macro_f1(gt_groups, pr_groups, big)
    out["macro_f1_groups"] = len(big)

    freq = Counter(d for ds in a["defects"] for d in ds)  # 이 평가 집합 기준 세부 결함 빈도
    by = {}
    for lo, hi, name in FREQ_BINS:
        m = [lo <= min(freq[d] for d in ds) < hi for ds in a["defects"]]
        sel = [x for x, k in zip(det, m) if k]
        by[name] = {"n": len(sel), "acc": float(np.mean(sel)) if sel else None}
    out["detail_exact_by_freq"] = by

    pt = (a["loc_mode"] == "point").to_numpy()
    exact, hit = [], []
    for p, g, frac, is_pt in zip(parsed, gtp, a["point_cell_frac"], pt):
        if not is_pt:
            continue
        loc = p.location if p is not None else None
        exact.append(loc == g.location)
        hit.append(loc in CELLS and frac is not None and frac[CELLS.index(loc)] >= HIT_FRAC)
    out["location_exact"] = float(np.mean(exact)) if exact else None
    out["location_hit"] = float(np.mean(hit)) if hit else None
    out["location_n"] = len(exact)

    acts = []
    for p, cat in zip(parsed, a["category"]):
        if p is None:
            continue
        groups = {g["ko_label"]: g for g in kb[cat]["groups"].values()}
        want = UNKNOWN_ACTION if p.group == UNKNOWN_LABEL else (groups[p.group]["action"] if p.group in groups else None)
        acts.append(want is not None and p.action == want)
    out["action_consistency"] = float(np.mean(acts)) if acts else None

    sev_miss = []
    for p, g, cat in zip(parsed, gtp, a["category"]):
        groups = {x["ko_label"]: x for x in kb[cat]["groups"].values()}
        if groups.get(g.group, {}).get("severity") != "라인정지":
            continue
        pg = groups.get(p.group) if p is not None else None
        sev_miss.append(pg is None or pg.get("severity") != "라인정지")
    out["severity_miss"] = float(np.mean(sev_miss)) if sev_miss else None
    out["severity_n"] = len(sev_miss)

    out["by_type"] = {}
    for lab in ("logical", "structural"):
        m = (a["label"] == lab).to_numpy()
        out["by_type"][lab] = {"n": int(m.sum()), "group_acc": float(np.mean(np.array(grp)[m])) if m.any() else None,
                               "detail_exact": float(np.mean(np.array(det)[m])) if m.any() else None}
    out["by_category"] = {c: float(np.mean(np.array(grp)[(a["category"] == c).to_numpy()]))
                          for c in sorted(a["category"].unique())}
    return out


def evaluate(pred_paths, kb: dict | None = None) -> dict:
    kb = kb or load_kb()
    idx = pd.read_parquet(ROOT / "data" / "index.parquet")
    pr = load_preds(pred_paths)
    df = pr.merge(idx, left_on="id", right_on="image_id", how="left", validate="one_to_one")
    if df["image_id"].isna().any():
        raise ValueError("인덱스에 없는 예측 id")
    raw = df["pred"].fillna("")
    steps = Counter()
    guarded = []
    for text, cat in zip(raw, df["category"]):
        g, s = guard(text, kb, cat, None)
        guarded.append(g)
        steps.update(s)
    guarded = pd.Series(guarded, index=df.index)
    n_anom = int((df["label"] != "good").sum())
    return {"files": [str(p) for p in pred_paths], "n": len(df),
            "raw": score_texts(df, raw, kb), "guarded": score_texts(df, guarded, kb),
            "guard_step_rate": {k: v / len(df) for k, v in steps.items()}, "n_anomaly": n_anom}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    m = evaluate(a.pred)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
    r, g = m["raw"], m["guarded"]
    print(f"이상 {r['n_anomaly']} / 정상 {r['n_normal']}")
    for k in ("format_ok", "group_acc", "macro_f1", "type_acc", "detail_exact", "location_hit", "action_consistency",
              "severity_miss", "normal_correct"):
        print(f"{k:20s} 원출력 {r[k]}  Guard 후 {g[k]}")


if __name__ == "__main__":
    main()
