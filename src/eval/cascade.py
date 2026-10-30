"""RQ4 캐스케이드 분석: 탐지기(SALAD) 단독 vs 탐지기 + VLM 거부 단계, 이상 비율(π) 보정 호출률.

VLM은 학습 후 held-out의 모든 이미지에 원출력을 냈으므로(train_lora --predict-heldout), 캐스케이드를
GPU 없이 다시 계산할 수 있다. fold f 모델의 held-out은 다른 fold 이미지이고, 임계값은 fold f 학습 측으로 정한다.

    score <  tau_low             -> 무출력
    tau_low <= score < tau_high  -> VLM이 "정상"이면 무출력(오탐 거부)
    score >= tau_high            -> VLM 출력 (정상이라 해도 미분류 템플릿)

지표
- 탐지기 단독(score >= tau_low)과 캐스케이드(최종 출력 있음)의 재현율·FPR, 오경보 감소율, 재현율 손실
  (RQ4 가설: 재현율 손실 1%p 이하로 오경보를 절반 이상 줄인다)
- VLM 호출률 π 보정: π·TPR_det + (1-π)·FPR_det (π = 1%, 5%, test 비율)
- tau_low 민감도: 학습 측 재현율 목표 0.90 / 0.95 / 0.98 / 0.99
- 종단 그룹 정확도(이상 이미지, 최종 출력 기준)
- 종단 F1 (성공 기준: Ours >= B2 VLM 단독). 최종 출력(Guard 후) 기준 두 가지
  - 탐지 F1: 출력 있음 = 이상 판정
  - 정답 그룹 F1: 정밀도 = 정답 그룹 출력 / 전체 출력, 재현율 = 정답 그룹 출력 / 이상 이미지
  - π 보정: 정밀도 = π·R / (π·TPR + (1-π)·FPR). test 비율과 π = 1%, 5%. 차이는 이상·정상을 따로 다시 뽑는
    층화 paired bootstrap 95% CI (같은 이미지 집합).
- 오류 분해(표2): 이상 이미지는 탐지기 놓침 / 판정 구간 VLM 오거부 / 그룹 오답(구간별) / 정답,
  정상 이미지는 통과 / VLM 거부 / 판정 구간 오경보 / 설명 구간 오경보. 탐지기 놓침 중 VLM 원출력이
  맞았던 수(oracle 게이팅이었다면 맞혔을 이미지)도 센다.
- 점수 구간별 VLM "정상" 응답률: VLM은 held-out 전 이미지에 원출력을 냈으므로 통과 구간까지 볼 수 있다.

    python -m src.eval.cascade
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.detector.cache import load_scores
from src.detector.calibrate import fold_thresholds
from src.eval.explanation import load_preds
from src.kb import load_kb
from src.output_spec import is_normal, parse_report
from src.pipeline import decide, finalize
from src.vlm.guard import guard

ROOT = Path(__file__).resolve().parents[2]
CATS = ["breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors"]
PIS = (0.01, 0.05)
RECALLS = (0.90, 0.95, 0.98, 0.99)
# 백분위 100 = 학습 측 정상 최댓값 이상. 정상 표본이 카테고리·fold당 47~69장이라 99~100 사이 값은 나오지 않는다
PCT_BINS = [(0, 50, "<50"), (50, 90, "50-90"), (90, 100, "90-100 미만"), (100, 1e9, "100")]
N_BOOT = 5000


def train_side_percentile(scores: pd.Series, index: pd.DataFrame, splits: pd.DataFrame, fold: int, cat: str):
    """build_sft의 탐지기 백분위와 같은 기준: 학습 fold 측 test/good 점수 분포 대비 백분위."""
    df = index.merge(splits[["image_id", "fold"]], on="image_id")
    ids = df[(df["category"] == cat) & (df["split"] == "test") & (df["fold"] == fold) & (df["label"] == "good")]["image_id"]
    normal = np.sort(scores.loc[ids].to_numpy())
    return lambda v: 100.0 * np.searchsorted(normal, v, side="right") / len(normal)


def group_correct(final: str, gt_report: str) -> bool:
    p = parse_report(final) if final else None
    return p is not None and p.group == parse_report(gt_report).group


def judge_final(stage: str, pred: str, kb: dict, cat: str, judge_normal: bool | None) -> str:
    """판정 구간의 거부를 외부 심판(B1)에 맡길 때의 최종 출력 (2차 개선 J1·J2).
    judge_normal=None이면 기존 규칙(VLM이 "정상"이면 거부). 심판이 이상이라 하면 설명 구간과 같은 규칙(거부 불가)."""
    if stage != "judge" or judge_normal is None:
        return finalize(stage, pred, kb, cat, None)[0]
    return "" if judge_normal else finalize("describe", pred, kb, cat, None)[0]


def load_judge(path: Path) -> pd.DataFrame:
    """crop_classifier --probs-out 파일 -> id 인덱스, p_good·argmax."""
    df = pd.read_json(path, lines=True)
    return df.set_index("id")[["p_good", "argmax"]]


def threshold_from_pgood(p_good: np.ndarray, pass_frac: float = 0.97) -> float:
    """J2 임계값: 판정 구간 이상(학습 측 OOF)의 pass_frac이 거부되지 않도록 p(good) 상위 분위수. 비면 1.0(거부 없음)."""
    p = np.asarray(p_good, dtype=float)
    if len(p) == 0:
        return 1.0
    return float(np.quantile(p, pass_frac, method="higher"))


def oof_threshold(oof_path: Path, fold: int, recall: float, index: pd.DataFrame, splits: pd.DataFrame,
                  pass_frac: float = 0.97) -> dict:
    """fold 학습 측 OOF 확률에서 판정 구간 이상만 골라 J2 임계값을 정한다 (held-out 미사용)."""
    scores = pd.concat([load_scores("salad", c) for c in CATS]).set_index("image_id")["score"]
    idx = index.set_index("image_id")
    th = {c: fold_thresholds("salad", c, index, splits, fold, recall) for c in CATS}
    oof = pd.read_json(oof_path, lines=True).set_index("id")
    sel = []
    for iid, row in oof.iterrows():
        cat = idx.loc[iid, "category"]
        if idx.loc[iid, "label"] != "good" and decide(float(scores.loc[iid]), th[cat]) == "judge":
            sel.append(float(row["p_good"]))
    return {"threshold": threshold_from_pgood(np.array(sel), pass_frac), "n_judge_anomaly": len(sel), "pass_frac": pass_frac}


def cascade(models: dict, recall: float, kb: dict, index: pd.DataFrame, splits: pd.DataFrame,
            judge: dict | None = None, judge_rule: str = "argmax", judge_thr: dict | None = None) -> pd.DataFrame:
    """models: {학습 fold: 예측 jsonl}. 이미지별 단계·최종 출력 표.
    judge: {fold: load_judge(...)}를 주면 판정 구간의 거부를 B1에 맡긴다(judge_rule 'argmax': argmax=="good"이면 거부,
    'threshold': p_good >= judge_thr[fold]이면 거부). judge_normal 열에 실제로 거부 판단을 내린 쪽의 결정을 적는다."""
    scores = pd.concat([load_scores("salad", c) for c in CATS]).set_index("image_id")["score"]
    idx = index.set_index("image_id")
    rows = []
    for fold, path in models.items():
        th = {c: fold_thresholds("salad", c, index, splits, fold, recall) for c in CATS}
        pct = {c: train_side_percentile(scores, index, splits, fold, c) for c in CATS}
        pr = load_preds([path])
        jd = judge.get(fold) if judge else None
        for r in pr.itertuples():
            cat = idx.loc[r.id, "category"]
            stage = decide(float(scores.loc[r.id]), th[cat])
            vn = is_normal(r.pred)
            jn = None
            if jd is not None and stage == "judge":
                jn = bool(jd.loc[r.id, "argmax"] == "good") if judge_rule == "argmax" \
                    else bool(float(jd.loc[r.id, "p_good"]) >= judge_thr[fold])
            final = judge_final(stage, r.pred, kb, cat, jn)
            rows.append({"id": r.id, "fold": fold, "category": cat, "label": idx.loc[r.id, "label"],
                         "stage": stage, "final": final, "pred": r.pred, "vlm_normal": vn,
                         "judge": "vlm" if jd is None else f"b1_{judge_rule}", "judge_normal": vn if jn is None else jn,
                         "percentile": float(pct[cat](float(scores.loc[r.id]))),
                         "gt_report": idx.loc[r.id, "gt_report"]})
    return pd.DataFrame(rows)


def vlm_only(paths: list, kb: dict, index: pd.DataFrame) -> pd.DataFrame:
    """B2: 탐지기 없이 VLM이 모든 이미지를 판정. "정상"이면 무출력, 아니면 Guard 후 출력."""
    idx = index.set_index("image_id")
    rows = []
    for r in load_preds(paths).itertuples():
        cat = idx.loc[r.id, "category"]
        final = "" if is_normal(r.pred) else guard(r.pred, kb, cat, None)[0]
        rows.append({"id": r.id, "category": cat, "label": idx.loc[r.id, "label"], "stage": "vlm_only",
                     "final": final, "pred": r.pred, "vlm_normal": is_normal(r.pred),
                     "gt_report": idx.loc[r.id, "gt_report"]})
    return pd.DataFrame(rows)


def _outcomes(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(이상 이미지 출력 여부, 이상 이미지 정답 그룹 출력 여부, 정상 이미지 출력 여부). id 순 정렬."""
    df = df.sort_values("id")
    anom = (df["label"] != "good").to_numpy()
    out = (df["final"] != "").to_numpy()
    corr = np.array([a and group_correct(f, g) for a, f, g in zip(anom, df["final"], df["gt_report"])])
    return out[anom], corr[anom], out[~anom]


def _f1_from_rates(tpr, rc, fpr, pi):
    """π 보정 탐지 F1과 정답 그룹 F1 (배열 연산 가능)."""
    denom = pi * tpr + (1 - pi) * fpr
    with np.errstate(divide="ignore", invalid="ignore"):
        p_det = np.where(denom > 0, pi * tpr / denom, 0.0)
        p_c = np.where(denom > 0, pi * rc / denom, 0.0)
        f1_det = np.where(p_det + tpr > 0, 2 * p_det * tpr / (p_det + tpr), 0.0)
        f1_c = np.where(p_c + rc > 0, 2 * p_c * rc / (p_c + rc), 0.0)
    return f1_det, f1_c, p_det, p_c


def end_to_end(df: pd.DataFrame, pis: tuple) -> dict:
    out_a, corr_a, out_n = _outcomes(df)
    tpr, rc, fpr = out_a.mean(), corr_a.mean(), out_n.mean()
    res = {"recall": float(tpr), "fpr": float(fpr), "correct_group_recall": float(rc)}
    for pi in pis:
        f1_det, f1_c, p_det, p_c = _f1_from_rates(tpr, rc, fpr, pi)
        res[f"pi={pi:g}"] = {"precision": float(p_det), "f1_detect": float(f1_det),
                             "precision_group": float(p_c), "f1_group": float(f1_c)}
    return res


def compare_e2e(a: pd.DataFrame, b: pd.DataFrame, pis: tuple, n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """같은 이미지 집합에서 a - b의 종단 F1 차이와 층화 paired bootstrap 95% CI."""
    if set(a["id"]) != set(b["id"]):
        raise ValueError("두 시스템의 이미지 집합이 다름")
    (oa, ca, na), (ob, cb, nb) = _outcomes(a), _outcomes(b)
    rng = np.random.default_rng(seed)
    ia = rng.integers(0, len(oa), size=(n_boot, len(oa)))
    inn = rng.integers(0, len(na), size=(n_boot, len(na)))
    res = {}
    for pi in pis:
        point = {}
        for key, k in (("f1_detect", 0), ("f1_group", 1)):
            fa = _f1_from_rates(oa.mean(), ca.mean(), na.mean(), pi)[k]
            fb = _f1_from_rates(ob.mean(), cb.mean(), nb.mean(), pi)[k]
            da = _f1_from_rates(oa[ia].mean(1), ca[ia].mean(1), na[inn].mean(1), pi)[k]
            db = _f1_from_rates(ob[ia].mean(1), cb[ia].mean(1), nb[inn].mean(1), pi)[k]
            d = da - db
            point[key] = {"a": float(fa), "b": float(fb), "diff": float(fa - fb),
                          "ci95": [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))]}
        res[f"pi={pi:g}"] = point
    return res


def decompose(df: pd.DataFrame) -> dict:
    """표2: 탐지기 오류와 VLM 오류 분리 (최종 출력 기준)."""
    anom = df["label"] != "good"
    corr = pd.Series([a and group_correct(f, g) for a, f, g in zip(anom, df["final"], df["gt_report"])], index=df.index)
    raw_corr = pd.Series([a and group_correct("" if n else p, g) for a, n, p, g in
                          zip(anom, df["vlm_normal"], df["pred"], df["gt_report"])], index=df.index)
    st, vn = df["stage"], df["vlm_normal"]
    a = {
        "탐지기 놓침(통과 구간)": anom & (st == "pass"),
        "VLM 오거부(판정 구간)": anom & (st == "judge") & vn,
        "그룹 오답(판정 구간)": anom & (st == "judge") & ~vn & ~corr,
        "그룹 오답(설명 구간)": anom & (st == "describe") & ~corr,
        "정답": anom & corr,
    }
    n = {
        "통과(무출력)": ~anom & (st == "pass"),
        "VLM 거부(판정 구간)": ~anom & (st == "judge") & vn,
        "오경보: 판정 구간 VLM 이상 판단": ~anom & (st == "judge") & ~vn,
        "오경보: 설명 구간(거부 불가)": ~anom & (st == "describe"),
    }
    return {"anomaly": {k: int(v.sum()) for k, v in a.items()},
            "anomaly_missed_but_vlm_raw_correct": int((anom & (st == "pass") & raw_corr).sum()),
            "oracle_group_acc": float(raw_corr[anom].mean()),
            "normal": {k: int(v.sum()) for k, v in n.items()},
            "normal_describe_vlm_said_normal": int((~anom & (st == "describe") & vn).sum())}


def band_rejection(df: pd.DataFrame) -> list:
    """탐지기 백분위 구간별 VLM "정상" 응답률 (정상·이상 각각)."""
    out = []
    for lo, hi, name in PCT_BINS:
        m = (df["percentile"] >= lo) & (df["percentile"] < hi)
        g = df[m]
        nm, an = g[g["label"] == "good"], g[g["label"] != "good"]
        out.append({"bin": name, "normal_n": len(nm), "normal_said_normal": float(nm["vlm_normal"].mean()) if len(nm) else None,
                    "anomaly_n": len(an), "anomaly_said_normal": float(an["vlm_normal"].mean()) if len(an) else None,
                    "stages": g["stage"].value_counts().to_dict()})
    return out


def summarize(df: pd.DataFrame) -> dict:
    anom = df["label"] != "good"
    det = df["stage"] != "pass"
    cas = df["final"] != ""
    tpr_d, fpr_d = float(det[anom].mean()), float(det[~anom].mean())
    tpr_c, fpr_c = float(cas[anom].mean()), float(cas[~anom].mean())
    test_pi = float(anom.mean())
    grp = []
    for t, g in zip(df.loc[anom, "final"], df.loc[anom, "gt_report"]):
        p, q = parse_report(t) if t else None, parse_report(g)
        grp.append(p is not None and p.group == q.group)
    judge = df["stage"] == "judge"
    jn = df["judge_normal"] if "judge_normal" in df else df["vlm_normal"]  # 판정 구간에서 실제로 거부 판단을 내린 쪽
    external = "judge" in df and bool((df["judge"] != "vlm").any())
    # VLM 호출: VLM 심판이면 tau_low 이상 전체(판정 구간도 VLM이 봐야 함), 외부 심판(B1)이면 설명 구간 + 심판이 이상이라 한 판정 구간
    call = ((df["stage"] == "describe") | (judge & ~jn)) if external else det
    tpr_call, fpr_call = float(call[anom].mean()), float(call[~anom].mean())
    return {
        "n_anomaly": int(anom.sum()), "n_normal": int((~anom).sum()),
        "detector_only": {"recall": tpr_d, "fpr": fpr_d},
        "cascade": {"recall": tpr_c, "fpr": fpr_c},
        "recall_loss_pp": 100 * (tpr_d - tpr_c),
        "false_alarm_reduction": (fpr_d - fpr_c) / fpr_d if fpr_d > 0 else None,
        "rq4_hypothesis_met": (100 * (tpr_d - tpr_c) <= 1.0) and fpr_d > 0 and (fpr_d - fpr_c) / fpr_d >= 0.5,
        "judge_band": {"normal_n": int((judge & ~anom).sum()), "normal_rejected": float(jn[judge & ~anom].mean())
                       if (judge & ~anom).any() else None,
                       "anomaly_n": int((judge & anom).sum()), "anomaly_wrongly_rejected": float(jn[judge & anom].mean())
                       if (judge & anom).any() else None},
        "vlm_call_rate": {f"pi={p:g}": p * tpr_call + (1 - p) * fpr_call for p in PIS + (test_pi,)},
        "end_to_end_group_acc": sum(grp) / max(len(grp), 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="m2_4b_f{f}_s0", help="run 이름 형식 ({f}=학습 fold)")
    ap.add_argument("--b2", default="m3_b2_4b_f{f}", help="B2 run 이름 형식 (없으면 건너뜀)")
    ap.add_argument("--out", default=str(ROOT / "reports" / "tables" / "m3_cascade"))
    a = ap.parse_args()
    kb = load_kb()
    index = pd.read_parquet(ROOT / "data" / "index.parquet")
    splits = pd.read_csv(ROOT / "data" / "splits.csv")
    pred = lambda name, f: ROOT / "runs" / name.format(f=f) / "final" / "heldout_predictions.jsonl"  # noqa: E731
    models = {f: pred(a.run, f) for f in (0, 1)}
    res = {"run": a.run, "by_recall_target": {}}
    frames = {}
    for rc in RECALLS:
        frames[rc] = cascade(models, rc, kb, index, splits)
        res["by_recall_target"][str(rc)] = summarize(frames[rc])
    main_res = res["by_recall_target"]["0.95"]
    main_df = frames[0.95]
    test_pi = float((main_df["label"] != "good").mean())
    pis = (test_pi,) + PIS
    res["end_to_end"] = {"ours": end_to_end(main_df, pis)}
    res["decomposition"] = decompose(main_df)
    res["band_rejection"] = band_rejection(main_df)
    b2 = {f: pred(a.b2, f) for f in (0, 1)}
    if all(p.exists() for p in b2.values()):
        b2_df = vlm_only(list(b2.values()), kb, index)
        e2e = end_to_end(b2_df, pis)
        res["b2_vlm_only"] = {"recall": e2e["recall"], "fpr": e2e["fpr"]}
        res["end_to_end"]["b2"] = e2e
        res["end_to_end"]["ours_minus_b2"] = compare_e2e(main_df, b2_df, pis)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # 그림 스크립트(src.report.figures pi_curve)가 읽는 형식으로도 저장
    out.with_name(out.name + "_pi.summary.json").write_text(json.dumps(main_res, ensure_ascii=False, indent=1),
                                                             encoding="utf-8")
    lines = [f"# RQ4 캐스케이드 ({a.run}, 두 held-out fold 합산, 이상 {main_res['n_anomaly']} / 정상 {main_res['n_normal']})", "",
             "| tau_low 학습 측 재현율 목표 | 탐지기 단독 재현율 / FPR | 캐스케이드 재현율 / FPR | 재현율 손실(%p) | 오경보 감소 | 가설 충족 |",
             "|---|---|---|---|---|---|"]
    for rc, s in res["by_recall_target"].items():
        d, c = s["detector_only"], s["cascade"]
        red = "-" if s["false_alarm_reduction"] is None else f"{100 * s['false_alarm_reduction']:.0f}%"
        lines.append(f"| {rc} | {100 * d['recall']:.1f} / {100 * d['fpr']:.1f} | {100 * c['recall']:.1f} / {100 * c['fpr']:.1f} | "
                     f"{s['recall_loss_pp']:.1f} | {red} | {'예' if s['rq4_hypothesis_met'] else '아니오'} |")
    jb = main_res["judge_band"]
    lines += ["", f"- 판정 구간(tau_low~tau_high): 정상 {jb['normal_n']}장 중 VLM이 정상으로 거부한 비율 "
              f"{100 * (jb['normal_rejected'] or 0):.1f}%, 이상 {jb['anomaly_n']}장 중 잘못 거부한 비율 {100 * (jb['anomaly_wrongly_rejected'] or 0):.1f}%",
              "- VLM 호출률(π 보정, 목표 0.95): " + ", ".join(f"{k} {100 * v:.1f}%" for k, v in main_res["vlm_call_rate"].items()),
              f"- 종단 그룹 정확도(이상 이미지): {100 * main_res['end_to_end_group_acc']:.1f}%"]
    if "b2_vlm_only" in res:
        lines.append(f"- B2 VLM 단독: 재현율 {100 * res['b2_vlm_only']['recall']:.1f}, FPR {100 * res['b2_vlm_only']['fpr']:.1f}")

    e = res["end_to_end"]
    pct = lambda v: f"{100 * v:.1f}"  # noqa: E731
    lines += ["", "## 종단 F1 (최종 출력 기준, tau_low 목표 0.95)", "",
              "정답 그룹 F1: 정밀도 = 정답 그룹 출력 / 전체 출력, 재현율 = 정답 그룹 출력 / 이상 이미지. "
              "π 보정 정밀도 = π·R / (π·TPR + (1-π)·FPR).", "",
              "| 시스템 | 재현율 | FPR | 정답 그룹 재현율 | " + " | ".join(f"탐지 F1 / 그룹 F1 (π={p:.3g})" for p in pis) + " |",
              "|---|---|---|---|" + "---|" * len(pis)]
    for name, key in (("Ours 캐스케이드", "ours"), ("B2 VLM 단독", "b2")):
        if key in e:
            s = e[key]
            lines.append(f"| {name} | {pct(s['recall'])} | {pct(s['fpr'])} | {pct(s['correct_group_recall'])} | "
                         + " | ".join(f"{pct(s[f'pi={p:g}']['f1_detect'])} / {pct(s[f'pi={p:g}']['f1_group'])}" for p in pis) + " |")
    if "ours_minus_b2" in e:
        lines += ["", "| π | 탐지 F1 차이 Ours - B2 (%p) [95% CI] | 그룹 F1 차이 (%p) [95% CI] |", "|---|---|---|"]
        for p in pis:
            d = e["ours_minus_b2"][f"pi={p:g}"]
            lines.append(f"| {p:.3g} | " + " | ".join(
                f"{100 * d[k]['diff']:+.1f} [{100 * d[k]['ci95'][0]:+.1f}, {100 * d[k]['ci95'][1]:+.1f}]"
                for k in ("f1_detect", "f1_group")) + " |")

    dc = res["decomposition"]
    lines += ["", "## 오류 분해: 탐지기 오류 vs VLM 오류 (tau_low 목표 0.95)", "",
              f"이상 {main_res['n_anomaly']}장: " + ", ".join(f"{k} {v}" for k, v in dc["anomaly"].items()),
              f"- 탐지기가 놓친 이상 중 VLM 원출력은 정답이었던 이미지 {dc['anomaly_missed_but_vlm_raw_correct']}장 "
              f"(oracle 게이팅 그룹 정확도 {pct(dc['oracle_group_acc'])} -> 종단 {pct(main_res['end_to_end_group_acc'])})",
              f"정상 {main_res['n_normal']}장: " + ", ".join(f"{k} {v}" for k, v in dc["normal"].items()),
              f"- 설명 구간 오경보 중 VLM은 정상이라 답했던 이미지 {dc['normal_describe_vlm_said_normal']}장 (거부 불가 규칙 때문에 출력됨)"]

    lines += ["", "## 탐지기 백분위 구간별 VLM \"정상\" 응답률 (held-out 전 이미지)", "",
              "| 백분위 구간 | 정상 n | 정상을 정상이라 답함 | 이상 n | 이상을 정상이라 답함 | 단계 분포 |", "|---|---|---|---|---|---|"]
    for b in res["band_rejection"]:
        f = lambda v: "-" if v is None else pct(v)  # noqa: E731
        lines.append(f"| {b['bin']} | {b['normal_n']} | {f(b['normal_said_normal'])} | {b['anomaly_n']} | "
                     f"{f(b['anomaly_said_normal'])} | {', '.join(f'{k} {v}' for k, v in sorted(b['stages'].items()))} |")
    out.with_suffix(".json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
