"""본 실험 결과 집계: 주력, 경량판, 입력 제거 실험(ablation), 기준선을 같은 채점기로 평가하고 쌍대(paired) 통계를 붙인다.

- 두 held-out fold 예측을 합쳐 채점한다 (ablation은 fold 0 모델이라 fold 1 이미지만).
- 주 비교: Ours(4B seed 0) vs B0/B1/B3, McNemar + paired bootstrap(그룹 정확도, 이상 이미지).
- ablation(RQ2): 2B 기본 vs 정상 참조 제거 / crop 제거, 논리·구조로 나눠 쌍대 차이 + Holm 보정.
- 4B seed 3개가 있으면 seed별 지표와 평균을 함께 적는다.
- 있는 결과만 집계하므로 대기열 중간에도 다시 실행할 수 있다.

    python -m src.report.m2_report
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.explanation import evaluate, load_preds
from src.eval.stats import holm, mcnemar, paired_bootstrap
from src.output_spec import parse_report

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "tables"
RUNS = ROOT / "runs"
BASE = ROOT / "outputs" / "baselines"


def heldout(run: str) -> Path:
    return RUNS / run / "final" / "heldout_predictions.jsonl"


SYSTEMS = {
    "Ours 4B s0": [heldout("m2_4b_f0_s0"), heldout("m2_4b_f1_s0")],
    "Ours 4B s1": [heldout("m2_4b_f0_s1"), heldout("m2_4b_f1_s1")],
    "Ours 4B s2": [heldout("m2_4b_f0_s2"), heldout("m2_4b_f1_s2")],
    "Ours 2B": [heldout("m2_2b_f0_s0"), heldout("m2_2b_f1_s0")],
    "B0 최빈값": [BASE / "b0_f0_heldout.jsonl", BASE / "b0_f1_heldout.jsonl"],
    "B1 비-VLM": [BASE / "b1_f0_heldout.jsonl", BASE / "b1_f1_heldout.jsonl"],
    "B3 zero-shot(엄격)": [BASE / "b3_zero_f0" / "strict.jsonl", BASE / "b3_zero_f1" / "strict.jsonl"],
    "B3 zero-shot(관대)": [BASE / "b3_zero_f0" / "lenient.jsonl", BASE / "b3_zero_f1" / "lenient.jsonl"],
    "B3 few-shot(엄격)": [BASE / "b3_few_f0" / "strict.jsonl", BASE / "b3_few_f1" / "strict.jsonl"],
    "B3 few-shot(관대)": [BASE / "b3_few_f0" / "lenient.jsonl", BASE / "b3_few_f1" / "lenient.jsonl"],
}
ABLATIONS = {"2B 기본": heldout("m2_2b_f0_s0"), "정상 참조 제거": heldout("m2_2b_f0_noref"),
             "crop 제거": heldout("m2_2b_f0_nocrop")}
COLS = [("group_acc", "그룹"), ("macro_f1", "macro-F1"), ("type_acc", "유형"), ("detail_exact", "세부"),
        ("location_hit", "위치 적중"), ("action_consistency", "조치 정합"), ("severity_miss", "심각도 누락"),
        ("format_ok", "형식"), ("normal_correct", "정상 거부")]


def correct_vector(paths) -> pd.Series:
    """이상 이미지별 그룹 정답 여부 (원출력 기준), index = image_id."""
    idx = pd.read_parquet(ROOT / "data" / "index.parquet").set_index("image_id")
    pr = load_preds(paths).set_index("id")
    pr = pr[idx.loc[pr.index, "label"] != "good"]
    gt = idx.loc[pr.index, "gt_report"].map(parse_report)
    p = pr["pred"].map(lambda t: parse_report(t or ""))
    return pd.Series([a is not None and a.group == b.group for a, b in zip(p, gt)], index=pr.index)


def pct(v):
    return "-" if v is None else f"{100 * v:.1f}"


def main():
    res, lines = {}, ["# 본 실험 결과 요약 (두 held-out fold 합산, 원출력 / Guard 후)", ""]
    avail = {k: v for k, v in SYSTEMS.items() if all(Path(p).exists() for p in v)}
    lines += ["| 시스템 | " + " | ".join(c for _, c in COLS) + " |", "|---" * (len(COLS) + 1) + "|"]
    for name, paths in avail.items():
        m = evaluate(paths)
        res[name] = m
        lines.append(f"| {name} | " + " | ".join(f"{pct(m['raw'][k])} / {pct(m['guarded'][k])}" for k, _ in COLS) + " |")

    seeds = [k for k in ("Ours 4B s0", "Ours 4B s1", "Ours 4B s2") if k in res]
    if len(seeds) > 1:
        g = [res[k]["raw"]["group_acc"] for k in seeds]
        lines += ["", f"- 4B seed {len(seeds)}개 그룹 정확도: " + ", ".join(pct(x) for x in g)
                  + f" (평균 {pct(float(np.mean(g)))}, 범위 {pct(min(g))}~{pct(max(g))})"]

    if "Ours 4B s0" in avail:
        ours = correct_vector(avail["Ours 4B s0"])
        lines += ["", "## Ours 4B(s0) 대 기준선: 이상 이미지 그룹 정확도 쌍대 비교", "",
                  "| 비교 | 차이(%p) | 95% CI | McNemar p | Ours만 정답 / 기준선만 정답 |", "|---|---|---|---|---|"]
        res["pairwise"] = {}
        for name in avail:
            if name.startswith("Ours 4B"):
                continue
            other = correct_vector(avail[name]).reindex(ours.index)
            bt, mc = paired_bootstrap(ours.values, other.values), mcnemar(ours.values, other.values)
            res["pairwise"][name] = {**bt, **mc}
            lines.append(f"| vs {name} | {100 * bt['diff']:+.1f} | [{100 * bt['ci95'][0]:+.1f}, {100 * bt['ci95'][1]:+.1f}] "
                         f"| {mc['p']:.2g} | {mc['n10']} / {mc['n01']} |")

    abl = {k: v for k, v in ABLATIONS.items() if v.exists()}
    if "2B 기본" in abl and len(abl) > 1:
        idx = pd.read_parquet(ROOT / "data" / "index.parquet").set_index("image_id")
        base = correct_vector([abl["2B 기본"]])
        lines += ["", "## 입력 ablation (RQ2, 2B, fold 0 모델 → fold 1 이미지)", "",
                  "| 제거 입력 | 유형 | 차이(%p, 기본 - 제거) | 95% CI | McNemar p | Holm 보정 p |", "|---|---|---|---|---|---|"]
        rows, ps = [], {}
        for name, path in abl.items():
            if name == "2B 기본":
                continue
            v = correct_vector([path]).reindex(base.index)
            for lab in ("logical", "structural"):
                m = (idx.loc[base.index, "label"] == lab).to_numpy()
                bt, mc = paired_bootstrap(base.values[m], v.values[m]), mcnemar(base.values[m], v.values[m])
                key = f"{name}|{lab}"
                ps[key] = mc["p"]
                rows.append((key, bt, mc))
        adj = holm(ps)
        res["ablation"] = {}
        for key, bt, mc in rows:
            name, lab = key.split("|")
            res["ablation"][key] = {**bt, **mc, "p_holm": adj[key]}
            lines.append(f"| {name} | {lab} | {100 * bt['diff']:+.1f} | [{100 * bt['ci95'][0]:+.1f}, "
                         f"{100 * bt['ci95'][1]:+.1f}] | {mc['p']:.2g} | {adj[key]:.2g} |")

    mm = {}
    for f in (0, 1):
        b, l = ROOT / "outputs" / "mmad" / f"base_4b_f{f}.json", ROOT / "outputs" / "mmad" / f"lora_4b_s0_f{f}.json"
        if b.exists() and l.exists():
            mm[f] = (json.loads(b.read_text(encoding="utf-8")), json.loads(l.read_text(encoding="utf-8")))
    if mm:
        lines += ["", "## MMAD 객관식 정확도 (학습 전 → 학습 후, held-out 이미지)", "", "| fold | 문항 수 | 기본 | LoRA |",
                  "|---|---|---|---|"]
        for f, (b, l) in mm.items():
            lines.append(f"| {f} | {b['n_questions']} | {pct(b['accuracy'])} | {pct(l['accuracy'])} |")
        res["mmad"] = {f: {"base": b["accuracy"], "lora": l["accuracy"]} for f, (b, l) in mm.items()}

    train = []
    for run in ("m2_4b_f0_s0", "m2_4b_f1_s0", "m2_2b_f0_s0", "m2_2b_f1_s0"):
        p = RUNS / run / "final" / "summary.json"
        if p.exists():
            s = json.loads(p.read_text(encoding="utf-8"))
            train.append(f"| {run} | {s['train_samples']} | {s['global_step']} | {(s.get('train_runtime') or 0) / 3600:.2f} | "
                         f"{s['peak_vram_alloc_gb']} / {s['peak_vram_reserved_gb']} | "
                         f"{(s.get('heldout_generation') or {}).get('seconds_per_image', '-')} |")
    if train:
        lines += ["", "## 학습 비용", "", "| run | 학습 샘플 | step | 학습 시간(h) | 최대 VRAM 할당/예약(GB) | 생성 지연(초/장) |",
                  "|---|---|---|---|---|---|"] + train

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "m2_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "m2_summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
