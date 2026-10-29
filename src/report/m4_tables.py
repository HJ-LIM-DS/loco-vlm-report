"""PDF·README 표1: 본 결과 (B0~B3, Ours 2B/4B, 있으면 P2 8B). 모든 시스템을 같은 채점기로 평가한다.

- 설명 지표(oracle 게이팅, 이상 이미지 993장 / 정상 575장): 그룹, macro-F1, 유형, 세부, 위치 적중, 정상 거부, 형식 (원출력)
- 종단 그룹 F1: 각 시스템의 예측을 같은 SALAD 캐스케이드(tau_low 학습 측 재현율 0.95)에 넣었을 때
  최종 출력 기준 (test 비율과 π=1%). B2는 탐지기 없이 VLM 단독.
- 비용: 학습 시간(fold당 중앙값), 최대 VRAM(예약), 이상 이미지 1장 생성 지연. 탐지기 지연은 별도 줄.
- Ours 4B는 seed 3개 평균(범위)을 적는다.

    python -m src.report.m4_tables
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.cascade import cascade, end_to_end, vlm_only
from src.eval.explanation import evaluate
from src.kb import load_kb
from src.report.m2_report import heldout

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "tables"
BASE = ROOT / "outputs" / "baselines"
PIS = (0.01,)
COLS = [("group_acc", "그룹"), ("macro_f1", "macro-F1"), ("type_acc", "유형"), ("detail_exact", "세부"),
        ("location_hit", "위치 적중"), ("normal_correct", "정상 거부"), ("format_ok", "형식")]


def runs(pattern: str, seeds=(None,)) -> list:
    """seed별 [fold 0 예측, fold 1 예측] 목록."""
    out = []
    for s in seeds:
        name = pattern if s is None else pattern.replace("{s}", str(s))
        out.append([heldout(name.format(f=f)) for f in (0, 1)])
    return out


# (표시 이름, 방식, seed별 예측 경로, 캐스케이드 사용 여부, 비용을 읽을 run 이름 형식)
SYSTEMS = [
    ("B0 최빈값", "카테고리별 최빈 그룹·위치", [[BASE / f"b0_f{f}_heldout.jsonl" for f in (0, 1)]], True, None),
    ("B1 비-VLM", "DINOv2 전역+crop + SALAD 3항 -> 로지스틱", [[BASE / f"b1_f{f}_heldout.jsonl" for f in (0, 1)]], True, None),
    ("B2 VLM 단독", "4B QLoRA, 탐지기·crop 없음", runs("m3_b2_4b_f{f}"), False, "m3_b2_4b_f{f}"),
    ("B3 zero-shot (엄격)", "4B 무학습", [[BASE / f"b3_zero_f{f}" / "strict.jsonl" for f in (0, 1)]], True, None),
    ("B3 zero-shot (관대)", "4B 무학습, 그룹명만 찾아 교정", [[BASE / f"b3_zero_f{f}" / "lenient.jsonl" for f in (0, 1)]], True, None),
    ("B3 few-shot (엄격)", "4B 무학습, 예시 포함", [[BASE / f"b3_few_f{f}" / "strict.jsonl" for f in (0, 1)]], True, None),
    ("Ours 2B", "2B bf16 LoRA", runs("m2_2b_f{f}_s0"), True, "m2_2b_f{f}_s0"),
    ("Ours 4B", "4B QLoRA (NF4)", runs("m2_4b_f{f}_s{s}", (0, 1, 2)), True, "m2_4b_f{f}_s{s}"),
    ("P2: 8B", "8B QLoRA (NF4)", runs("p2_8b_f{f}"), True, "p2_8b_f{f}"),
]


def cost(pattern: str | None, seeds) -> dict:
    if pattern is None:
        return {}
    rows = []
    for s in seeds:
        for f in (0, 1):
            p = ROOT / "runs" / pattern.replace("{s}", str(s)).format(f=f) / "final" / "summary.json"
            if p.exists():
                rows.append(json.loads(p.read_text(encoding="utf-8")))
    if not rows:
        return {}
    # 재개로 학습 시간이 짧게 기록된 run(생성 단계만 다시 한 경우)은 학습 시간 중앙값에서 뺀다
    tr = [r["train_runtime"] / 3600 for r in rows if (r.get("train_runtime") or 0) > 600]
    return {"train_h": float(np.median(tr)) if tr else None,
            "vram_gb": max(r["peak_vram_reserved_gb"] for r in rows),
            "sec_per_image": float(np.median([(r.get("heldout_generation") or {}).get("seconds_per_image", np.nan) for r in rows]))}


def e2e(paths: list, use_cascade: bool, kb, index, splits) -> dict:
    df = cascade({0: paths[0], 1: paths[1]}, 0.95, kb, index, splits) if use_cascade else vlm_only(paths, kb, index)
    test_pi = float((df["label"] != "good").mean())
    e = end_to_end(df, (test_pi,) + PIS)
    return {"test": e[f"pi={test_pi:g}"]["f1_group"], "pi=0.01": e["pi=0.01"]["f1_group"],
            "recall": e["recall"], "fpr": e["fpr"]}


def pct(v):
    return "-" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{100 * v:.1f}"


def main():
    kb = load_kb()
    index = pd.read_parquet(ROOT / "data" / "index.parquet")
    splits = pd.read_csv(ROOT / "data" / "splits.csv")
    det = json.loads((ROOT / "outputs" / "detector" / "salad" / "breakfast_box" / "meta.json").read_text(encoding="utf-8"))
    res, lines = {}, ["# 표1. 본 결과 (두 held-out fold 합산, 이상 993 / 정상 575, 원출력)", "",
                      "| 시스템 | 방식 | " + " | ".join(c for _, c in COLS)
                      + " | 종단 그룹 F1 (test 비율 / π=1%) | 학습(h/fold) | 최대 VRAM(GB) | 생성(초/장) |",
                      "|---|---|" + "---|" * (len(COLS) + 4)]
    for name, how, seed_paths, use_cas, cost_pat in SYSTEMS:
        seed_paths = [p for p in seed_paths if all(x.exists() for x in p)]
        if not seed_paths:
            continue
        ms = [evaluate(p)["raw"] for p in seed_paths]
        es = [e2e(p, use_cas, kb, index, splits) for p in seed_paths]
        seeds = (0, 1, 2) if len(seed_paths) > 1 else (None,)
        c = cost(cost_pat, [s for s in seeds if s is not None] or [None]) if cost_pat else {}
        agg = {k: float(np.mean([m[k] for m in ms])) for k, _ in COLS}
        rng = {k: (min(m[k] for m in ms), max(m[k] for m in ms)) for k, _ in COLS}
        res[name] = {"how": how, "n_seeds": len(ms), "metrics": agg, "range": rng,
                     "end_to_end_group_f1": {k: float(np.mean([e[k] for e in es])) for k in es[0]}, "cost": c}
        cell = lambda k: pct(agg[k]) + (f" ({pct(rng[k][0])}~{pct(rng[k][1])})" if len(ms) > 1 and k == "group_acc" else "")  # noqa: E731
        ee = res[name]["end_to_end_group_f1"]
        lines.append(f"| {name}{f' (seed {len(ms)}개 평균)' if len(ms) > 1 else ''} | {how} | "
                     + " | ".join(cell(k) for k, _ in COLS)
                     + f" | {pct(ee['test'])} / {pct(ee['pi=0.01'])}{'' if use_cas else ' (탐지기 없음)'} | "
                     + (f"{c['train_h']:.2f}" if c.get("train_h") else "-") + f" | {c.get('vram_gb', '-')} | "
                     + (f"{c['sec_per_image']:.1f}" if c.get("sec_per_image") else "-") + " |")
    res["detector"] = {"name": "SALAD", "latency_ms_median": det.get("latency_ms_median")}
    lines += ["", f"- 탐지기 SALAD(공식 가중치 재현, 평균 image AUROC 96.11): 이미지당 지연 중앙값 {det.get('latency_ms_median')}ms. "
                  "캐스케이드는 학습 측 재현율 0.95 기준 tau_low와 정상 FPR 1% 기준 tau_high를 fold별로 쓴다.",
              "- 종단 그룹 F1: 정밀도 = 정답 그룹 출력 / 전체 출력, 재현율 = 정답 그룹 출력 / 이상 이미지, π 보정 정밀도 = "
              "π·R / (π·TPR + (1-π)·FPR). 같은 캐스케이드에 각 시스템의 예측을 넣어 계산한다(B2만 탐지기 없음).",
              "- RTX 3060 12GB. 생성 지연은 held-out 생성 시 배치 처리 기준(4B·8B 배치 2, 2B 배치 4)."]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "table1_main.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "table1_main.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"저장: {OUT / 'table1_main.md'}")


if __name__ == "__main__":
    main()
