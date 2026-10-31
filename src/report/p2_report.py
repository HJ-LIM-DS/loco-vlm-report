"""추가 실험 1(모델 크기와 라벨 수) 결과 집계. 가설과 판정 기준은 결과를 보기 전에 적어 둔 것을 그대로 적용한다.

- 모델 크기: 8B QLoRA vs 4B seed 0 (두 held-out fold 합산 993장), 8B vs B1. 4B seed 3개 범위와 비교.
- 라벨 수 곡선: 2B LoRA와 B1을 같은 학습 측 부분집합(25/50/100%)으로 학습해 fold 1 이미지(496장)에서 비교.
  인접 비율 쌍대 차이(McNemar), 라벨 2배당 증가폭, 결함 종류별 분해. 그림 `reports/figures/label_curve.png`.
- 있는 결과만 집계하므로 대기열 중간에도 실행할 수 있다.

    python -m src.report.p2_report
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.explanation import evaluate
from src.eval.failure import COUNT_GROUPS
from src.eval.stats import mcnemar, paired_bootstrap
from src.report.m2_report import correct_vector, heldout
from src.report.m3_report import verdict

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "tables"
BASE = ROOT / "outputs" / "baselines"
FRACS = (0.25, 0.5, 1.0)
VLM_CURVE = {0.25: [heldout("p2_2b_f0_frac25")], 0.5: [heldout("p2_2b_f0_frac50")], 1.0: [heldout("m2_2b_f0_s0")]}
B1_CURVE = {0.25: [BASE / "p2_b1_f0_frac25_heldout.jsonl"], 0.5: [BASE / "p2_b1_f0_frac50_heldout.jsonl"],
            1.0: [BASE / "b1_f0_heldout.jsonl"]}
FOUR_B = {s: [heldout(f"m2_4b_f{f}_s{s}") for f in (0, 1)] for s in (0, 1, 2)}
EIGHT_B = [heldout(f"p2_8b_f{f}") for f in (0, 1)]
B1_ALL = [BASE / f"b1_f{f}_heldout.jsonl" for f in (0, 1)]
KINDS = ("개수 계열", "기타 논리", "구조")


def pct(v, sign=False):
    return "-" if v is None else (f"{100 * v:+.1f}" if sign else f"{100 * v:.1f}")


def kinds(ids: pd.Index) -> pd.Series:
    idx = pd.read_parquet(ROOT / "data" / "index.parquet").set_index("image_id").loc[ids]
    cnt = [(c, g[0]) in COUNT_GROUPS for c, g in zip(idx["category"], idx["groups"])]
    return pd.Series(np.where(cnt, "개수 계열", np.where(idx["label"] == "logical", "기타 논리", "구조")), index=ids)


def compare(a: pd.Series, b: pd.Series) -> dict:
    """a - b 정확도 차이 (같은 이미지)."""
    b = b.reindex(a.index)
    return {**paired_bootstrap(a.values, b.values), **mcnemar(a.values, b.values), "n": len(a)}


def range_note(eight: float, rng: list, in_range: bool, p: float) -> str:
    """8B 그룹 정확도가 4B seed 범위에 드는지에 대한 판정 문구 (결과 전에 정한 규칙: 범위 안이면 용량이 병목이 아님)."""
    if in_range:
        return "안에 있음. 용량이 병목이 아니라는 해석"
    off = eight - (max(rng) if eight > max(rng) else min(rng))
    if p >= 0.05:
        return f"밖({pct(off, True)}%p)이지만 쌍대 차이의 95% CI가 0을 포함해 용량 2배의 이득은 확인되지 않음"
    return f"밖({pct(off, True)}%p)이고 쌍대 차이도 유의함"


def size_section(res: dict) -> list:
    lines = ["## 1. 모델 크기: 8B QLoRA vs 4B (두 held-out fold 합산)", ""]
    if not all(p.exists() for p in EIGHT_B):
        return lines + ["- 대기열 진행 중: p2_8b_f0, p2_8b_f1"]
    four = {s: correct_vector(p) for s, p in FOUR_B.items() if all(x.exists() for x in p)}
    eight, b1 = correct_vector(EIGHT_B), correct_vector(B1_ALL)
    k = kinds(eight.index)
    cols = [("group_acc", "그룹"), ("macro_f1", "macro-F1"), ("type_acc", "유형"), ("detail_exact", "세부"),
            ("location_hit", "위치 적중"), ("normal_correct", "정상 거부"), ("format_ok", "형식")]
    systems = {"8B QLoRA": EIGHT_B, **{f"4B QLoRA s{s}": FOUR_B[s] for s in four}, "B1 비-VLM": B1_ALL}
    lines += ["| 시스템 | " + " | ".join(c for _, c in cols) + " | " + " | ".join(KINDS) + " |",
              "|---" * (len(cols) + len(KINDS) + 1) + "|"]
    res["size"] = {"metrics": {}}
    for name, paths in systems.items():
        m = evaluate(paths)["raw"]
        v = correct_vector(paths).reindex(eight.index)
        res["size"]["metrics"][name] = {**{c: m[c] for c, _ in cols}, **{kd: float(v[k == kd].mean()) for kd in KINDS}}
        lines.append(f"| {name} | " + " | ".join(pct(m[c]) for c, _ in cols) + " | "
                     + " | ".join(pct(v[k == kd].mean()) for kd in KINDS) + " |")
    d4, db1 = compare(eight, four[0]), compare(eight, b1)
    rng = [float(v.mean()) for v in four.values()]
    in_range = min(rng) <= float(eight.mean()) <= max(rng)
    res["size"].update({"vs_4b_s0": d4, "vs_b1": db1, "four_b_range": [min(rng), max(rng)], "within_4b_seed_range": in_range,
                        "by_kind_vs_4b_s0": {kd: compare(eight[k == kd], four[0][k == kd]) for kd in KINDS}})
    lines += ["", "| 비교 | 차이(%p) [95% CI] | McNemar p | 앞선 쪽만 정답 / 뒤 쪽만 정답 |", "|---|---|---|---|"]
    for name, d in (("8B - 4B s0", d4), ("8B - B1", db1)):
        lines.append(f"| {name} | {pct(d['diff'], True)} [{pct(d['ci95'][0], True)}, {pct(d['ci95'][1], True)}] | "
                     f"{d['p']:.2g} | {d['n10']} / {d['n01']} |")
    lines += ["", "- 결함 종류별 8B - 4B s0: " + ", ".join(
        f"{kd} {pct(v['diff'], True)}(p={v['p']:.2g}, n={v['n']})" for kd, v in res["size"]["by_kind_vs_4b_s0"].items()),
              f"- 가설(8B > 4B s0) 판정: {verdict(d4, d4['p'])}. 8B {pct(eight.mean())}는 4B seed 범위 "
              f"{pct(min(rng))}~{pct(max(rng))} {range_note(float(eight.mean()), rng, in_range, d4['p'])}."]
    res["size"]["offset_from_4b_range"] = 0.0 if in_range else float(eight.mean()) - (max(rng) if eight.mean() > max(rng) else min(rng))
    cost, res["size"]["cost"] = [], {}
    for run in ("p2_8b_f0", "p2_8b_f1", "m2_4b_f0_s0", "m2_4b_f1_s0"):
        p = ROOT / "runs" / run / "final" / "summary.json"
        if p.exists():
            s = json.loads(p.read_text(encoding="utf-8"))
            rt = s.get("train_runtime") or 0
            # 재개된 run의 train_runtime은 마지막 구간만 재므로(m4_tables.cost와 같은 600초 기준) 집계하지 않는다
            hours = rt / 3600 if rt >= 600 else None
            spi = (s.get("heldout_generation") or {}).get("seconds_per_image")
            res["size"]["cost"][run] = {"hours": hours, "vram_alloc": s["peak_vram_alloc_gb"],
                                        "vram_reserved": s["peak_vram_reserved_gb"], "seconds_per_image": spi}
            cost.append(f"| {run} | {'재개(미집계)' if hours is None else f'{hours:.2f}'} | {s['peak_vram_alloc_gb']} / "
                        f"{s['peak_vram_reserved_gb']} | {'-' if spi is None else f'{spi:.1f}'} |")
    if cost:
        lines += ["", "| run | 학습 시간(h) | 최대 VRAM 할당/예약(GB) | 생성 지연(초/장) |", "|---|---|---|---|"] + cost
    return lines


def curve_section(res: dict) -> list:
    lines = ["", "## 2. 라벨 수 곡선 (fold 0 학습 측 라벨 비율로 학습, fold 1 이미지로 평가)", ""]
    have = {name: {f: p for f, p in curve.items() if all(x.exists() for x in p)}
            for name, curve in (("VLM 2B", VLM_CURVE), ("B1", B1_CURVE))}
    if any(len(v) < len(FRACS) for v in have.values()):
        missing = [f"{n} {int(100 * f)}%" for n, v in have.items() for f in FRACS if f not in v]
        lines.append("- 대기열 진행 중: " + ", ".join(missing))
        if not all(len(v) >= 2 for v in have.values()):
            return lines
    res["curve"] = {}
    meta = {f: json.loads((ROOT / "data" / "sft" / (f"p2_frac{int(100 * f)}_f0" if f < 1 else "m2_salad_f0") / "meta.json")
                          .read_text(encoding="utf-8"))["counts"] for f in FRACS
            if (ROOT / "data" / "sft" / (f"p2_frac{int(100 * f)}_f0" if f < 1 else "m2_salad_f0") / "meta.json").exists()}
    lines += ["| 라벨 비율 | VLM 학습 이상/정상 | VLM 2B 그룹 | B1 그룹 | " + " | ".join(f"VLM / B1 {kd}" for kd in KINDS)
              + " | VLM 정상 거부 |", "|---|---|---|---|" + "---|" * len(KINDS) + "---|"]
    vec = {name: {f: correct_vector(p) for f, p in v.items()} for name, v in have.items()}
    ids = vec["VLM 2B"][max(vec["VLM 2B"])].index
    k = kinds(ids)
    for f in FRACS:
        row = {}
        for name in vec:
            if f in vec[name]:
                v = vec[name][f].reindex(ids)
                row[name] = {"all": float(v.mean()), **{kd: float(v[k == kd].mean()) for kd in KINDS}}
        if f in have["VLM 2B"]:
            row["VLM 2B"]["normal_correct"] = evaluate(have["VLM 2B"][f])["raw"]["normal_correct"]
        c = meta.get(f, {}).get("train", {})
        row["train_counts"] = c
        res["curve"][str(f)] = row
        g = lambda n, kk="all": pct(row[n][kk]) if n in row else "-"  # noqa: E731
        lines.append(f"| {int(100 * f)}% | {c.get('anomaly', '-')} / {c.get('good', '-')} | {g('VLM 2B')} | {g('B1')} | "
                     + " | ".join(f"{g('VLM 2B', kd)} / {g('B1', kd)}" for kd in KINDS)
                     + f" | {pct(row.get('VLM 2B', {}).get('normal_correct'))} |")
    lines += ["", "| 모델 | 25에서 50% (%p, p) | 50에서 100% (%p, p) | 라벨 2배당 평균 증가(%p) |", "|---|---|---|---|"]
    res["curve_steps"] = {}
    for name, v in vec.items():
        steps = {}
        for lo, hi in ((0.25, 0.5), (0.5, 1.0)):
            if lo in v and hi in v:
                steps[f"{lo}->{hi}"] = compare(v[hi].reindex(ids), v[lo].reindex(ids))
        slope = float(np.mean([d["diff"] for d in steps.values()])) if steps else None
        res["curve_steps"][name] = {"steps": steps, "gain_per_doubling": slope}
        cell = lambda key: (f"{pct(steps[key]['diff'], True)} ({steps[key]['p']:.2g})" if key in steps else "-")  # noqa: E731
        lines.append(f"| {name} | {cell('0.25->0.5')} | {cell('0.5->1.0')} | {pct(slope, True)} |")
    s = res["curve_steps"]
    # 비율별 B1 - VLM 격차: 기울기 비교만으로는 격차가 닫히는지 보이지 않으므로 함께 보고
    gap = {str(f): res["curve"][str(f)]["B1"]["all"] - res["curve"][str(f)]["VLM 2B"]["all"]
           for f in FRACS if {"B1", "VLM 2B"} <= res["curve"][str(f)].keys()}
    res["curve_gap_b1_minus_vlm"] = gap
    if "0.5->1.0" in s["VLM 2B"]["steps"]:
        top = s["VLM 2B"]["steps"]["0.5->1.0"]
        sv, sb = s["VLM 2B"]["gain_per_doubling"], s["B1"]["gain_per_doubling"]
        if sv is not None and sb is not None and sv > sb:
            h2 = (f"VLM이 {pct(sv - sb, True)}%p 더 가파름(사전 기준 충족). 다만 B1 - VLM 격차는 "
                  + " → ".join(pct(v) for v in gap.values()) + "%p로 거의 일정해 라벨을 늘려도 격차가 닫힐 징후는 약함")
        else:
            h2 = "VLM이 더 가파르지 않음"
        lines += ["", f"- 가설 1(VLM은 50에서 100%로 늘려도 증가) 판정: {verdict(top, top['p'])}.",
                  f"- 가설 2(VLM의 2배당 증가폭 > B1): VLM {pct(sv, True)} vs B1 {pct(sb, True)} → {h2}."]
    return lines


def fig_label_curve(res: dict):
    """라벨 수 곡선: VLM 2B와 B1의 그룹 정확도 (x는 라벨 비율, 로그 2 눈금)."""
    import matplotlib.pyplot as plt

    from src.report import style
    from src.report.style import SERIES, TEXT_2, save, setup

    setup()
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    for i, name in enumerate(("VLM 2B", "B1")):
        pts = [(float(f), r[name]["all"]) for f, r in res["curve"].items() if name in r]
        if not pts:
            continue
        x, y = zip(*sorted(pts))
        ax.plot([100 * v for v in x], [100 * v for v in y], color=SERIES[i], marker="o", markersize=6,
                markeredgecolor=style.SURFACE, markeredgewidth=2, label=name)
        ax.annotate(f"{name} {100 * y[-1]:.1f}", (100 * x[-1], 100 * y[-1]), textcoords="offset points", xytext=(6, -3),
                    fontsize=8, color=TEXT_2)
    ax.set_xscale("log", base=2)
    ax.set_xticks([25, 50, 100], ["25%", "50%", "100%"])
    ax.set_xlabel("학습 측 라벨 비율 (fold 0)")
    ax.set_ylabel("그룹 정확도 (%, fold 1 이상 이미지)")
    ax.set_title("라벨 수 곡선", loc="left")
    ax.legend(loc="lower right")
    return save(fig, "label_curve.png")


def main():
    res = {}
    lines = ["# 추가 실험 1 결과 요약: 모델 크기와 라벨 수 (가설·판정 기준은 결과 전 기록)", ""]
    lines += size_section(res)
    lines += curve_section(res)
    if res.get("curve"):
        lines += ["", f"그림: `reports/figures/{fig_label_curve(res).name}`"]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "p2_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "p2_summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"저장: {OUT / 'p2_summary.md'}")


if __name__ == "__main__":
    main()
