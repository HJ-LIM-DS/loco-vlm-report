"""포트폴리오·README 그림. 결과 파일(summary.json 등)만 읽어 다시 그릴 수 있게 한다.

    python -m src.report.figures loc_modes
    python -m src.report.figures pi_curve --pipeline outputs/pipeline/m1_f0_heldout.summary.json
    python -m src.report.figures scores --detector efficientad --category breakfast_box --fold 0
    python -m src.report.figures loss --summary runs/m1_vlm2b_f0/final/summary.json
    python -m src.report.figures f1 --ref runs/f1_ref/final/summary.json --resumed runs/f1_kill/final/summary.json
    python -m src.report.figures ablation     (reports/tables/m3_summary.json)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.report import style
from src.report.style import ANOMALY_COLOR, NORMAL_COLOR, SERIES, TEXT_2, save, setup

ROOT = Path(__file__).resolve().parents[2]
LOC_ORDER = ["point", "global", "instances"]
LOC_KO = {"point": "칸 위치", "global": "전체", "instances": "위치 생략"}


def fig_loc_modes():
    """Phase 0: 카테고리별 이상 이미지의 위치 방식 분포 (가로 누적 막대, 2px 간격)."""
    import matplotlib.pyplot as plt

    df = pd.read_parquet(ROOT / "data" / "index.parquet")
    a = df[df["label"] != "good"]
    tab = a.groupby(["category", "loc_mode"]).size().unstack(fill_value=0).reindex(columns=LOC_ORDER, fill_value=0)
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    left = np.zeros(len(tab))
    for i, m in enumerate(LOC_ORDER):
        v = tab[m].to_numpy()
        ax.barh(tab.index, v, left=left, color=SERIES[i], height=0.6, edgecolor=style.SURFACE, linewidth=2,
                label=LOC_KO[m])
        for y, (x0, w) in enumerate(zip(left, v)):
            if w >= 15:  # 좁은 칸에는 숫자를 넣지 않는다 (선택적 직접 라벨)
                ax.text(x0 + w / 2, y, str(w), ha="center", va="center", color=style.TEXT, fontsize=8)
        left += v
    ax.invert_yaxis()
    ax.set_xlabel("이상 이미지 수")
    ax.set_title("위치 라벨 방식 (loc_mode)", loc="left")
    ax.grid(axis="y", visible=False)
    ax.legend(ncol=3, loc="lower right", bbox_to_anchor=(1, 1.0))
    return save(fig, "phase0_loc_modes.png")


def fig_pi_curve(pipeline_summary: Path, test_ratio: float = 0.633):
    """VLM 호출률 = pi*TPR + (1-pi)*FPR (탐지기 단계 기준), pi = 이상 비율."""
    import matplotlib.pyplot as plt

    s = json.loads(Path(pipeline_summary).read_text(encoding="utf-8"))
    tpr, fpr = s["detector_only"]["recall"], s["detector_only"]["fpr"]
    pi = np.logspace(-3, np.log10(0.7), 200)
    rate = pi * tpr + (1 - pi) * fpr
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.plot(pi * 100, rate * 100, color=NORMAL_COLOR)
    for p, name in [(0.01, "π=1%"), (0.05, "π=5%"), (test_ratio, f"test 비율 {test_ratio:.0%}")]:
        r = p * tpr + (1 - p) * fpr
        ax.plot([p * 100], [r * 100], "o", color=NORMAL_COLOR, markersize=6, markeredgecolor=style.SURFACE,
                markeredgewidth=2)
        ax.annotate(f"{name}: {r:.1%}", (p * 100, r * 100), textcoords="offset points", xytext=(6, -12),
                    fontsize=8, color=TEXT_2)
    ax.axhline(10, color=TEXT_2, linewidth=1, linestyle="--")
    ax.text(0.1, 10.5, "목표: 정상 호출 10%", fontsize=8, color=TEXT_2)
    ax.set_xscale("log")
    ax.set_xlabel("이상 비율 π (%)")
    ax.set_ylabel("VLM 호출률 (%)")
    ax.set_title(f"π 보정 VLM 호출률 (TPR {tpr:.2f}, FPR {fpr:.2f})", loc="left")
    return save(fig, "pi_call_rate.png")


def fig_scores(detector: str, category: str, fold: int):
    """학습 측 test 이미지의 탐지 점수 분포와 2단계 임계값."""
    import matplotlib.pyplot as plt

    from src.detector.cache import load_scores
    from src.detector.calibrate import fold_thresholds

    index = pd.read_parquet(ROOT / "data" / "index.parquet")
    splits = pd.read_csv(ROOT / "data" / "splits.csv")
    th = fold_thresholds(detector, category, index, splits, fold)
    s = load_scores(detector, category).merge(index[["image_id", "label", "split"]]).merge(
        splits[["image_id", "fold"]])
    s = s[(s["split"] == "test") & (s["fold"] == fold)]
    fig, ax = plt.subplots(figsize=(5.6, 2.8))
    bins = np.linspace(s["score"].min(), s["score"].max(), 40)
    for lab, color, name in [("normal", NORMAL_COLOR, "정상"), ("anomaly", ANOMALY_COLOR, "이상")]:
        v = s[(s["label"] == "good") == (lab == "normal")]["score"]
        ax.hist(v, bins=bins, color=color, alpha=0.75, label=f"{name} ({len(v)})", edgecolor=style.SURFACE,
                linewidth=1)
    for k, name in [("tau_low", "τ_low"), ("tau_high", "τ_high")]:
        ax.axvline(th[k], color=TEXT_2, linewidth=1, linestyle="--")
        ax.text(th[k], ax.get_ylim()[1] * 0.95, f" {name}", fontsize=8, color=TEXT_2, va="top")
    ax.set_xlabel(f"{detector} 이미지 점수")
    ax.set_ylabel("이미지 수")
    ax.set_title(f"{category} fold {fold} 학습 측 점수 분포", loc="left")
    ax.legend()
    return save(fig, f"scores_{detector}_{category}_f{fold}.png")


def fig_loss(summary: Path, name: str = "train_loss.png"):
    import matplotlib.pyplot as plt

    s = json.loads(Path(summary).read_text(encoding="utf-8"))
    d = sorted((int(k), v) for k, v in s["train_loss_by_step"].items())
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    ax.plot([x for x, _ in d], [y for _, y in d], color=NORMAL_COLOR)
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("학습 손실 (정답 토큰)")
    ax.set_title(f"학습 손실: {Path(s['data']).name}", loc="left")
    return save(fig, name)


def fig_f1(ref: Path, resumed: Path):
    """고장 주입 F1: 끊김 없는 실행과 강제 종료 후 재개한 실행의 손실 곡선 비교."""
    import matplotlib.pyplot as plt

    a = json.loads(Path(ref).read_text(encoding="utf-8"))["train_loss_by_step"]
    b = json.loads(Path(resumed).read_text(encoding="utf-8"))["train_loss_by_step"]
    steps = sorted(int(k) for k in a)
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    ax.plot(steps, [a[str(k)] if str(k) in a else a[k] for k in steps], color=NORMAL_COLOR, label="끊김 없음")
    ks = sorted(int(k) for k in b)
    ax.plot(ks, [b[str(k)] if str(k) in b else b[k] for k in ks], color=ANOMALY_COLOR, linestyle="--",
            label="강제 종료 후 재개")
    ax.set_xlabel("step")
    ax.set_ylabel("손실")
    ax.set_title("F1: 재개 전후 손실 곡선", loc="left")
    ax.legend()
    return save(fig, "ops_f1_loss.png")


def fig_ablation(summary: Path = ROOT / "reports" / "tables" / "m3_summary.json"):
    """그림3: ablation 쌍대 차이와 95% CI (논리/구조 분리). 2B 선별과 4B 확인(있으면)을 위아래 패널로."""
    import matplotlib.pyplot as plt

    s = json.loads(Path(summary).read_text(encoding="utf-8"))
    panels = [(k, t) for k, t in (("ablation_2b", "2B 선별 (fold 0 모델 -> fold 1 이미지)"),
                                  ("confirm_4b", "4B 확인 (두 held-out fold)")) if s.get(k)]
    heights = [len(s[k]) for k, _ in panels]
    fig, axes = plt.subplots(len(panels), 1, figsize=(6.4, 0.42 * sum(heights) + 1.2 * len(panels)), sharex=True,
                             gridspec_kw={"height_ratios": heights}, squeeze=False)
    for ax, (key, title) in zip(axes[:, 0], panels):
        names = list(s[key])
        for j, (lab, ko) in enumerate((("logical", "논리"), ("structural", "구조"))):
            for i, n in enumerate(names):
                r = s[key][n][lab]
                y = i + (j - 0.5) * 0.3
                ax.plot([100 * r["ci95"][0], 100 * r["ci95"][1]], [y, y], color=SERIES[j], linewidth=2,
                        solid_capstyle="round")
                ax.plot([100 * r["diff"]], [y], "o", color=SERIES[j], markersize=6, markeredgecolor=style.SURFACE,
                        markeredgewidth=2, label=ko if i == 0 else None)
                if s[key][n]["p_holm"][lab] < 0.05:  # 유의한 점만 값 라벨 (선택적 직접 라벨)
                    ax.annotate(f"{100 * r['diff']:+.1f}", (100 * r["ci95"][1], y), textcoords="offset points",
                                xytext=(4, -3), fontsize=8, color=TEXT_2)
        ax.axvline(0, color=TEXT_2, linewidth=1)
        ax.set_yticks(range(len(names)), names)
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        ax.set_title(title, loc="left")
    axes[0, 0].legend(ncol=2, loc="lower right", bbox_to_anchor=(1, 1.0))
    axes[-1, 0].set_xlabel("그룹 정확도 효과 (%p, 적용 - 미적용), 95% CI. 값 라벨 = Holm p < 0.05")
    return save(fig, "ablation_effects.png")


def main():
    setup()
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["loc_modes", "pi_curve", "scores", "loss", "f1", "ablation"])
    ap.add_argument("--pipeline")
    ap.add_argument("--detector", default="efficientad")
    ap.add_argument("--category")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--summary")
    ap.add_argument("--ref")
    ap.add_argument("--resumed")
    a = ap.parse_args()
    out = {
        "loc_modes": lambda: fig_loc_modes(),
        "pi_curve": lambda: fig_pi_curve(a.pipeline),
        "scores": lambda: fig_scores(a.detector, a.category, a.fold),
        "loss": lambda: fig_loss(a.summary),
        "f1": lambda: fig_f1(a.ref, a.resumed),
        "ablation": lambda: fig_ablation(),
    }[a.what]()
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
