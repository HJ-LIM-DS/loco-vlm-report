"""PDF 그림1(시스템 구조 + 실제 사례 1건)과 그림4(held-out 성공 3건, 실패 3건).

사례 선택 규칙(결과를 보고 고르지 않도록 고정): 모든 사례는 그 이미지를 학습하지 않은 fold 모델의 예측이다.
- 출력은 Ours 4B seed 0 캐스케이드(tau_low 학습 측 재현율 0.95)의 최종 출력(Guard 후, 무출력 포함).
- 성공 3건: `reports/tables/m3_failure.json` 사례 후보(세 seed 모두 정답 우선, 카테고리별 id 순)에서
  논리·구조·논리 순으로 카테고리가 겹치지 않게.
- 실패 3건: 실패 유형이 서로 다르게 - 개수 계열 그룹 혼동, crop이 결함을 놓침, 탐지기 놓침(무출력).
  각 유형 안에서 앞서 고른 사례와 카테고리가 겹치지 않는 첫 이미지.
- 그림1 사례는 성공 사례 중 개수 계열(없으면 첫 성공 사례).

    python -m src.report.m4_figures
"""

import json
import textwrap
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from src.data.masks import load_defect_masks
from src.detector.cache import FoldCalibration
from src.eval.cascade import cascade
from src.eval.failure import COUNT_GROUPS
from src.kb import load_kb
from src.output_spec import parse_report
from src.report import style
from src.report.m2_report import heldout
from src.report.style import ANOMALY_COLOR, NORMAL_COLOR, TEXT, TEXT_2, save, setup
from src.vlm.build_sft import expand_box

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "MVTec-LOCO"
CAT_KO = {"breakfast_box": "아침식사 박스", "juice_bottle": "주스 병", "pushpins": "푸시핀",
          "screw_bag": "나사 봉투", "splicing_connectors": "커넥터"}


def system_outputs() -> pd.DataFrame:
    kb = load_kb()
    index = pd.read_parquet(ROOT / "data" / "index.parquet")
    splits = pd.read_csv(ROOT / "data" / "splits.csv")
    df = cascade({f: heldout(f"m2_4b_f{f}_s0") for f in (0, 1)}, 0.95, kb, index, splits)
    df["correct"] = [lab != "good" and bool(f) and parse_report(f) is not None and parse_report(f).group == parse_report(g).group
                     for lab, f, g in zip(df["label"], df["final"], df["gt_report"])]
    return df.set_index("id")


def pick_cases(df: pd.DataFrame) -> list:
    cand = json.loads((ROOT / "reports" / "tables" / "m3_failure.json").read_text(encoding="utf-8"))["case_candidates"]
    cat = lambda i: i.split("/")[0]  # noqa: E731
    used, out = set(), []

    def take(ids, kind, label):
        for i in ids:
            if cat(i) not in used and i in df.index:
                used.add(cat(i))
                out.append((i, kind, label))
                return

    for key in ("성공_논리", "성공_구조", "성공_논리"):
        take([i for i in cand[key] if df.loc[i, "correct"]], "success", "성공")
    used.clear()
    take([i for i in cand["실패_개수_다른그룹"] if not df.loc[i, "correct"]], "fail", "실패: 수량 결함을 다른 그룹으로")
    take([i for i in cand["실패_crop_놓침"] if not df.loc[i, "correct"]], "fail", "실패: crop이 결함을 놓침")
    miss = df[(df["label"] != "good") & (df["stage"] == "pass")].sort_index().index
    take(list(miss), "fail", "실패: 탐지기 놓침(무출력)")
    return out


def overlay(image_id: str, fold_of: pd.Series, idx: pd.DataFrame, width: int = 640) -> np.ndarray:
    """검사 이미지 + GT 결함 윤곽(주황) + 탐지기 crop 상자(파랑)."""
    r = idx.loc[image_id]
    img = Image.open(DATA / r["image_path"]).convert("RGB")
    w0, h0 = img.size
    s = width / w0
    arr = np.array(img.resize((width, round(h0 * s)), Image.BILINEAR))
    h, w = arr.shape[:2]
    hexrgb = lambda c: tuple(int(c[i:i + 2], 16) for i in (1, 3, 5))  # noqa: E731
    if isinstance(r["mask_dir"], str) and r["label"] != "good":
        masks, _ = load_defect_masks(DATA / r["mask_dir"])
        m = np.zeros_like(next(iter(masks.values())))
        for v in masks.values():
            m |= v
        m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(arr, cnts, -1, hexrgb(ANOMALY_COLOR), 3)
    model_fold = 1 - int(fold_of[image_id])
    sp = pd.read_csv(ROOT / "data" / "splits.csv").merge(idx[["category", "label", "split"]], left_on="image_id", right_index=True)
    normal = sp[(sp["category"] == r["category"]) & (sp["split"] == "test") & (sp["fold"] == model_fold)
                & (sp["label"] == "good")]["image_id"].tolist()
    region, _ = FoldCalibration("salad", r["category"], normal).region(image_id, (h, w))
    if region is not None:
        ys, xs = np.nonzero(region)
        x0, y0, x1, y1 = expand_box(xs.min(), ys.min(), xs.max() + 1, ys.max() + 1, w, h)
        cv2.rectangle(arr, (x0, y0), (x1 - 1, y1 - 1), hexrgb(NORMAL_COLOR), 3)
    return letterbox(arr, hexrgb(style.SURFACE))


def letterbox(arr: np.ndarray, fill, aspect: float = 1.35) -> np.ndarray:
    """종횡비가 다른 카테고리 이미지를 같은 틀(가로/세로 = aspect)에 가운데 정렬해 칸을 맞춘다."""
    h, w = arr.shape[:2]
    H, W = (h, round(h * aspect)) if w / h < aspect else (round(w / aspect), w)
    out = np.empty((H, W, 3), dtype=arr.dtype)
    out[:] = fill
    y, x = (H - h) // 2, (W - w) // 2
    out[y:y + h, x:x + w] = arr
    return out


def first_line(text: str) -> str:
    return text.splitlines()[0] if text else "(무출력)"


def wrap(text: str, n: int = 21) -> str:
    return "\n".join(textwrap.wrap(text, n, break_long_words=True))


def fig_cases(df: pd.DataFrame, cases: list):
    import matplotlib.pyplot as plt

    idx = pd.read_parquet(ROOT / "data" / "index.parquet").set_index("image_id")
    fold_of = pd.read_csv(ROOT / "data" / "splits.csv").set_index("image_id")["fold"]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.6), gridspec_kw={"hspace": 0.75, "wspace": 0.06})
    for ax, (iid, kind, label) in zip(axes.ravel(), cases):
        ax.imshow(overlay(iid, fold_of, idx))
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.grid(False)
        r = df.loc[iid]
        ax.set_title(f"{label}\n{CAT_KO[idx.loc[iid, 'category']]} {iid.split('/')[-1]}", loc="left", fontsize=8,
                     color=TEXT if kind == "success" else ANOMALY_COLOR)
        txt = f"정답: {wrap(first_line(r['gt_report']).removeprefix('이상: '))}\n" \
              f"출력: {wrap(first_line(r['final']).removeprefix('이상: '))}"
        ax.text(0, -0.04, txt, transform=ax.transAxes, va="top", ha="left", fontsize=6.5, color=TEXT_2, linespacing=1.3)
    fig.text(0.01, 0.995, "주황 윤곽 = GT 결함 영역, 파랑 상자 = 탐지기 crop 영역. 출력 = Ours 4B 캐스케이드 최종 출력(held-out).",
             fontsize=7, color=TEXT_2, va="top")
    return save(fig, "cases.png")


def fig_system(df: pd.DataFrame, case: str):
    """그림1: 구조도(왼쪽) + 실제 사례의 VLM 입력 3장과 2줄 출력(오른쪽)."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fold_of = pd.read_csv(ROOT / "data" / "splits.csv").set_index("image_id")["fold"]
    model_fold = 1 - int(fold_of[case])
    rows = {json.loads(x)["id"]: json.loads(x) for x in
            (ROOT / "data" / "sft" / f"m2_salad_f{model_fold}" / "heldout.jsonl").read_text(encoding="utf-8").splitlines()}
    rec = rows[case]
    fig = plt.figure(figsize=(7.2, 3.3))
    ax = fig.add_axes([0, 0, 0.56, 1])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    def box(x, y, w, h, title, body, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.25", linewidth=1.2,
                                    edgecolor=color, facecolor=style.SURFACE))
        ax.text(x + 0.2, y + h - 0.25, title, fontsize=7.5, fontweight="bold", color=TEXT, va="top")
        ax.text(x + 0.2, y + h - 0.95, body, fontsize=6.3, color=TEXT_2, va="top", linespacing=1.35)

    def arrow(x0, y0, x1, y1, label=None):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9, color=TEXT_2, linewidth=1))
        if label:
            ax.text((x0 + x1) / 2 + 0.1, (y0 + y1) / 2, label, fontsize=6, color=TEXT_2, va="center")

    box(0.2, 7.3, 4.0, 2.4, "검사 이미지 + 카테고리", "카테고리별 탐지기·임계값\n(라인별 제품은 알려져 있다고 가정)", TEXT_2)
    box(0.2, 3.7, 4.0, 2.9, "Stage 1  SALAD 탐지기", "train/good만 학습, 공식 가중치 재현\n이미지 점수(국소+Mahalanobis+구성)\n"
        "이상 맵 → crop, 의심 칸\n지연 약 111ms/장", NORMAL_COLOR)
    box(0.2, 0.2, 4.0, 2.8, "2단계 임계값 (fold별)", "score < τ_low: 무출력\nτ_low ~ τ_high: VLM 판정\n  (\"정상\"이면 무출력 = 오탐 거부)\n"
        "score ≥ τ_high: VLM 설명(거부 불가)", TEXT_2)
    box(5.4, 4.5, 4.4, 5.2, "Stage 2  Qwen3-VL-4B + QLoRA", "NF4, LoRA r=16(q/k/v/o+MLP)\n입력 이미지 3장(각 약 256토큰)\n"
        " 1 검사 이미지  2 crop(윤곽선)\n 3 정상 참조(DINOv2 최근접)\n텍스트: 정상 규격, 결함 그룹 목록,\n"
        " 탐지기 백분위·의심 위치\n학습: 이상 2줄 + 정상 하드\n 네거티브 150, 목록 드롭아웃 12%", ANOMALY_COLOR)
    box(5.4, 0.2, 4.4, 2.8, "Guard + 출력", "KB 대조: 세부 → 그룹 → 조치 교정\n파싱 실패 시 미분류 템플릿\n"
        "이상: [유형] 그룹 - 세부 (위치)\n조치: KB 조치 문장", TEXT_2)
    arrow(2.2, 7.3, 2.2, 6.6)
    arrow(2.2, 3.7, 2.2, 3.0)
    arrow(4.2, 1.6, 5.4, 5.6, "판정·설명 구간")
    arrow(7.6, 4.5, 7.6, 3.0)

    from src.vlm.prompts import IMAGE_ROLES
    for k, (p, role) in enumerate(zip(rec["images"], IMAGE_ROLES)):
        a = fig.add_axes([0.585 + k * 0.138, 0.50, 0.13, 0.36])
        a.imshow(Image.open(ROOT / p).convert("RGB"))
        a.axis("off")
        a.set_title(role.replace("(윤곽선 표시)", ""), fontsize=6.5, color=TEXT_2)
    r = df.loc[case]
    t = fig.add_axes([0.585, 0.02, 0.41, 0.42])
    t.axis("off")
    det = rec.get("detector") or {}
    out = r["final"] or "(무출력)"
    t.text(0, 1, f"실제 사례 (held-out): {CAT_KO[r['category']]} {case.split('/')[-1]}\n"
                 f"탐지기 백분위 {det.get('percentile', 0):.0f}, 의심 위치 {det.get('location', '-')}, 단계 {r['stage']}",
           fontsize=6.5, color=TEXT_2, va="top", linespacing=1.4)
    t.text(0, 0.62, "\n".join(wrap(line, 30) for line in out.splitlines()), fontsize=7, color=TEXT, va="top",
           linespacing=1.4)
    return save(fig, "system.png")


def main():
    setup()
    df = system_outputs()
    cases = pick_cases(df)
    print("사례:", cases)
    succ = [i for i, k, _ in cases if k == "success"]
    idx = pd.read_parquet(ROOT / "data" / "index.parquet").set_index("image_id")
    count = [i for i in succ if (idx.loc[i, "category"], idx.loc[i, "groups"][0]) in COUNT_GROUPS]
    print(f"저장: {fig_cases(df, cases)}")
    print(f"저장: {fig_system(df, (count or succ)[0])}")
    (ROOT / "reports" / "tables" / "figure_cases.json").write_text(
        json.dumps([{"id": i, "kind": k, "label": lab} for i, k, lab in cases], ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
