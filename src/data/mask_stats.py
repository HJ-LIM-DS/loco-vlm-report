"""세부 결함별 GT 마스크 통계와 loc_mode 초안, 확인용 마스크 시트를 만든다.

이미지마다 결함(픽셀값)별 마스크를 분리해 면적, 연결 성분 수(closing 전후),
3x3 칸 중 면적 최대 칸의 점유율을 계산하고 세부 결함 단위로 집계한다.
초안 규칙은 참고용이며 최종 loc_mode는 시트를 보고 action_kb.yaml에 확정한다.

사용법:
    python -m src.data.mask_stats --out data/defect_mask_stats.csv --sheets reports/phase0/mask_sheets
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from src.data.masks import CATEGORIES, MASK_DOWNSCALE, cell_areas, close_mask, load_defect_masks
from src.kb import defect_table, load_kb


def image_stats(mask: np.ndarray, long_side: int) -> dict:
    closed = close_mask(mask, long_side)
    n_raw = cv2.connectedComponents(mask.astype(np.uint8))[0] - 1
    n_closed = cv2.connectedComponents(closed.astype(np.uint8))[0] - 1
    cells = cell_areas(closed)
    ys, xs = np.nonzero(mask)
    h, w = mask.shape
    bbox_frac = (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1) / (h * w)
    return {
        "area_frac": mask.mean(),
        "n_comp_raw": n_raw,
        "n_comp_closed": n_closed,
        "top_cell_share": cells.max() / max(cells.sum(), 1),
        "n_cells_10pct": int((cells >= 0.1 * cells.sum()).sum()),
        "bbox_frac": bbox_frac,
    }


def draft_loc_mode(name: str, s: pd.Series) -> str:
    """통계 기반 loc_mode 초안 (시트 확인 전 참고용)."""
    if "additional" in name:
        return "instances"
    if s["med_bbox_frac"] >= 0.5 or s["med_top_cell_share"] < 0.5:
        return "global"
    return "point"


def collect(data_root: Path) -> pd.DataFrame:
    rows = []
    for cat in CATEGORIES:
        cdir = data_root / cat
        cfg = json.load(open(cdir / "defects_config.json", encoding="utf-8"))
        pixel_to_name = {d["pixel_value"]: d["defect_name"] for d in cfg}
        for sub in ["logical_anomalies", "structural_anomalies"]:
            for mdir in sorted((cdir / "ground_truth" / sub).iterdir()):
                masks, long_side = load_defect_masks(mdir)
                for v, m in masks.items():
                    rows.append({
                        "category": cat,
                        "folder": sub.split("_")[0],
                        "stem": mdir.name,
                        "defect": pixel_to_name[v],
                        "n_defects_in_image": len(masks),
                        **image_stats(m, long_side),
                    })
    return pd.DataFrame(rows)


def summarize(per_img: pd.DataFrame) -> pd.DataFrame:
    g = per_img.groupby(["category", "defect"])
    s = pd.DataFrame({
        "folders": g["folder"].agg(lambda x: "+".join(sorted(set(x)))),
        "n_images": g.size(),
        "med_area_frac": g["area_frac"].median(),
        "med_n_comp_raw": g["n_comp_raw"].median(),
        "med_n_comp_closed": g["n_comp_closed"].median(),
        "med_top_cell_share": g["top_cell_share"].median(),
        "frac_top_cell_lt_half": g["top_cell_share"].agg(lambda x: (x < 0.5).mean()),
        "med_bbox_frac": g["bbox_frac"].median(),
    }).reset_index()
    s["draft_loc_mode"] = [draft_loc_mode(n, r) for n, (_, r) in zip(s["defect"], s.iterrows())]
    return s.round(3)


def draw_sheets(data_root: Path, per_img: pd.DataFrame, summary: pd.DataFrame,
                out_dir: Path, per_defect: int = 3, rows_per_sheet: int = 5):
    """카테고리별로 세부 결함마다 예시 이미지에 마스크 윤곽선과 3x3 격자를 그린 시트.

    판독 가능한 크기를 위해 결함 rows_per_sheet종씩 나눠 저장한다.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    for cat in CATEGORIES:
        cdir = data_root / cat
        cfg = json.load(open(cdir / "defects_config.json", encoding="utf-8"))
        name_to_pixel = {d["defect_name"]: d["pixel_value"] for d in cfg}
        sm_all = summary[summary["category"] == cat].reset_index(drop=True)
        for start in range(0, len(sm_all), rows_per_sheet):
            sm = sm_all.iloc[start:start + rows_per_sheet].reset_index(drop=True)
            _draw_one_sheet(cdir, cat, per_img, sm, name_to_pixel, per_defect,
                            out_dir / f"{cat}_{start // rows_per_sheet}.jpg", plt)


def _draw_one_sheet(cdir, cat, per_img, sm, name_to_pixel, per_defect, out_path, plt):
    n = len(sm)
    fig, axes = plt.subplots(n, per_defect, figsize=(4.0 * per_defect, 3.4 * n))
    axes = np.atleast_2d(axes)
    for i, r in sm.iterrows():
        ex = per_img[(per_img["category"] == cat) & (per_img["defect"] == r["defect"])]
        # 서로 다른 모습을 보이도록 앞·뒤 예시를 고른다
        picks = ex.iloc[np.linspace(0, len(ex) - 1, per_defect).round().astype(int)]
        for j, (_, e) in enumerate(picks.iterrows()):
            ax = axes[i, j]
            sub = f"{e['folder']}_anomalies"
            img = Image.open(cdir / "test" / sub / f"{e['stem']}.png").convert("RGB")
            w, h = img.size
            img = np.array(img.resize((w // MASK_DOWNSCALE, h // MASK_DOWNSCALE)))
            masks, _ = load_defect_masks(cdir / "ground_truth" / sub / e["stem"])
            m = masks[name_to_pixel[r["defect"]]].astype(np.uint8)
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(img, cnts, -1, (255, 0, 0), 2)
            hh, ww = m.shape
            for k in (1, 2):
                img[hh * k // 3, :] = (255, 255, 0)
                img[:, ww * k // 3] = (255, 255, 0)
            ax.imshow(img)
            ax.set_title(
                f"{r['defect']} [{r['draft_loc_mode']}] {e['stem']}\n"
                f"comp {e['n_comp_closed']} top {e['top_cell_share']:.2f} "
                f"bbox {e['bbox_frac']:.2f}",
                fontsize=8,
            )
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=72)
    plt.close(fig)


def main():
    root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=root / "MVTec-LOCO")
    ap.add_argument("--out", type=Path, default=root / "data" / "defect_mask_stats.csv")
    ap.add_argument("--sheets", type=Path, default=root / "reports" / "phase0" / "mask_sheets")
    ap.add_argument("--no-sheets", action="store_true", help="통계만 계산하고 시트는 그리지 않음")
    args = ap.parse_args()

    per_img = collect(args.data_root)
    summary = summarize(per_img)
    # 시트 확인 후 KB에 확정한 값을 초안 옆에 남긴다 (초안과 다른 결함 추적용)
    kb = load_kb()
    summary["kb_loc_mode"] = [
        defect_table(kb, c)[d]["loc_mode"] for c, d in zip(summary["category"], summary["defect"])
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out, index=False, encoding="utf-8")
    per_img.to_csv(args.out.with_name("defect_mask_stats_per_image.csv"), index=False, encoding="utf-8")
    print(f"저장: {args.out} ({len(summary)}종)")
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(summary.drop(columns=["med_n_comp_raw"]))
    if not args.no_sheets:
        draw_sheets(args.data_root, per_img, summary, args.sheets)
        print(f"시트: {args.sheets}")


if __name__ == "__main__":
    main()
