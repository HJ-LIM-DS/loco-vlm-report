"""GT 마스크 공용 처리: 결함별 마스크 분리, closing, 3x3 칸 면적."""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

CATEGORIES = ["breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors"]
# 마스크 처리 속도를 위한 축소 배율
MASK_DOWNSCALE = 4
# point 위치 라벨의 closing 반경: 이미지 긴 변 대비 비율 (계획 v5)
CLOSE_RADIUS_RATIO = 0.02


def load_defect_masks(mask_dir: Path) -> tuple[dict, int]:
    """마스크 폴더를 읽어 {픽셀값: bool 마스크(축소)}와 원본 긴 변 길이를 반환한다."""
    out = {}
    long_side = 0
    for mp in sorted(mask_dir.glob("*.png")):
        img = Image.open(mp)
        w, h = img.size
        long_side = max(long_side, w, h)
        arr = np.array(img.resize((w // MASK_DOWNSCALE, h // MASK_DOWNSCALE), Image.NEAREST))
        for v in np.unique(arr):
            if v == 0:
                continue
            m = arr == v
            out[int(v)] = m if int(v) not in out else (out[int(v)] | m)
    return out, long_side


def close_mask(mask: np.ndarray, long_side_full: int) -> np.ndarray:
    """긴 변의 2% 반경(원본 해상도 기준)으로 closing한다."""
    r = max(1, round(CLOSE_RADIUS_RATIO * long_side_full / MASK_DOWNSCALE))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, k) > 0


def cell_areas(mask: np.ndarray) -> np.ndarray:
    """3x3 칸별 마스크 픽셀 수 (행 우선 3x3 배열)."""
    h, w = mask.shape
    ys = np.minimum(np.arange(h) * 3 // h, 2)
    xs = np.minimum(np.arange(w) * 3 // w, 2)
    out = np.zeros((3, 3), dtype=np.int64)
    for r in range(3):
        for c in range(3):
            out[r, c] = mask[np.ix_(ys == r, xs == c)].sum()
    return out
