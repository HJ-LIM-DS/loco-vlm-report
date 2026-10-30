"""탐지 결과 캐시 형식과 fold별 보정 (탐지기 공통).

    outputs/detector/<detector>/<category>/scores.csv   image_id, score (+ 탐지기별 부가 항)
    outputs/detector/<detector>/<category>/maps/<key>.npy   이상 맵 (float16, 탐지기 해상도)

보정은 fold의 학습 측 test/good만 쓴다 (validation/good은 탐지기 정규화에 쓰여 점수가 in-sample).
- 점수 백분위: 학습 측 정상 점수 분포 대비
- 맵 기준: 학습 측 정상 맵 픽셀값의 99.9 분위 (카테고리별 고정, 이미지별 min-max 금지)
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.output_spec import GRID_KO

ROOT = Path(__file__).resolve().parents[2]
DET_ROOT = ROOT / "outputs" / "detector"
MAP_QUANTILE = 0.999


def map_key(image_id: str) -> str:
    return image_id.replace("/", "__")


def det_dir(detector: str, category: str) -> Path:
    return DET_ROOT / detector / category


def load_scores(detector: str, category: str) -> pd.DataFrame:
    return pd.read_csv(det_dir(detector, category) / "scores.csv")


def load_map(detector: str, image_id: str) -> np.ndarray:
    cat = image_id.split("/", 1)[0]
    return np.load(det_dir(detector, cat) / "maps" / f"{map_key(image_id)}.npy").astype(np.float32)


class FoldCalibration:
    """한 카테고리·한 fold의 학습 측 정상 이미지로 점수 백분위와 맵 기준을 정한다."""

    TERMS = {"local": "s_local", "mah": "s_mah", "comp": "s_comp"}  # SALAD 세 항 (P3 변형 A)

    def __init__(self, detector: str, category: str, normal_ids: list):
        self.detector = detector
        table = load_scores(detector, category).set_index("image_id")
        s = table["score"]
        self.normal_scores = np.sort(s.loc[normal_ids].to_numpy())
        self.scores = s
        self.terms = {k: table[c] for k, c in self.TERMS.items() if c in table.columns}
        self.normal_terms = {k: np.sort(v.loc[normal_ids].to_numpy()) for k, v in self.terms.items()}
        pix = np.concatenate([load_map(detector, i).ravel() for i in normal_ids])
        self.map_thr = float(np.quantile(pix, MAP_QUANTILE))

    def percentile(self, image_id: str) -> float:
        v = self.scores.loc[image_id]
        return 100.0 * np.searchsorted(self.normal_scores, v, side="right") / len(self.normal_scores)

    def term_percentiles(self, image_id: str) -> dict:
        """세 항 각각의 학습 측 정상 대비 백분위 {local, mah, comp} (탐지기에 그 항이 없으면 빈 dict)."""
        return {k: 100.0 * np.searchsorted(self.normal_terms[k], self.terms[k].loc[image_id], side="right")
                / len(self.normal_terms[k]) for k in self.terms}

    def region(self, image_id: str, out_hw: tuple[int, int]):
        """기준을 넘는 최대 연결 영역 마스크(out_hw 해상도)와 최댓값 위치의 3x3 칸. 없으면 (None, 칸)."""
        m = cv2.resize(load_map(self.detector, image_id), (out_hw[1], out_hw[0]), interpolation=cv2.INTER_LINEAR)
        r, c = np.unravel_index(int(m.argmax()), m.shape)
        cell = GRID_KO[min(3 * r // m.shape[0], 2)][min(3 * c // m.shape[1], 2)]
        above = (m > self.map_thr).astype(np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(above, connectivity=8)
        if n <= 1:
            return None, cell
        k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        return lab == k, cell
