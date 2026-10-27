"""2단계 임계값 (fold별, 카테고리별). 학습 측 test/good과 학습 측 이상만 쓴다.

- validation/good은 탐지기 정규화에 쓰여 점수가 in-sample이므로 제외한다.
- tau_low : 학습 측 이상 재현율이 0.95가 되는 점수 (놓침 최소화, 오탐은 VLM이 거른다)
- tau_high: 학습 측 정상 FPR 1% 지점. 정상 표본이 적어 1%를 만족할 수 없으면 정상 최댓값에
            정상 점수 표준편차의 10%를 더한다. 항상 tau_high >= tau_low.
"""

import numpy as np
import pandas as pd

from src.detector.cache import load_scores

TARGET_RECALL = 0.95
TARGET_FPR = 0.01


def thresholds(scores: pd.Series, labels: pd.Series, recall: float = TARGET_RECALL) -> dict:
    """scores/labels: 같은 인덱스의 학습 측 test 이미지 점수와 라벨(good/logical/structural)."""
    normal = np.sort(scores[labels == "good"].to_numpy())
    anom = np.sort(scores[labels != "good"].to_numpy())
    # 재현율 0.95: 이상 점수의 하위 5% 분위
    tau_low = float(np.quantile(anom, 1 - recall, method="lower"))
    k = int(np.floor(TARGET_FPR * len(normal)))
    if k >= 1:
        tau_high = float(normal[len(normal) - k])  # 이 값 이상인 정상이 k개 이하 (FPR <= 1%)
        rule = "fpr1"
    else:
        tau_high = float(normal[-1] + 0.1 * normal.std())
        rule = "normal_max_margin"
    tau_high = max(tau_high, tau_low)
    return {"tau_low": tau_low, "tau_high": tau_high, "tau_high_rule": rule,
            "n_normal": int(len(normal)), "n_anomaly": int(len(anom))}


def fold_thresholds(detector: str, category: str, index: pd.DataFrame, splits: pd.DataFrame, fold: int,
                    recall: float = TARGET_RECALL) -> dict:
    s = load_scores(detector, category).set_index("image_id")["score"]
    df = index.merge(splits[["image_id", "fold"]], on="image_id")
    side = df[(df["category"] == category) & (df["split"] == "test") & (df["fold"] == fold)]
    return thresholds(s.loc[side["image_id"]].reset_index(drop=True), side["label"].reset_index(drop=True), recall)
