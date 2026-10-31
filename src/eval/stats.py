"""쌍대(같은 이미지로 짝지은, paired) 비교 통계: paired bootstrap 95% CI와 McNemar 정확 검정.

두 시스템이 같은 이미지 집합에서 낸 정오(0/1) 벡터를 비교한다. fold 2개라 fold 간 표준편차는 쓰지 않는다.
"""

import numpy as np
from scipy.stats import binomtest


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 10000, seed: int = 0) -> dict:
    """정확도 차이(a - b)의 점추정과 95% 백분위 CI."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    d = a[idx].mean(1) - b[idx].mean(1)
    return {"diff": float(a.mean() - b.mean()), "ci95": [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))]}


def mcnemar(a: np.ndarray, b: np.ndarray) -> dict:
    """불일치 쌍(a만 정답 n10, b만 정답 n01)에 대한 정확 이항 검정."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
    p = 1.0 if n10 + n01 == 0 else float(binomtest(n10, n10 + n01, 0.5).pvalue)
    return {"n10": n10, "n01": n01, "p": p}


def holm(pvals: dict) -> dict:
    """Holm 보정 p값 (ablation 다중 비교용)."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, prev = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, max(prev, (m - i) * p))
        out[k] = adj
        prev = adj
    return out
