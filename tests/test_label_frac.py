"""P2 라벨 수 곡선: 학습 측 층별 부분추출 (src.vlm.build_sft.subsample_labels)."""

import numpy as np
import pandas as pd

from src.vlm.build_sft import subsample_labels


def _pool():
    rows = []
    for role in ("train", "val", "heldout"):
        for cat in ("a", "b"):
            for g, n in (("good", 8), ("g1", 12), ("g2", 3), ("g3", 1)):
                for k in range(n):
                    rows.append({"image_id": f"{cat}/{role}/{g}/{k:03d}", "role": role, "category": cat,
                                 "groups": np.array([] if g == "good" else [g], dtype=object)})
    return pd.DataFrame(rows)


def test_full_fraction_keeps_all():
    assert subsample_labels(_pool(), 1.0) is None


def test_stratified_min_one_and_heldout_untouched():
    pool = _pool()
    keep = subsample_labels(pool, 0.25)
    assert not any("/heldout/" in i for i in keep)
    for role in ("train", "val"):
        for cat in ("a", "b"):
            for g, n in (("good", 8), ("g1", 12), ("g2", 3), ("g3", 1)):
                got = [i for i in keep if i.startswith(f"{cat}/{role}/{g}/")]
                assert len(got) == max(1, round(0.25 * n))


def test_nested_and_deterministic():
    pool = _pool()
    k25, k50 = subsample_labels(pool, 0.25), subsample_labels(pool, 0.5)
    assert k25 <= k50
    assert k25 == subsample_labels(pool.sample(frac=1, random_state=3), 0.25)  # 행 순서와 무관
