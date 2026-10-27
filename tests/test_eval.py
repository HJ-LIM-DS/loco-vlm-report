"""채점기 테스트: 정답 그대로의 예측은 만점, B0는 형식만 맞고 정확도는 낮아야 한다."""

import json
from pathlib import Path

import pandas as pd
import pytest

from src.eval.explanation import evaluate

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "data" / "index.parquet"
pytestmark = pytest.mark.skipif(not INDEX.exists(), reason="data/index.parquet 없음")


def _write(tmp_path, rows, name="p.jsonl"):
    p = tmp_path / name
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    return p


def test_perfect_predictions(tmp_path):
    df = pd.read_parquet(INDEX)
    test = df[df["split"] == "test"]
    p = _write(tmp_path, [{"id": r.image_id, "pred": r.gt_report} for r in test.itertuples()])
    m = evaluate([p])["raw"]
    for k in ("format_ok", "group_acc", "type_acc", "detail_exact", "location_exact", "action_consistency",
              "normal_correct"):
        assert m[k] == pytest.approx(1.0), k
    assert m["severity_miss"] == 0.0 and m["normal_miss"] == 0.0
    assert m["location_hit"] > 0.95  # 라벨 칸(closing 후 최대)의 원 마스크 비율이 10% 미만인 경우만 빠짐


def test_duplicate_ids_rejected(tmp_path):
    p = _write(tmp_path, [{"id": "pushpins/test/good/000", "pred": "정상"}] * 2)
    with pytest.raises(ValueError):
        evaluate([p])


def test_majority_baseline_scores_low(tmp_path):
    from src.baselines.majority import majority_reports
    from src.kb import load_kb

    kb = load_kb()
    df = pd.read_parquet(INDEX).merge(pd.read_csv(ROOT / "data" / "splits.csv")[["image_id", "fold"]])
    reports = majority_reports(df[df["fold"] == 0], kb)
    held = df[(df["split"] == "test") & (df["fold"] == 1)]
    p = _write(tmp_path, [{"id": r.image_id, "pred": reports[r.category]} for r in held.itertuples()])
    m = evaluate([p])["raw"]
    assert m["format_ok"] == 1.0 and m["normal_correct"] == 0.0
    assert 0.0 < m["group_acc"] < 0.5


def test_stats_paired():
    import numpy as np

    from src.eval.stats import holm, mcnemar, paired_bootstrap

    a = np.array([1] * 80 + [0] * 20)
    b = np.array([1] * 60 + [0] * 40)
    bt = paired_bootstrap(a, b)
    assert bt["diff"] == pytest.approx(0.2) and bt["ci95"][0] > 0
    mc = mcnemar(a, b)
    assert mc["n10"] == 20 and mc["n01"] == 0 and mc["p"] < 1e-4
    adj = holm({"x": 0.01, "y": 0.04, "z": 0.03})
    assert adj["x"] == pytest.approx(0.03) and adj["y"] == pytest.approx(0.06) and adj["z"] == pytest.approx(0.06)
