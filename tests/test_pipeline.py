"""2단계 임계값과 캐스케이드 판정 테스트."""

import numpy as np
import pandas as pd

from src.detector.calibrate import thresholds
from src.kb import load_kb
from src.output_spec import parse_report
from src.pipeline import decide, finalize

KB = load_kb()


def test_thresholds_recall_and_fpr():
    rng = np.random.default_rng(0)
    normal = rng.normal(0, 1, 200)
    anom = rng.normal(3, 1, 400)
    s = pd.Series(np.concatenate([normal, anom]))
    lab = pd.Series(["good"] * 200 + ["logical"] * 400)
    th = thresholds(s, lab)
    assert (anom >= th["tau_low"]).mean() >= 0.95
    assert (normal >= th["tau_high"]).mean() <= 0.01
    assert th["tau_high"] >= th["tau_low"] and th["tau_high_rule"] == "fpr1"


def test_thresholds_few_normals_uses_margin():
    s = pd.Series([0.1, 0.2, 0.3, 5.0, 6.0, 7.0])
    lab = pd.Series(["good"] * 3 + ["structural"] * 3)
    th = thresholds(s, lab)
    assert th["tau_high_rule"] == "normal_max_margin" and th["tau_high"] > 0.3


def test_decide_stages():
    th = {"tau_low": 1.0, "tau_high": 2.0}
    assert [decide(x, th) for x in (0.5, 1.5, 2.5)] == ["pass", "judge", "describe"]


def test_finalize_rejects_only_in_judge_band():
    assert finalize("judge", "정상", KB, "screw_bag", "중앙") == ("", [])
    out, steps = finalize("describe", "정상", KB, "screw_bag", "중앙")
    assert steps == ["forced_describe"] and parse_report(out).location == "중앙"
    assert finalize("pass", None, KB, "screw_bag", None) == ("", [])
