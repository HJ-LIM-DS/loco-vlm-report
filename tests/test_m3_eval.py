"""M3 채점 테스트: 캐스케이드 요약(RQ4), 종단 F1·오류 분해, open-set 채점(RQ3), 가설 판정."""

import json

import numpy as np
import pandas as pd
import pytest

from src.eval import openset
from src.eval.cascade import (
    _f1_from_rates, compare_e2e, decompose, end_to_end, judge_final, summarize, threshold_from_pgood,
)
from src.kb import load_kb
from src.report.m3_report import verdict

GT = "이상: [구조] 이물 혼입 (중앙)\n조치: 해당 봉투를 격리하고, 포장 구간 이물 유입 경로와 설비 청결 상태를 점검"


def test_judge_final_b1_judge():
    """2차 개선 J1: 판정 구간 거부를 B1에 맡기면 VLM의 "정상"은 더 이상 거부가 아니다."""
    kb = load_kb()
    assert judge_final("judge", "정상", kb, "screw_bag", None) == ""  # 기존 규칙: VLM 정상 -> 거부
    assert judge_final("judge", "정상", kb, "screw_bag", True) == ""  # B1 정상 -> 거부
    forced = judge_final("judge", "정상", kb, "screw_bag", False)  # B1 이상 + VLM 정상 -> 미분류 템플릿
    assert forced.startswith("이상: [미상]")
    assert judge_final("judge", GT, kb, "screw_bag", False) == GT
    assert judge_final("judge", GT, kb, "screw_bag", True) == ""  # B1이 정상이라 하면 VLM 출력이 있어도 거부
    assert judge_final("pass", GT, kb, "screw_bag", False) == ""  # 통과·설명 구간은 심판과 무관
    assert judge_final("describe", "정상", kb, "screw_bag", True).startswith("이상: [미상]")


def test_threshold_from_pgood():
    assert threshold_from_pgood(np.array([])) == 1.0
    p = np.linspace(0, 0.99, 100)
    t = threshold_from_pgood(p, 0.97)
    assert (p < t).mean() >= 0.97 and t <= 0.99  # 97% 이상은 거부되지 않는다


def test_summarize_external_judge_call_rate():
    rows = [
        {"label": "logical", "stage": "describe", "final": GT, "vlm_normal": False, "judge": "b1_argmax", "judge_normal": False, "gt_report": GT},
        {"label": "logical", "stage": "judge", "final": "", "vlm_normal": False, "judge": "b1_argmax", "judge_normal": True, "gt_report": GT},
        {"label": "good", "stage": "judge", "final": "", "vlm_normal": False, "judge": "b1_argmax", "judge_normal": True, "gt_report": "정상"},
        {"label": "good", "stage": "judge", "final": GT, "vlm_normal": True, "judge": "b1_argmax", "judge_normal": False, "gt_report": "정상"},
    ]
    s = summarize(pd.DataFrame(rows))
    # 심판 구간 판단은 B1 기준: 정상 2장 중 1장 거부, 이상 1장 잘못 거부
    assert s["judge_band"] == {"normal_n": 2, "normal_rejected": 0.5, "anomaly_n": 1, "anomaly_wrongly_rejected": 1.0}
    # VLM 호출은 설명 구간 + B1이 이상이라 한 판정 구간: 이상 1/2, 정상 1/2
    assert s["vlm_call_rate"]["pi=0.5"] == pytest.approx(0.5 * 0.5 + 0.5 * 0.5)
    base = summarize(pd.DataFrame([{**r, "judge": "vlm"} for r in rows]))
    assert base["vlm_call_rate"]["pi=0.5"] == pytest.approx(1.0)  # VLM 심판이면 tau_low 이상 전체


def test_cascade_summary():
    df = pd.DataFrame([
        # 이상 4장: 탐지기 통과 1장, 판정 구간에서 VLM이 정상이라 거부 1장, 나머지 출력
        {"label": "logical", "stage": "pass", "final": "", "vlm_normal": False, "gt_report": GT},
        {"label": "logical", "stage": "judge", "final": "", "vlm_normal": True, "gt_report": GT},
        {"label": "structural", "stage": "describe", "final": GT, "vlm_normal": False, "gt_report": GT},
        {"label": "structural", "stage": "judge", "final": GT, "vlm_normal": False, "gt_report": GT},
        # 정상 4장: 판정 구간 2장 중 1장 거부, 탐지기 통과 2장
        {"label": "good", "stage": "judge", "final": "", "vlm_normal": True, "gt_report": "정상"},
        {"label": "good", "stage": "judge", "final": GT, "vlm_normal": False, "gt_report": "정상"},
        {"label": "good", "stage": "pass", "final": "", "vlm_normal": True, "gt_report": "정상"},
        {"label": "good", "stage": "pass", "final": "", "vlm_normal": True, "gt_report": "정상"},
    ])
    s = summarize(df)
    assert s["detector_only"] == {"recall": 0.75, "fpr": 0.5}
    assert s["cascade"] == {"recall": 0.5, "fpr": 0.25}
    assert s["recall_loss_pp"] == 25.0 and s["false_alarm_reduction"] == 0.5 and not s["rq4_hypothesis_met"]
    assert s["judge_band"]["normal_rejected"] == 0.5 and s["end_to_end_group_acc"] == 0.5


def test_openset_scoring(tmp_path, monkeypatch):
    monkeypatch.setattr(openset, "ROOT", tmp_path)
    sft = tmp_path / "data" / "sft" / "s"
    run = tmp_path / "runs" / "r" / "final"
    sft.mkdir(parents=True)
    run.mkdir(parents=True)
    rows = [
        {"id": "a", "label": "structural", "group": "contamination", "openset": True, "gt_report": GT},
        {"id": "b", "label": "structural", "group": "contamination", "openset": True, "gt_report": GT},
        {"id": "c", "label": "structural", "group": "part_broken", "openset": False,
         "gt_report": "이상: [구조] 부품 파손 (중앙)\n조치: 파손 부품을 교체하고, 부품 입고 검사 기준과 이송 시 충격을 점검"},
        {"id": "d", "label": "good", "group": None, "openset": False, "gt_report": "정상"},
    ]
    preds = [
        {"id": "a", "pred": "이상: [구조] 미분류 이상 - 이물 섞임 (중앙)\n조치: 해당 제품을 격리하고, 품질 담당자 확인을 요청"},
        {"id": "b", "pred": "이상: [구조] 부품 파손 (중앙)\n조치: 파손 부품을 교체하고, 부품 입고 검사 기준과 이송 시 충격을 점검"},
        {"id": "c", "pred": "이상: [구조] 미분류 이상 - 깨짐\n조치: 해당 제품을 격리하고, 품질 담당자 확인을 요청"},
        {"id": "d", "pred": "정상"},
    ]
    (sft / "heldout.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    (run / "heldout_predictions.jsonl").write_text("\n".join(json.dumps(p, ensure_ascii=False) for p in preds),
                                                   encoding="utf-8")
    s = openset.score("r", "s")
    assert s["unseen_unknown_rate"] == 0.5 and s["unseen_to_known_group"] == 0.5 and s["unseen_keyword_hit"] == 0.5
    assert s["known_false_unknown"] == 1.0 and s["known_group_acc"] == 0.0 and s["normal_correct"] == 1.0


OTHER = "이상: [구조] 부품 파손 (중앙)\n조치: 파손 부품을 교체하고, 부품 입고 검사 기준과 이송 시 충격을 점검"


def _frame():
    """이상 4장(정답 출력 2, 오답 출력 1, 무출력 1) + 정상 4장(출력 1)."""
    rows = [("logical", "describe", GT, False, GT), ("logical", "judge", GT, False, GT),
            ("structural", "describe", OTHER, False, GT), ("structural", "pass", "", False, GT),
            ("good", "judge", GT, False, "정상"), ("good", "judge", "", True, "정상"),
            ("good", "pass", "", True, "정상"), ("good", "pass", "", False, "정상")]
    return pd.DataFrame([{"id": f"i{k}", "label": a, "stage": b, "final": c, "pred": c or "정상", "vlm_normal": d,
                          "gt_report": e} for k, (a, b, c, d, e) in enumerate(rows)])


def test_end_to_end_f1_matches_counts_at_test_ratio():
    e = end_to_end(_frame(), (0.5, 0.01))
    # 출력 4장 중 정답 그룹 2장: 정밀도 0.5, 정답 그룹 재현율 2/4, 탐지 정밀도 3/4, 탐지 재현율 3/4
    assert e["recall"] == 0.75 and e["fpr"] == 0.25 and e["correct_group_recall"] == 0.5
    assert e["pi=0.5"]["precision"] == pytest.approx(0.75) and e["pi=0.5"]["f1_detect"] == pytest.approx(0.75)
    assert e["pi=0.5"]["precision_group"] == pytest.approx(0.5) and e["pi=0.5"]["f1_group"] == pytest.approx(0.5)
    # π가 작아지면 같은 FPR에서도 정밀도가 떨어진다
    assert e["pi=0.01"]["precision"] < e["pi=0.5"]["precision"]


def test_f1_from_rates_zero_outputs():
    f1_det, f1_c, p_det, p_c = _f1_from_rates(np.array(0.0), np.array(0.0), np.array(0.0), 0.01)
    assert float(f1_det) == 0.0 and float(f1_c) == 0.0


def test_compare_e2e_same_system_is_zero():
    df = _frame()
    d = compare_e2e(df, df.copy(), (0.5,), n_boot=200)
    assert d["pi=0.5"]["f1_group"]["diff"] == 0.0 and d["pi=0.5"]["f1_group"]["ci95"] == [0.0, 0.0]
    with pytest.raises(ValueError):
        compare_e2e(df, df.iloc[1:], (0.5,), n_boot=10)


def test_decompose_partitions_images():
    d = decompose(_frame())
    assert sum(d["anomaly"].values()) == 4 and d["anomaly"]["정답"] == 2
    assert d["anomaly"]["탐지기 놓침(통과 구간)"] == 1 and d["anomaly"]["그룹 오답(설명 구간)"] == 1
    assert sum(d["normal"].values()) == 4 and d["normal"]["VLM 거부(판정 구간)"] == 1
    assert d["normal"]["오경보: 판정 구간 VLM 이상 판단"] == 1


def test_verdict():
    assert verdict({"diff": 0.1}, 0.01) == "지지"
    assert verdict({"diff": 0.1}, 0.2) == "방향 일치, 유의하지 않음"
    assert verdict({"diff": -0.1}, 0.01) == "반대 방향"
    assert verdict({"diff": 0.0}, 1.0) == "효과 없음"
