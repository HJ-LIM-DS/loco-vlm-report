"""Guard 단계별 교정 테스트."""

from src.kb import load_kb
from src.output_spec import UNKNOWN_ACTION, format_report, gt_report, parse_report
from src.vlm.guard import guard

KB = load_kb()
GOOD = format_report(gt_report(KB, "screw_bag", ["1_additional_nut"], None))
ACTION = KB["screw_bag"]["groups"]["part_count"]["action"]


def test_valid_output_unchanged():
    out, steps = guard(GOOD, KB, "screw_bag")
    assert out == GOOD and steps == []


def test_normal_passthrough():
    assert guard("정상", KB, "screw_bag") == ("정상", [])


def test_step1_unknown_detail_removed():
    text = f"이상: [논리] 부품 구성 수량 불일치 - 볼트 9개 초과\n조치: {ACTION}"
    out, steps = guard(text, KB, "screw_bag")
    assert steps == ["1_detail"] and parse_report(out).details == ()


def test_step1_type_contradiction_removed():
    text = f"이상: [구조] 부품 구성 수량 불일치 - 너트 1개 초과\n조치: {ACTION}"
    out, steps = guard(text, KB, "screw_bag")
    assert "1_detail" in steps


def test_step1_exclusive_group_multi_detail_removed():
    act = KB["breakfast_box"]["groups"]["fruit_count"]["action"]
    text = f"이상: [논리] 과일 구성 수량 불일치 - 천도복숭아 2개, 귤 1개 + 과일 전부 없음\n조치: {act}"
    out, steps = guard(text, KB, "breakfast_box")
    assert steps == ["1_detail"]


def test_step2_unknown_group():
    text = f"이상: [논리] 나사 색 변질 (중앙)\n조치: {ACTION}"
    out, steps = guard(text, KB, "screw_bag")
    r = parse_report(out)
    assert steps == ["2_group"] and r.type == "미상" and r.location == "중앙" and r.action == UNKNOWN_ACTION


def test_step3_action_replaced():
    text = "이상: [논리] 부품 구성 수량 불일치 - 너트 1개 초과\n조치: 아무렇게나 처리"
    out, steps = guard(text, KB, "screw_bag")
    assert steps == ["3_action"] and out == GOOD


def test_step4_parse_fail_uses_detector_cell():
    out, steps = guard("너트가 하나 더 있습니다", KB, "screw_bag", fallback_location="우측 하단")
    assert steps == ["4_parse_fail"]
    assert out.split("\n")[0] == "이상: [미상] 미분류 이상 (우측 하단)"


def test_unknown_label_free_text_trimmed():
    text = "이상: [논리] 미분류 이상 - " + "가" * 30 + "\n조치: 격리"
    out, steps = guard(text, KB, "screw_bag")
    r = parse_report(out)
    assert len(r.details[0]) == 20 and r.action == UNKNOWN_ACTION
