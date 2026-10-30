"""P3 2단계 자동 선별: 대기열의 enabled 표식 줄만 바꾸고 주석은 보존하는지."""

from src.report.p3_select import apply_queue, set_enabled

QUEUE = """runs:
  # 주석은 그대로
  - name: p3_4b_f0_terms
    enabled: true   # p3_select:terms
    cmd: [python]
  - name: p3_4b_f0_hints
    enabled: true   # p3_select:hints
    cmd: [python]
  - name: other
    enabled: false
"""


def test_set_enabled_touches_only_marked_lines():
    out = set_enabled(QUEUE, "hints", False)
    assert "enabled: false   # p3_select:hints" in out
    assert "enabled: true   # p3_select:terms" in out
    assert "  - name: other\n    enabled: false\n" in out and "# 주석은 그대로" in out
    assert set_enabled(out, "hints", True) == QUEUE


def test_apply_queue_selected_and_none(tmp_path):
    q = tmp_path / "run_queue.yaml"
    q.write_text(QUEUE, encoding="utf-8")
    assert apply_queue(q, "terms") == {"terms": True, "hints": False}
    text = q.read_text(encoding="utf-8")
    assert "enabled: true   # p3_select:terms" in text and "enabled: false   # p3_select:hints" in text
    assert apply_queue(q, None) == {"terms": False, "hints": False}
    assert "enabled: true   # p3_select" not in q.read_text(encoding="utf-8")
