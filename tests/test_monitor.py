"""모니터 표시 테스트: 단계 항상 표시, 'i/n' 단계의 진행 막대·속도·남은 시간."""

import json
import time

from rich.console import Console

from ops import monitor


def _render(panel) -> str:
    c = Console(record=True, width=120, color_system=None)
    c.print(panel)
    return c.export_text()


def test_phase_rate():
    r = monitor.PhaseRate()
    assert r.rate(("a", 1, "p"), 10, 100.0) is None
    assert r.rate(("a", 1, "p"), 20, 110.0) == 1.0
    assert r.rate(("a", 1, "p"), 5, 120.0) is None  # 단계가 다시 시작되면 기준점을 새로 잡는다


def test_generation_phase_shows_progress(tmp_path, monkeypatch):
    monkeypatch.setattr(monitor, "PROGRESS_DIR", tmp_path)
    monkeypatch.setattr(monitor, "phase_rate", monitor.PhaseRate())
    state = {"runs": {"r": {"status": "running", "attempts": 1, "pid": None, "started_at": time.time()}}}
    now = time.time()
    for done, t in ((100, now - 60), (130, now)):
        (tmp_path / "r.json").write_text(json.dumps(
            {"attempt": 1, "step": 213, "total_steps": 213, "phase": f"predict_heldout {done}/783",
             "updated_at": t}), encoding="utf-8")
        text = _render(monitor.current_panel(state, {}, {}))
    assert "held-out 생성 평가" in text and "학습 step 213/213 완료" in text
    assert "130/783" in text and "30.0장/분" in text and "남은 시간" in text


def test_training_phase_without_counter(tmp_path, monkeypatch):
    monkeypatch.setattr(monitor, "PROGRESS_DIR", tmp_path)
    state = {"runs": {"r": {"status": "running", "attempts": 1, "pid": None, "started_at": time.time()}}}
    (tmp_path / "r.json").write_text(json.dumps(
        {"attempt": 1, "step": 50, "total_steps": 213, "phase": "train", "it_s": 0.03, "eta_s": 5400,
         "updated_at": time.time()}), encoding="utf-8")
    text = _render(monitor.current_panel(state, {}, {}))
    assert "단계 학습" in text and "학습 step 50/213" in text and "장/분" not in text
