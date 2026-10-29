"""M4 표·그림 보조 함수: 사례 칸 맞춤(letterbox), 줄바꿈, 비용 집계의 재개 run 처리."""

import json

import numpy as np

from src.report import m4_tables
from src.report.m4_figures import letterbox, wrap


def test_letterbox_keeps_image_and_aspect():
    tall = np.full((100, 40, 3), 7, np.uint8)
    out = letterbox(tall, (1, 2, 3), aspect=1.35)
    assert out.shape[0] == 100 and abs(out.shape[1] / out.shape[0] - 1.35) < 0.01
    assert (out == 7).all(axis=2).sum() == 100 * 40  # 원본 픽셀은 그대로, 나머지는 채움색
    wide = np.zeros((40, 100, 3), np.uint8)
    assert abs(letterbox(wide, (0, 0, 0)).shape[1] / letterbox(wide, (0, 0, 0)).shape[0] - 1.35) < 0.02


def test_wrap_width():
    assert all(len(x) <= 21 for x in wrap("가" * 50).splitlines())


def test_cost_skips_resumed_runtime(tmp_path, monkeypatch):
    monkeypatch.setattr(m4_tables, "ROOT", tmp_path)
    for f, rt in ((0, 7200.0), (1, 30.0)):  # fold 1은 생성 단계만 다시 한 재개 run
        d = tmp_path / "runs" / f"x_f{f}" / "final"
        d.mkdir(parents=True)
        d.joinpath("summary.json").write_text(json.dumps({
            "train_runtime": rt, "peak_vram_reserved_gb": 6.0 + f, "heldout_generation": {"seconds_per_image": 5.0 + f}}))
    c = m4_tables.cost("x_f{f}", [None])
    assert c["train_h"] == 2.0 and c["vram_gb"] == 7.0 and c["sec_per_image"] == 5.5


def test_portfolio_template_renders(tmp_path, monkeypatch):
    """원고 템플릿의 모든 변수가 결과 파일로 채워지는지 (StrictUndefined). Edge 인쇄는 하지 않는다."""
    import pytest

    from src.report import build_portfolio as bp
    need = [bp.TABLES / f for f in ("table1_main.json", "m2_summary.json", "m3_summary.json", "m3_cascade.json",
                                    "m3_failure.json")]
    if not all(p.exists() for p in need) or not (bp.ROOT / "outputs" / "detector" / "salad").exists():
        pytest.skip("결과 파일 없음 (새 환경)")
    monkeypatch.setattr(bp, "HTML", tmp_path / "p.html")
    html = bp.render().read_text(encoding="utf-8")
    assert "{{" not in html and "음성 결과" in html and "표 1." in html
