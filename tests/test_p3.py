"""P3 입력 표현 변형(탐지기 세 항 백분위, B1 후보 힌트)의 프롬프트·힌트 합산 시험."""

import json

import numpy as np
import pytest

from src.kb import load_kb
from src.output_spec import NORMAL
from src.vlm.build_sft import group_hints, load_hint_probs
from src.vlm.prompts import detector_line, hint_line, user_text


def test_detector_line_with_terms():
    base = detector_line({"percentile": 97.4, "location": "중앙"})
    assert base == "[탐지기] 이상 점수 백분위 97 (정상 이미지 대비), 의심 위치: 중앙"
    with_terms = detector_line({"percentile": 97.4, "location": "중앙", "terms": {"local": 92.2, "mah": 88.0, "comp": 99.6}})
    assert with_terms == "[탐지기] 이상 점수 백분위 97 (정상 이미지 대비), 세부 백분위 국소 92 / 전역 88 / 구성 100, 의심 위치: 중앙"
    assert detector_line(None) is None


def test_hint_line_and_prompt_placement():
    assert hint_line(None) is None and hint_line([]) is None
    assert hint_line([("칸막이 누락", 0.623), (NORMAL, 0.2)]) == "[분류기 후보] 칸막이 누락 0.62, 정상 0.20"
    kb = load_kb()
    groups = [g["ko_label"] for g in kb["pushpins"]["groups"].values()]
    text = user_text(kb, "pushpins", groups, {"percentile": 50.0, "location": None}, hints=[("칸막이 누락", 0.6)])
    lines = text.splitlines()
    i_det = next(i for i, x in enumerate(lines) if x.startswith("[탐지기]"))
    assert lines[i_det + 1] == "[분류기 후보] 칸막이 누락 0.60"
    assert "[분류기 후보]" not in user_text(kb, "pushpins", groups, {"percentile": 50.0, "location": None})


def test_group_hints_aggregates_by_group_and_filters():
    kb = load_kb()
    groups = kb["pushpins"]["groups"]
    gid_a, gid_b = list(groups)[:2]
    da = list(groups[gid_a]["defects"])
    db = list(groups[gid_b]["defects"])[0]
    probs = {"good": 0.3, da[0]: 0.25, (da[1] if len(da) > 1 else da[0]): 0.2, db: 0.25}
    # 같은 그룹의 세부 결함 확률은 합산된다 (다른 세부가 2개면 0.45, 1개면 같은 키라 0.2만 남음)
    hints = group_hints(kb, "pushpins", probs, None, k=3)
    labels = [g for g, _ in hints]
    assert groups[gid_a]["ko_label"] in labels and NORMAL in labels and groups[gid_b]["ko_label"] in labels
    top = dict(hints)
    assert top[groups[gid_a]["ko_label"]] == pytest.approx(0.45 if len(da) > 1 else 0.2)
    # allowed로 목록에 없는 그룹은 뺀다 (정규화 없음)
    only = group_hints(kb, "pushpins", probs, {NORMAL, groups[gid_b]["ko_label"]})
    assert [g for g, _ in only] == [NORMAL, groups[gid_b]["ko_label"]] or [g for g, _ in only] == [groups[gid_b]["ko_label"], NORMAL]
    assert sum(p for _, p in only) == pytest.approx(0.55, abs=0.011)
    with pytest.raises(KeyError):
        group_hints(kb, "pushpins", {"없는_결함": 1.0})


def test_load_hint_probs_merges_heldout_and_oof(tmp_path):
    p = tmp_path / "b1_f0_probs.jsonl"
    p.write_text(json.dumps({"id": "a", "probs": {"good": 1.0}}) + "\n", encoding="utf-8")
    (tmp_path / "b1_f0_probs.oof.jsonl").write_text(json.dumps({"id": "b", "probs": {"good": 0.5}}) + "\n", encoding="utf-8")
    d = load_hint_probs(p)
    assert set(d) == {"a", "b"} and d["b"] == {"good": 0.5}
    assert np.isclose(sum(d["a"].values()), 1.0)
