"""출력 명세 단위 테스트: KB 스키마, 모든 KB 조합의 줄 길이와 파서 왕복, 인덱스 정답."""

from itertools import combinations
from pathlib import Path

import pandas as pd
import pytest

from src.kb import DISPOSITIONS, LOC_MODES, ROOT_CAUSES, SEVERITIES, TYPES, defect_table, load_kb
from src.output_spec import (
    CELLS, LOC_GLOBAL, MAX_FREE_DETAIL, MAX_LINE, NORMAL, RESERVED, TYPE_UNKNOWN, UNKNOWN_ACTION,
    UNKNOWN_LABEL, Report, format_report, gt_report, is_normal, line_lengths, parse_report,
    resolve_loc_mode, unknown_report,
)

KB = load_kb()
INDEX = Path(__file__).resolve().parents[1] / "data" / "index.parquet"


def locations_for(mode: str):
    return {"point": CELLS, "global": (LOC_GLOBAL,), "instances": (None,)}[mode]


def all_kb_cases():
    """(category, 세부 결함명 1~2개) 전체. 2개 조합은 같은 그룹 안에서만 만든다."""
    for cat, c in KB.items():
        for g in c["groups"].values():
            names = list(g["defects"])
            for k in (1, 2):
                for combo in combinations(names, k):
                    yield cat, combo


# --- KB 스키마 ---

@pytest.mark.parametrize("cat", list(KB))
def test_kb_schema(cat):
    c = KB[cat]
    assert c["spec"] and c["ko_name"]
    assert c["unknown"]["ko_label"] == UNKNOWN_LABEL
    assert c["unknown"]["action"] == UNKNOWN_ACTION
    labels = [g["ko_label"] for g in c["groups"].values()]
    assert len(labels) == len(set(labels)), "카테고리 안에서 그룹 이름이 중복됨"
    assert UNKNOWN_LABEL not in labels
    for gid, g in c["groups"].items():
        assert g["disposition"] in DISPOSITIONS, gid
        assert g["severity"] in SEVERITIES, gid
        assert g["root_cause"] in ROOT_CAUSES, gid
        kos = [d["ko"] for d in g["defects"].values()]
        assert len(kos) == len(set(kos)), f"{gid}: 세부 결함 이름 중복"
        for name, d in g["defects"].items():
            assert d["type"] in TYPES, name
            assert d["loc_mode"] in LOC_MODES, name
        # 1줄에 들어가는 문자열만 구분자 토큰을 금지한다 (2줄 조치는 줄바꿈만 금지)
        for text in [g["ko_label"], *kos]:
            for tok in RESERVED:
                assert tok not in text, f"{cat}/{gid}: 예약 토큰 {tok!r} 포함: {text!r}"
        assert "\n" not in g["action"]
        assert isinstance(g.get("exclusive", False), bool)


def test_kb_has_89_defects():
    assert sum(len(defect_table(KB, cat)) for cat in KB) == 89


# --- 모든 KB 조합: 60자 이하 + 파서 왕복 ---

@pytest.mark.parametrize("cat,names", list(all_kb_cases()))
def test_all_combinations_roundtrip(cat, names):
    table = defect_table(KB, cat)
    types = {table[n]["type"] for n in names}
    exclusive = KB[cat]["groups"][table[names[0]]["group"]].get("exclusive", False)
    if len(types) > 1 or (exclusive and len(names) > 1):
        # 논리·구조가 섞이거나 배타적 그룹의 2종 조합은 한 이미지에 나올 수 없음 → 생성 거부 확인
        with pytest.raises(ValueError):
            gt_report(KB, cat, names, None)
        return
    mode = resolve_loc_mode(table[n]["loc_mode"] for n in names)
    for loc in locations_for(mode):
        r = gt_report(KB, cat, names, loc)
        text = format_report(r)
        assert max(line_lengths(text)) <= MAX_LINE, text
        assert parse_report(text) == r, text


@pytest.mark.parametrize("loc", CELLS + (LOC_GLOBAL, None))
def test_guard_fallback(loc):
    r = unknown_report(location=loc)
    text = format_report(r)
    assert text.startswith(f"이상: [{TYPE_UNKNOWN}] {UNKNOWN_LABEL}")
    assert text.endswith(f"조치: {UNKNOWN_ACTION}")
    assert max(line_lengths(text)) <= MAX_LINE
    assert parse_report(text) == r


@pytest.mark.parametrize("typ", list(TYPES))
def test_dropout_target_with_free_text(typ):
    free = "가" * MAX_FREE_DETAIL
    r = unknown_report(location="좌측 상단", typ=typ, free_text=free)
    text = format_report(r)
    assert max(line_lengths(text)) <= MAX_LINE
    assert parse_report(text) == r
    with pytest.raises(ValueError):
        unknown_report(typ=typ, free_text=free + "가")


# --- 파서 실패 사례 ---

@pytest.mark.parametrize("text", [
    "이상: [구조] 이물 혼입 (중앙)",                                  # 1줄
    "이상: [구조] 이물 혼입\n조치: 격리\n추가 줄",                       # 3줄
    "이상: [기타] 이물 혼입\n조치: 격리",                               # 허용되지 않은 유형
    "이상: [구조] 이물 혼입 (가운데)\n조치: 격리",                       # 허용되지 않은 위치
    "이상: 이물 혼입\n조치: 격리",                                     # 유형 누락
    "이상: [구조] 이물 혼입\n해결: 격리",                               # 2줄 머리말 오류
    NORMAL,
])
def test_parse_rejects(text):
    assert parse_report(text) is None


def test_normal():
    assert is_normal("정상") and is_normal(" 정상\n")
    assert not is_normal("정상입니다")


def test_resolve_loc_mode():
    assert resolve_loc_mode(["instances", "point"]) == "point"
    assert resolve_loc_mode(["instances", "global"]) == "global"
    assert resolve_loc_mode(["instances"]) == "instances"


def test_single_detail_group_omits_detail():
    r = gt_report(KB, "pushpins", ["missing_separator"], "중앙")
    assert r.details == ()
    assert format_report(r).split("\n")[0] == "이상: [논리] 칸막이 누락 (중앙)"


def test_details_follow_kb_order():
    r = gt_report(KB, "pushpins", ["missing_pushpin", "1_additional_pushpin"], "중앙")
    assert r.details == ("푸시핀 1개 초과", "푸시핀 누락")


# --- 인덱스 정답 (data/index.parquet가 있을 때만) ---

@pytest.mark.skipif(not INDEX.exists(), reason="data/index.parquet 없음 (src.data.index_loco 실행 필요)")
def test_index_reports():
    df = pd.read_parquet(INDEX)
    anom = df[df["label"] != "good"]
    assert (df.loc[df["label"] == "good", "gt_report"] == NORMAL).all()
    for row in anom.itertuples():
        r = parse_report(row.gt_report)
        assert r is not None, row.image_id
        assert max(line_lengths(row.gt_report)) <= MAX_LINE, row.image_id
        assert r.location == (None if pd.isna(row.location) else row.location), row.image_id
        assert r.type == row.defect_type, row.image_id
        if row.loc_mode == "point":
            assert r.location in CELLS, row.image_id


SFT_DIR = Path(__file__).resolve().parents[1] / "data" / "sft"


@pytest.mark.skipif(not SFT_DIR.exists(), reason="data/sft 없음")
def test_sft_targets_parse():
    """만들어진 SFT 데이터의 모든 정답이 출력 명세를 따른다 (위치 NaN 같은 잘못된 목표 방지)."""
    import json

    from src.report.publish_predictions import HINT_FILE

    checked = 0
    for f in SFT_DIR.glob("*/*.jsonl"):
        if f.name == HINT_FILE:  # 공개 사본에서 복원한 슬림 힌트 기록(id·라벨·정답·힌트만)에는 target이 없다
            continue
        for ln in f.read_text(encoding="utf-8").splitlines():
            t = json.loads(ln)["target"]
            assert t == NORMAL or parse_report(t) is not None, f"{f}: {t!r}"
            checked += 1
    if not checked:
        pytest.skip("SFT 기록 없음 (공개 사본만 복원한 환경)")


CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "vlm_cache"


@pytest.mark.skipif(not CACHE_DIR.exists(), reason="data/vlm_cache 없음")
def test_vlm_cache_has_no_empty_files():
    """쓰는 도중 끊겨 남은 빈 캐시 이미지가 없어야 한다 (원자적 쓰기 이전에 생긴 0바이트 PNG 사례)."""
    empty = [str(p) for p in CACHE_DIR.rglob("*.png") if p.stat().st_size == 0]
    leftovers = [str(p) for p in CACHE_DIR.rglob(".*.tmp*")]
    assert not empty and not leftovers, (empty[:5], leftovers[:5])
