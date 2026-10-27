"""Guard: VLM 원출력을 KB 기준으로 단계별 교정한다 (계획 v5 출력 명세).

1. 세부가 그룹 KB에 없거나, 세부의 type과 [유형]이 모순되거나, 배타적 그룹에 세부가 2종 이상 -> 세부 삭제
2. 그룹명이 해당 카테고리 KB에 없음 -> [미상] 미분류 이상 (위치는 유지)
3. 조치가 그룹 KB 조치와 다름 -> KB 조치로 교체
4. 파싱 실패 -> 이상: [미상] 미분류 이상 (<탐지기 최대 응답 칸>) / 조치: 미분류 조치
지표는 Guard 적용 전(원출력)과 후를 모두 보고한다.
"""

from dataclasses import replace

from src.output_spec import (
    MAX_FREE_DETAIL, TYPE_UNKNOWN, UNKNOWN_ACTION, UNKNOWN_LABEL, Report, format_report, is_normal, parse_report,
    unknown_report,
)


def guard(text: str, kb: dict, category: str, fallback_location: str | None = None) -> tuple[str, list]:
    """반환: (교정된 출력, 적용된 단계 목록). 정상 출력은 그대로 둔다."""
    if is_normal(text):
        return text.strip(), []
    r = parse_report(text)
    if r is None:
        return format_report(unknown_report(fallback_location)), ["4_parse_fail"]
    steps = []
    groups = {g["ko_label"]: g for g in kb[category]["groups"].values()}
    if r.group == UNKNOWN_LABEL:
        # 미분류 응답: 자유 서술 길이와 조치만 맞춘다
        if r.details and (len(r.details) > 1 or len(r.details[0]) > MAX_FREE_DETAIL):
            r = replace(r, details=(r.details[0][:MAX_FREE_DETAIL],))
            steps.append("1_free_text_trim")
        if r.action != UNKNOWN_ACTION:
            r = replace(r, action=UNKNOWN_ACTION)
            steps.append("3_action")
        return format_report(r), steps
    g = groups.get(r.group)
    if g is None:
        return format_report(Report(TYPE_UNKNOWN, UNKNOWN_LABEL, (), r.location, UNKNOWN_ACTION)), ["2_group"]
    if r.details:
        by_ko = {d["ko"]: d for d in g["defects"].values()}
        bad = any(d not in by_ko or by_ko[d]["type"] != r.type for d in r.details)
        if bad or (g.get("exclusive") and len(r.details) > 1) or len(g["defects"]) == 1:
            r = replace(r, details=())
            steps.append("1_detail")
    if r.action != g["action"]:
        r = replace(r, action=g["action"])
        steps.append("3_action")
    return format_report(r), steps
