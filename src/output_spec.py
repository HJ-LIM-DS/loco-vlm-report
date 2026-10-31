"""2줄 이상 리포트 출력 명세: 생성, 파싱, GT 정답 구성.

    이상: [<유형>] <그룹명>[ - <세부>[ + <세부>]][ (<위치>)]
    조치: <조치>

- 유형: 논리 | 구조 | 미상(Guard 대체 시에만)
- 그룹명: KB ko_label 또는 "미분류 이상"
- 세부: KB 세부 결함 ko. 그룹에 세부 결함이 1종뿐이면 생략, 복수면 " + "로 연결.
  "미분류 이상"일 때는 20자 이내 자유 서술.
- 위치: 3x3 칸 이름 9종 | "전체" | 생략 (세부 결함의 loc_mode에 따름)
- 정상 판정 시 VLM 출력은 "정상" 한 단어.
"""

import re
from dataclasses import dataclass

from src.kb import defect_table

GRID_KO = [
    ["좌측 상단", "상단 중앙", "우측 상단"],
    ["좌측", "중앙", "우측"],
    ["좌측 하단", "하단 중앙", "우측 하단"],
]
CELLS = tuple(c for row in GRID_KO for c in row)
LOC_GLOBAL = "전체"
LOCATIONS = CELLS + (LOC_GLOBAL,)

TYPE_UNKNOWN = "미상"
UNKNOWN_LABEL = "미분류 이상"
UNKNOWN_ACTION = "해당 제품을 격리하고, 품질 담당자 확인을 요청"
NORMAL = "정상"

DETAIL_SEP = " + "
MAX_LINE = 60
MAX_FREE_DETAIL = 20
# 출력 문법에서 구분자로 쓰여 KB 문자열 안에 나오면 안 되는 토큰
RESERVED = (" - ", DETAIL_SEP, "(", ")", "[", "]", "\n")

LINE1_RE = re.compile(r"^이상: \[(논리|구조|미상)\] (.+?)(?: - (.+?))?(?: \((.+)\))?$")
LINE2_RE = re.compile(r"^조치: (.+)$")


@dataclass(frozen=True)
class Report:
    type: str
    group: str
    details: tuple = ()
    location: str | None = None
    action: str = ""


def format_report(r: Report) -> str:
    line1 = f"이상: [{r.type}] {r.group}"
    if r.details:
        line1 += " - " + DETAIL_SEP.join(r.details)
    if r.location:
        line1 += f" ({r.location})"
    return f"{line1}\n조치: {r.action}"


def parse_report(text: str) -> Report | None:
    """2줄 리포트를 파싱한다. 형식이 맞지 않거나 위치가 허용 목록 밖이면 None."""
    lines = text.strip().split("\n")
    if len(lines) != 2:
        return None
    m1, m2 = LINE1_RE.match(lines[0].strip()), LINE2_RE.match(lines[1].strip())
    if not (m1 and m2):
        return None
    typ, group, details, loc = m1.groups()
    if loc is not None and loc not in LOCATIONS:
        return None
    return Report(
        type=typ,
        group=group,
        details=tuple(details.split(DETAIL_SEP)) if details else (),
        location=loc,
        action=m2.group(1),
    )


def is_normal(text: str) -> bool:
    return text.strip() == NORMAL


def line_lengths(text: str) -> list[int]:
    return [len(ln) for ln in text.split("\n")]


def resolve_loc_mode(modes) -> str:
    """세부 결함이 여럿일 때 이미지의 위치 방식: point > global > instances."""
    modes = set(modes)
    for m in ("point", "global", "instances"):
        if m in modes:
            return m
    raise ValueError(f"알 수 없는 loc_mode: {modes}")


def gt_report(kb: dict, category: str, defect_names, location: str | None) -> Report:
    """이미지의 세부 결함명 목록과 위치 라벨로 정답 리포트를 만든다."""
    table = defect_table(kb, category)
    rows = sorted((table[n] for n in defect_names), key=lambda d: d["order"])
    groups = {d["group"] for d in rows}
    types = {d["type"] for d in rows}
    if len(groups) != 1 or len(types) != 1:
        raise ValueError(f"{category} {defect_names}: 그룹 {groups} / 유형 {types}가 하나가 아님")
    g = kb[category]["groups"][groups.pop()]
    if g.get("exclusive") and len(rows) > 1:
        raise ValueError(f"{category} {defect_names}: 배타적 그룹에 세부 결함이 2종 이상")
    details = tuple(d["ko"] for d in rows) if len(g["defects"]) > 1 else ()
    return Report(types.pop(), g["ko_label"], details, location, g["action"])


def unknown_report(location: str | None = None, typ: str = TYPE_UNKNOWN,
                   free_text: str | None = None) -> Report:
    """미분류 템플릿. Guard 대체는 typ="미상", 목록 드롭아웃 정답은 typ=논리/구조 + 자유 서술."""
    if free_text is not None and len(free_text) > MAX_FREE_DETAIL:
        raise ValueError(f"자유 서술은 {MAX_FREE_DETAIL}자 이내: {free_text!r}")
    details = (free_text,) if free_text else ()
    return Report(typ, UNKNOWN_LABEL, details, location, UNKNOWN_ACTION)
