"""결함 그룹·조치 지식베이스(taxonomy/action_kb.yaml) 로드와 조회."""

from pathlib import Path

import yaml

KB_PATH = Path(__file__).resolve().parents[1] / "taxonomy" / "action_kb.yaml"

TYPES = ("논리", "구조")
LOC_MODES = ("point", "global", "instances")
DISPOSITIONS = ("폐기", "재작업", "격리")
SEVERITIES = ("라인정지", "일반")
ROOT_CAUSES = ("설비", "자재", "방법")
# 폴더 이름 → KB 유형
FOLDER_TYPE = {"logical": "논리", "structural": "구조"}


def load_kb(path: Path = KB_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def defect_table(kb: dict, category: str) -> dict:
    """세부 결함명 → {group, ko, type, loc_mode, order}. order는 KB 안의 선언 순서."""
    table = {}
    for gid, g in kb[category]["groups"].items():
        for i, (name, d) in enumerate(g["defects"].items()):
            table[name] = {"group": gid, "order": i, **d}
    return table
