"""MVTec LOCO AD (+ MMAD 부가자료) 이미지 인덱스 생성.

이미지마다 카테고리/분할/라벨, GT 마스크 픽셀값에서 복원한 세부 결함명,
action_kb 기준 결함 그룹·유형, loc_mode 규칙에 따른 위치 라벨, 정답 2줄 리포트,
MMAD QA 정답(annotation: false 문항 제외)을 한 행으로 정리한다.

위치 라벨 (세부 결함 loc_mode, 여럿이면 point > global > instances)
- point: point 세부 결함들의 마스크 합집합을 긴 변 2% 반경으로 closing한 뒤 면적 최대 3x3 칸
- global: "전체"
- instances: 없음 (같은 종류 부품 전체가 칠해져 위치를 특정할 수 없음)

사용법:
    python -m src.data.index_loco --data-root MVTec-LOCO --out data/index.parquet
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.masks import CATEGORIES, cell_areas, close_mask, load_defect_masks
from src.kb import FOLDER_TYPE, defect_table, load_kb
from src.output_spec import GRID_KO, LOC_GLOBAL, NORMAL, format_report, gt_report, resolve_loc_mode

SUBSETS = [
    ("train", "good"),
    ("validation", "good"),
    ("test", "good"),
    ("test", "logical_anomalies"),
    ("test", "structural_anomalies"),
]
LABEL_OF = {"good": "good", "logical_anomalies": "logical", "structural_anomalies": "structural"}
# 이미지 수 검증 기준 (MVTec LOCO AD 공식 구성)
EXPECTED_TOTAL = 3651
EXPECTED_TEST = {"good": 575, "anomalous": 993}


def read_mask_info(mask_dir: Path, pixel_to_name: dict, table: dict) -> dict:
    """GT 마스크 폴더에서 결함명, 위치 방식·라벨, 정규화 bbox를 계산한다."""
    masks, long_side = load_defect_masks(mask_dir)
    unknown = set(masks) - set(pixel_to_name)
    if unknown:
        raise ValueError(f"{mask_dir}: defects_config에 없는 픽셀값 {sorted(unknown)}")
    if not masks:
        raise ValueError(f"{mask_dir}: 결함 픽셀이 없음")
    names = sorted(pixel_to_name[v] for v in masks)
    union = np.logical_or.reduce(list(masks.values()))
    ys, xs = np.nonzero(union)
    hh, ww = union.shape
    bbox = [xs.min() / ww, ys.min() / hh, (xs.max() + 1) / ww, (ys.max() + 1) / hh]

    mode = resolve_loc_mode(table[n]["loc_mode"] for n in names)
    location, top_share, cell_frac = None, None, None
    if mode == "point":
        pm = np.logical_or.reduce(
            [m for v, m in masks.items() if table[pixel_to_name[v]]["loc_mode"] == "point"]
        )
        cells = cell_areas(close_mask(pm, long_side))
        r, c = np.unravel_index(cells.argmax(), cells.shape)
        # 라벨 칸에 closing 전 원 마스크가 실제로 들어 있어야 한다
        assert cell_areas(pm)[r, c] > 0, f"{mask_dir}: 위치 칸 {GRID_KO[r][c]}에 원 마스크 없음"
        location, top_share = GRID_KO[r][c], float(cells.max() / cells.sum())
        # 위치 적중 채점용: closing 전 원 마스크가 3x3 칸마다 차지하는 비율 (행 우선 9개)
        raw = cell_areas(pm)
        cell_frac = [round(float(v), 4) for v in (raw / raw.sum()).ravel()]
    elif mode == "global":
        location = LOC_GLOBAL
    return {
        "defects": names,
        "loc_mode": mode,
        "location": location,
        "loc_top_share": None if top_share is None else round(top_share, 4),
        "point_cell_frac": cell_frac,
        "bbox": [round(float(b), 4) for b in bbox],
    }


def qa_filter(conv: list) -> list:
    """MMAD 문항 중 annotation: false(검수에서 정답 오류로 표시)를 제외한다."""
    return [q for q in conv if q.get("annotation") is not False]


def qa_answers(conv: list) -> dict:
    """MMAD 대화에서 type이 있는 문항만 골라 {type: 정답 보기 텍스트}로 정리."""
    out = {}
    for q in conv:
        t = q.get("type")
        if t:
            out[t] = q["Options"][q["Answer"]]
    return out


def build_index(data_root: Path, kb: dict) -> pd.DataFrame:
    rows = []
    for cat in CATEGORIES:
        cdir = data_root / cat
        cfg = json.load(open(cdir / "defects_config.json", encoding="utf-8"))
        pixel_to_name = {d["pixel_value"]: d["defect_name"] for d in cfg}
        table = defect_table(kb, cat)
        missing = set(pixel_to_name.values()) - set(table)
        if missing:
            raise ValueError(f"{cat}: action_kb에 없는 결함명 {sorted(missing)}")
        qa = json.load(open(cdir / "QA.json", encoding="utf-8"))

        for split, sub in SUBSETS:
            for img_path in sorted((cdir / split / sub).glob("*.png")):
                stem = img_path.stem
                rel = f"{split}/{sub}/{img_path.name}"
                label = LABEL_OF[sub]
                row = {
                    "image_id": f"{cat}/{split}/{sub}/{stem}",
                    "category": cat,
                    "split": split,
                    "label": label,
                    "image_path": f"{cat}/{rel}",
                    "defects": [],
                    "defects_ko": [],
                    "groups": [],
                    "defect_type": None,
                    "loc_mode": None,
                    "location": None,
                    "loc_top_share": None,
                    "point_cell_frac": None,
                    "bbox": None,
                    "mask_dir": None,
                    "gt_report": NORMAL,
                }
                if label != "good":
                    mdir = cdir / "ground_truth" / sub / stem
                    info = read_mask_info(mdir, pixel_to_name, table)
                    names = info["defects"]
                    types = {table[n]["type"] for n in names}
                    if types != {FOLDER_TYPE[label]}:
                        raise ValueError(f"{mdir}: KB 유형 {types} != 폴더 {label}")
                    row.update(
                        info,
                        defects_ko=[table[n]["ko"] for n in names],
                        groups=sorted({table[n]["group"] for n in names}),
                        defect_type=FOLDER_TYPE[label],
                        mask_dir=f"{cat}/ground_truth/{sub}/{stem}",
                        gt_report=format_report(gt_report(kb, cat, names, info["location"])),
                    )
                # 캡션: 같은 이름의 txt만 사용 ("*_wrong.txt" 등 변형 파일은 제외)
                cap = img_path.with_suffix(".txt")
                row["caption_path"] = f"{cat}/{split}/{sub}/{cap.name}" if cap.exists() else None
                # MMAD QA: type이 붙은 test 문항만 정답 텍스트로 저장, 검수 통과 대화도 보관
                entry = qa.get(rel)
                conv = qa_filter(entry["conversation"]) if entry else []
                ans = qa_answers(conv)
                row["qa_anomaly"] = ans.get("Anomaly Detection")
                row["qa_defect_type"] = ans.get("Defect Classification")
                row["qa_location"] = ans.get("Defect Localization")
                row["qa_description"] = ans.get("Defect Description")
                row["qa_effect"] = ans.get("Defect Analysis")
                row["qa_json"] = json.dumps(conv, ensure_ascii=False) if ans else None
                rows.append(row)
    return pd.DataFrame(rows)


def check_index(df: pd.DataFrame):
    """계획 v5 Verification 1: 이미지 수, 이상 이미지의 그룹·유형·loc_mode·리포트 존재."""
    assert len(df) == EXPECTED_TOTAL, f"전체 {len(df)} != {EXPECTED_TOTAL}"
    test = df[df["split"] == "test"]
    n_good = int((test["label"] == "good").sum())
    assert n_good == EXPECTED_TEST["good"], n_good
    assert len(test) - n_good == EXPECTED_TEST["anomalous"], len(test) - n_good
    anom = df[df["label"] != "good"]
    assert (anom["groups"].map(len) == 1).all(), "그룹이 정확히 하나가 아닌 이상 이미지가 있음"
    assert anom["defect_type"].notna().all() and anom["loc_mode"].notna().all()
    assert (anom["location"].isna() == (anom["loc_mode"] == "instances")).all()
    assert (anom["gt_report"].str.count("\n") == 1).all()


def main():
    root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=root / "MVTec-LOCO")
    ap.add_argument("--kb", type=Path, default=root / "taxonomy" / "action_kb.yaml")
    ap.add_argument("--out", type=Path, default=root / "data" / "index.parquet")
    args = ap.parse_args()

    df = build_index(args.data_root, load_kb(args.kb))
    check_index(df)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"저장: {args.out} ({len(df)}행)")
    print(df.groupby(["category", "split", "label"]).size().unstack(fill_value=0))
    anom = df[df["label"] != "good"]
    print("\n위치 방식별 이상 이미지 수:")
    print(anom.groupby(["category", "loc_mode"]).size().unstack(fill_value=0))
    point = anom[anom["loc_mode"] == "point"]
    print(f"\npoint 위치 라벨 {len(point)}장 중 최대 칸 점유율 0.5 미만(모호): "
          f"{int((point['loc_top_share'] < 0.5).sum())}장")
    print(point["location"].value_counts().to_string())


if __name__ == "__main__":
    main()
