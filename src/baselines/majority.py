"""B0 최빈값 기준선: 학습 측에서 카테고리별 최빈 그룹 -> 그 그룹의 최빈 세부 결함 -> 최빈 위치.

이미지를 보지 않고 항상 같은 2줄을 낸다(정상 이미지에도). 진단기가 넘어야 할 최저선이다.

사용법:
    python -m src.baselines.majority --fold 0 --out outputs/baselines/b0_f0_heldout.jsonl
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from src.kb import load_kb
from src.output_spec import format_report, gt_report, resolve_loc_mode
from src.kb import defect_table

ROOT = Path(__file__).resolve().parents[2]


def majority_reports(train: pd.DataFrame, kb: dict) -> dict:
    out = {}
    for cat, g in train[train["label"] != "good"].groupby("category"):
        grp = Counter(x[0] for x in g["groups"]).most_common(1)[0][0]
        gg = g[g["groups"].map(lambda x: x[0] == grp)]
        defects = Counter(tuple(x) for x in gg["defects"]).most_common(1)[0][0]
        same = gg[gg["defects"].map(tuple) == defects]
        mode = resolve_loc_mode(defect_table(kb, cat)[d]["loc_mode"] for d in defects)
        loc = None if mode == "instances" else Counter(same["location"].dropna()).most_common(1)[0][0]
        out[cat] = format_report(gt_report(kb, cat, list(defects), loc))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True, help="학습 fold (예측은 다른 fold)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    kb = load_kb()
    df = pd.read_parquet(ROOT / "data" / "index.parquet").merge(
        pd.read_csv(ROOT / "data" / "splits.csv")[["image_id", "fold"]], on="image_id")
    train = df[df["fold"] == a.fold]
    held = df[(df["split"] == "test") & (df["fold"] != a.fold)]
    reports = majority_reports(train, kb)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in held.itertuples():
            f.write(json.dumps({"id": r.image_id, "target": r.gt_report, "pred": reports[r.category]},
                               ensure_ascii=False) + "\n")
    print(json.dumps(reports, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
