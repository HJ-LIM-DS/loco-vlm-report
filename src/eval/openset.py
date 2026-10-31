"""RQ3 open-set 채점: 학습·프롬프트 목록에서 지운 그룹(미등록 결함)에 "미분류 이상"으로 답하는가.

입력: open-set 학습 run의 held-out 예측과 그 run의 SFT 폴더(openset=true 표시, 제외 그룹 목록).
- 미등록 결함 이미지: 미분류 응답률, 기존 그룹으로 오분류한 비율, "정상"으로 놓친 비율,
  자유 서술의 핵심어 일치율(예: 이물 혼입 → "이물", 칸막이 누락 → "칸막이")
- 알려진 결함 이미지: 미분류로 잘못 답한 비율(과잉 미분류), 그룹 정확도
- 정상 이미지: "정상" 응답 비율
목록 드롭아웃 유무 쌍(같은 제외 그룹)을 나란히 보여 RQ3 가설(드롭아웃 없이는 미등록 결함을 기존 그룹으로
잘못 분류한다)을 확인한다.

    python -m src.eval.openset --runs m3_open_struct_4b_f0:m3_open_struct_f0 m3_open_struct_nodrop_4b_f0:m3_open_struct_nodrop_f0
"""

import argparse
import json
from pathlib import Path

from src.output_spec import UNKNOWN_LABEL, is_normal, parse_report

ROOT = Path(__file__).resolve().parents[2]
KEYWORDS = {
    "contamination": ["이물"], "fruit_count": ["과일", "귤", "천도복숭아", "복숭아", "수량", "개수"],
    "fruit_icon": ["아이콘"], "missing_separator": ["칸막이"], "screw_length": ["나사", "길이", "짧", "긴"],
    "cable_color": ["색"],
}


def score(run: str, sft: str) -> dict:
    rows = {json.loads(x)["id"]: json.loads(x) for x in
            (ROOT / "data" / "sft" / sft / "heldout.jsonl").read_text(encoding="utf-8").splitlines()}
    preds = [json.loads(x) for x in (ROOT / "runs" / run / "final" / "heldout_predictions.jsonl")
             .read_text(encoding="utf-8").splitlines()]
    unseen, known, normal = [], [], []
    for p in preds:
        r = rows[p["id"]]
        text = p["pred"]
        rep = None if is_normal(text) else parse_report(text)
        if r["label"] == "good":
            normal.append(is_normal(text))
            continue
        if r.get("openset"):
            unk = rep is not None and rep.group == UNKNOWN_LABEL
            free = " ".join(rep.details) if unk else ""
            unseen.append({"id": p["id"], "group": r["group"], "unknown": unk, "normal": is_normal(text),
                           "known_group": rep is not None and rep.group != UNKNOWN_LABEL,
                           "keyword": unk and any(k in free for k in KEYWORDS.get(r["group"], []))})
        else:
            gt = parse_report(r["gt_report"])
            known.append({"id": p["id"], "unknown": rep is not None and rep.group == UNKNOWN_LABEL,
                          "correct": rep is not None and rep.group == gt.group})

    def rate(xs, k):
        return sum(x[k] for x in xs) / max(len(xs), 1)

    by_group = {}
    for g in sorted({u["group"] for u in unseen}):
        us = [u for u in unseen if u["group"] == g]
        by_group[g] = {"n": len(us), "unknown_rate": rate(us, "unknown"), "keyword_hit": rate(us, "keyword")}
    return {"run": run, "sft": sft, "n_unseen": len(unseen), "unseen_unknown_rate": rate(unseen, "unknown"),
            "unseen_to_known_group": rate(unseen, "known_group"), "unseen_said_normal": rate(unseen, "normal"),
            "unseen_keyword_hit": rate(unseen, "keyword"), "n_known": len(known),
            "known_false_unknown": rate(known, "unknown"), "known_group_acc": rate(known, "correct"),
            "normal_correct": sum(normal) / max(len(normal), 1), "by_group": by_group,
            "unseen_per_image": [{k: u[k] for k in ("id", "group", "unknown", "known_group")} for u in unseen],
            "known_per_image": [{k: x[k] for k in ("id", "correct")} for x in known]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="run이름:SFT폴더이름 쌍")
    ap.add_argument("--out", default=str(ROOT / "reports" / "tables" / "m3_openset"))
    a = ap.parse_args()
    res = [score(*x.split(":")) for x in a.runs if (ROOT / "runs" / x.split(":")[0] / "final").exists()]
    pct = lambda v: f"{100 * v:.1f}"  # noqa: E731
    lines = ["# RQ3 open-set (fold 0 모델 → fold 1 이미지)", "",
             "| run | 미등록 n | 미분류 응답 | 기존 그룹 오분류 | 정상 응답 | 핵심어 일치 | 알려진 결함 과잉 미분류 | 알려진 결함 그룹 정확도 | 정상 거부 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in res:
        lines.append(f"| {r['run']} | {r['n_unseen']} | {pct(r['unseen_unknown_rate'])} | {pct(r['unseen_to_known_group'])} | "
                     f"{pct(r['unseen_said_normal'])} | {pct(r['unseen_keyword_hit'])} | {pct(r['known_false_unknown'])} | "
                     f"{pct(r['known_group_acc'])} | {pct(r['normal_correct'])} |")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
