"""전체 파이프라인: 탐지기가 1차 판정하고, 이상 의심일 때만 VLM이 2줄 리포트를 내며, Guard가 교정한다.

    score <  tau_low             → 출력 없음
    tau_low <= score < tau_high  → VLM 판정 ("정상"이면 출력 없음 = 오탐 거부)
    score >= tau_high            → VLM 설명 (거부 불가: "정상"이면 미분류 템플릿으로 대체)

배치 모드는 build_sft가 만든 held-out 레코드(이미지 3장 + 프롬프트)를 입력으로 쓴다.
임계값은 학습 fold의 학습 측 test 이미지로만 정한다 (src.detector.calibrate).

사용법:
    python -m src.pipeline --adapter runs/m1_vlm2b_f0/final --sft data/sft/m1_effad_f0 --detector efficientad \
        --out outputs/pipeline/m1_f0_heldout.jsonl
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch

from src.detector.calibrate import fold_thresholds
from src.detector.cache import load_scores
from src.kb import load_kb
from src.output_spec import NORMAL, format_report, is_normal, parse_report, unknown_report
from src.vlm.guard import guard

ROOT = Path(__file__).resolve().parents[1]


def decide(score: float, th: dict) -> str:
    if score < th["tau_low"]:
        return "pass"
    return "judge" if score < th["tau_high"] else "describe"


def finalize(stage: str, vlm_text: str | None, kb: dict, category: str, cell: str | None) -> tuple[str, list]:
    """최종 출력 (빈 문자열 = 무출력)과 Guard 단계."""
    if stage == "pass":
        return "", []
    if is_normal(vlm_text):
        if stage == "judge":
            return "", []
        return format_report(unknown_report(cell)), ["forced_describe"]
    out, steps = guard(vlm_text, kb, category, cell)
    return out, steps


def run(args):
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    from src.vlm.infer import generate

    kb = load_kb()
    meta = json.loads((Path(args.sft) / "meta.json").read_text(encoding="utf-8"))
    rows = [json.loads(x) for x in (Path(args.sft) / "heldout.jsonl").read_text(encoding="utf-8").splitlines()]
    index = pd.read_parquet(ROOT / "data" / "index.parquet")
    splits = pd.read_csv(ROOT / "data" / "splits.csv")
    th = {c: fold_thresholds(args.detector, c, index, splits, meta["fold"]) for c in meta["categories"]}
    scores = pd.concat([load_scores(args.detector, c) for c in meta["categories"]]).set_index("image_id")["score"]

    stages = {r["id"]: decide(float(scores.loc[r["id"]]), th[r["category"]]) for r in rows}
    to_vlm = [r for r in rows if stages[r["id"]] != "pass"]
    processor = AutoProcessor.from_pretrained(args.adapter)
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model, dtype=torch.bfloat16,
                                                            attn_implementation="sdpa", device_map={"": 0})
    model = PeftModel.from_pretrained(model, args.adapter).eval()
    t0 = time.time()
    texts = generate(model, processor, to_vlm, batch_size=args.batch_size)
    vlm_s = (time.time() - t0) / max(len(to_vlm), 1)
    vlm = dict(zip((r["id"] for r in to_vlm), texts))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    recs = []
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            stage = stages[r["id"]]
            cell = (r.get("detector") or {}).get("location")
            final, steps = finalize(stage, vlm.get(r["id"]), kb, r["category"], cell)
            rec = {"id": r["id"], "category": r["category"], "label": r["label"],
                   "score": float(scores.loc[r["id"]]), "stage": stage, "vlm_raw": vlm.get(r["id"]),
                   "final": final, "guard_steps": steps, "gt_report": r["gt_report"]}
            recs.append(rec)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    summary = summarize(recs, th, vlm_s)
    out.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


def summarize(recs: list, th: dict, vlm_s: float) -> dict:
    df = pd.DataFrame(recs)
    anom = df["label"] != "good"
    flagged = df["final"] != ""
    det_flag = df["stage"] != "pass"
    gt = df.loc[anom, "gt_report"].map(parse_report)
    pr = df.loc[anom, "final"].map(lambda t: parse_report(t) if t else None)
    grp = [p is not None and g is not None and p.group == g.group for p, g in zip(pr, gt)]
    return {
        "thresholds": th,
        "n": len(df), "n_anomaly": int(anom.sum()), "n_normal": int((~anom).sum()),
        "detector_only": {"recall": float(det_flag[anom].mean()), "fpr": float(det_flag[~anom].mean())},
        "cascade": {"recall": float(flagged[anom].mean()), "fpr": float(flagged[~anom].mean())},
        "vlm_call_rate": {"normal": float(det_flag[~anom].mean()), "anomaly": float(det_flag[anom].mean())},
        "group_acc_on_anomalies_end_to_end": float(sum(grp) / max(len(grp), 1)),
        "format_ok": float(df.loc[flagged, "final"].map(lambda t: parse_report(t) is not None).mean())
        if flagged.any() else None,
        "vlm_seconds_per_image": round(vlm_s, 3),
        "normal_output": NORMAL,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    ap.add_argument("--sft", required=True)
    ap.add_argument("--detector", default="efficientad")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=4)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
