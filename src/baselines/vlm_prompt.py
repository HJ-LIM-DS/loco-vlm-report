"""B3 캐스케이드 + 프롬프트만 쓰는 VLM (학습 없음): zero-shot / few-shot.

입력은 주력 모델과 같은 SFT 레코드(이미지 3장 + 프롬프트)이고, few-shot은 같은 카테고리의 KB로 만든
출력 예시 3개(정상 1, 이상 2)를 프롬프트 끝에 붙인다(이미지 예시는 넣지 않음).
예측은 두 가지로 저장한다.
- strict : 원출력 그대로 (형식까지 맞아야 채점됨)
- lenient: 원출력에서 KB 그룹 이름을 찾아 그 그룹의 2줄로 바꾼 것 (형식 학습 효과를 분리)

사전학습 노출 점검: 원출력에 영문 LOCO 결함명(예: missing_separator)이 그대로 나온 비율도 기록한다.

사용법:
    python -m src.baselines.vlm_prompt --sft data/sft/m2_salad_f0 --split heldout --mode zero \
        --out outputs/baselines/b3_zero_f0
"""

import argparse
import json
import re
from pathlib import Path

import torch

from src.kb import defect_table, load_kb
from src.output_spec import NORMAL, Report, format_report, gt_report, is_normal, parse_report
from src.vlm.infer import generate

ROOT = Path(__file__).resolve().parents[2]


def fewshot_block(kb: dict, category: str) -> str:
    """같은 카테고리 KB에서 고른 출력 예시 (정상 1, 서로 다른 그룹의 이상 2)."""
    groups = list(kb[category]["groups"].items())
    ex = [NORMAL]
    for gid, g in (groups[0], groups[-1]):
        name = next(iter(g["defects"]))
        mode = g["defects"][name]["loc_mode"]
        loc = {"point": "중앙", "global": "전체", "instances": None}[mode]
        ex.append(format_report(gt_report(kb, category, [name], loc)))
    return "\n[출력 예시]\n" + "\n---\n".join(ex)


def lenient(text: str, kb: dict, category: str) -> str:
    """형식이 틀려도 출력에 KB 그룹 이름이 있으면 그 그룹의 2줄로 바꾼다 (없으면 원출력)."""
    if is_normal(text) or parse_report(text) is not None:
        return text
    for g in sorted(kb[category]["groups"].values(), key=lambda x: -len(x["ko_label"])):
        if g["ko_label"] in text:
            typ = next(iter(g["defects"].values()))["type"]
            return format_report(Report(typ, g["ko_label"], (), None, g["action"]))
    if re.search(r"(^|\s)정상", text) and "이상" not in text:
        return NORMAL
    return text


def english_leak(text: str, kb: dict, category: str) -> bool:
    names = defect_table(kb, category).keys()
    t = text.lower()
    return any(n.lower() in t for n in names if len(n) > 5)


def run(args, ctx=None):
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    kb = load_kb()
    rows = [json.loads(x) for x in (Path(args.sft) / f"{args.split}.jsonl").read_text(encoding="utf-8").splitlines()]
    if args.limit:
        rows = rows[:args.limit]
    if args.mode == "few":
        for r in rows:
            r["prompt"] = r["prompt"] + fewshot_block(kb, r["category"])
    processor = AutoProcessor.from_pretrained(args.model)
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model, dtype=torch.bfloat16,
                                                            attn_implementation="sdpa", device_map={"": 0}).eval()
    beat = (lambda i, n: ctx.heartbeat(step=i, phase=f"generate {i}/{n}")) if ctx else None
    preds = generate(model, processor, rows, batch_size=args.batch_size, on_batch=beat)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    leak = 0
    with open(out / "strict.jsonl", "w", encoding="utf-8") as fs, open(out / "lenient.jsonl", "w", encoding="utf-8") as fl:
        for r, p in zip(rows, preds):
            leak += english_leak(p, kb, r["category"])
            fs.write(json.dumps({"id": r["id"], "target": r["target"], "pred": p}, ensure_ascii=False) + "\n")
            fl.write(json.dumps({"id": r["id"], "target": r["target"], "pred": lenient(p, kb, r["category"])},
                                ensure_ascii=False) + "\n")
    meta = {"model": args.model, "mode": args.mode, "n": len(rows), "english_defect_name_rate": leak / max(len(rows), 1)}
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False))


def main():
    from ops.train_hooks import RunContext, run_main

    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", required=True)
    ap.add_argument("--split", default="heldout")
    ap.add_argument("--mode", choices=["zero", "few"], default="zero")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    def task(a):
        ctx = RunContext.from_env(f"b3_{a.mode}")
        ctx.heartbeat(step=0, force=True, phase="generate")
        run(a, ctx)
        ctx.mark_done()

    run_main(task, args)


if __name__ == "__main__":
    main()
