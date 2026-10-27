"""VLM 추론(greedy)과 간이 채점. 정식 지표·통계는 Phase 3의 src/eval에서 계산한다.

사용법:
    python -m src.vlm.infer --adapter runs/<run>/final --data data/sft/<이름>_f0/heldout.jsonl --out <예측.jsonl>
"""

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image

from src.kb import load_kb
from src.output_spec import is_normal, parse_report
from src.vlm.guard import guard
from src.vlm.prompts import messages

ROOT = Path(__file__).resolve().parents[2]


@torch.no_grad()
def generate(model, processor, rows: list, batch_size: int = 4, max_new_tokens: int = 80, on_batch=None) -> list:
    """greedy 생성. on_batch(완료 수, 전체 수)는 감독자 하트비트용."""
    tok = processor.tokenizer
    outs = []
    for i in range(0, len(rows), batch_size):
        if on_batch is not None:
            on_batch(i, len(rows))
        batch = rows[i:i + batch_size]
        texts = [processor.apply_chat_template(messages(r["prompt"], len(r["images"])), add_generation_prompt=True,
                                               tokenize=False) for r in batch]
        imgs = [[Image.open(ROOT / p).convert("RGB") for p in r["images"]] for r in batch]
        enc = processor(text=texts, images=imgs, return_tensors="pt", padding=True, padding_side="left").to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        for g in gen[:, enc.input_ids.shape[1]:]:
            outs.append(tok.decode(g, skip_special_tokens=True).strip())
    return outs


def score(rows: list, preds: list, kb: dict) -> dict:
    """원출력과 Guard 후 출력 각각의 형식 준수율, 정상/이상 판정, 그룹·유형·세부·위치 일치율."""
    def tally(texts):
        n_fmt = n_dec = n_grp = n_typ = n_det = n_loc = n_anom = 0
        for r, t in zip(rows, texts):
            gt_normal = is_normal(r["target"])
            pr_normal = is_normal(t)
            parsed = parse_report(t)
            n_fmt += pr_normal or parsed is not None
            n_dec += gt_normal == pr_normal
            if not gt_normal:
                n_anom += 1
                gt = parse_report(r["target"])
                if parsed is not None and gt is not None:
                    n_grp += parsed.group == gt.group
                    n_typ += parsed.type == gt.type
                    n_det += parsed.details == gt.details
                    n_loc += parsed.location == gt.location
        n = max(len(texts), 1)
        a = max(n_anom, 1)
        return {"format_ok": n_fmt / n, "normal_decision_acc": n_dec / n, "group_acc": n_grp / a,
                "type_acc": n_typ / a, "detail_exact": n_det / a, "location_exact": n_loc / a}

    guarded, steps = [], {}
    for r, t in zip(rows, preds):
        loc = (r.get("detector") or {}).get("location")
        g, s = guard(t, kb, r["category"], loc)
        guarded.append(g)
        for k in s:
            steps[k] = steps.get(k, 0) + 1
    return {"n": len(rows), "raw": tally(preds), "guarded": tally(guarded), "guard_steps": steps}


def evaluate_records(model, processor, rows: list, out_path: Path, batch_size: int = 4, on_batch=None) -> dict:
    t0 = time.time()
    preds = generate(model, processor, rows, batch_size=batch_size, on_batch=on_batch)
    sec = time.time() - t0
    kb = load_kb()
    with open(out_path, "w", encoding="utf-8") as f:
        for r, p in zip(rows, preds):
            f.write(json.dumps({"id": r["id"], "target": r["target"], "pred": p}, ensure_ascii=False) + "\n")
    m = score(rows, preds, kb)
    m["seconds_per_image"] = round(sec / max(len(rows), 1), 3)
    return m


def main():
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None, help="LoRA 어댑터 폴더 (없으면 기본 모델 zero-shot)")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    rows = [json.loads(x) for x in Path(args.data).read_text(encoding="utf-8").splitlines() if x.strip()]
    rows = rows[:args.limit] if args.limit else rows
    processor = AutoProcessor.from_pretrained(args.adapter or args.model)
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model, dtype=torch.bfloat16,
                                                            attn_implementation="sdpa", device_map={"": 0})
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    m = evaluate_records(model, processor, rows, Path(args.out), args.batch_size)
    print(json.dumps(m, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
