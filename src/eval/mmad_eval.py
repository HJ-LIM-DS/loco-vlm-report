"""MMAD 객관식 정확도: 학습 전(기본 모델)과 학습 후(LoRA)를 같은 이미지에서 비교한다 (전이·망각 측정).

- MMAD는 학습에 쓰지 않는다. annotation: false 문항은 인덱스 단계에서 이미 뺐다(qa_json).
- summary.json(test 전체 요약)은 누수라 쓰지 않는다. 논문 수치와 직접 비교하지 않는다.
- 입력: 검사 이미지 1장(약 256토큰 캐시) + 질문 + 보기, "보기 글자 하나로 답"하도록 지시.
- 대상: --fold의 held-out 이미지 (LoRA는 그 이미지를 학습하지 않음).

사용법:
    python -m src.eval.mmad_eval --fold 0 --out outputs/mmad/base_f0.json
    python -m src.eval.mmad_eval --fold 0 --adapter runs/m2_4b_f0_s0/final --out outputs/mmad/lora_f0.json
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "vlm_cache" / "img"


def questions(fold: int) -> list:
    df = pd.read_parquet(ROOT / "data" / "index.parquet").merge(
        pd.read_csv(ROOT / "data" / "splits.csv")[["image_id", "fold"]], on="image_id")
    held = df[(df["split"] == "test") & (df["fold"] != fold) & df["qa_json"].notna()]
    out = []
    for r in held.itertuples():
        for q in json.loads(r.qa_json):
            if "Options" not in q or "Answer" not in q:
                continue
            out.append({"image_id": r.image_id, "image_path": r.image_path, "type": q.get("type") or "other",
                        "question": q["Question"], "options": q["Options"], "answer": q["Answer"]})
    return out


def prompt(q: dict) -> str:
    opts = "\n".join(f"{k}. {v}" for k, v in q["options"].items())
    return f"{q['question']}\n{opts}\nAnswer with the option's letter from the given choices directly."


@torch.no_grad()
def answer(model, processor, qs: list, batch_size: int, ctx=None) -> list:
    from src.vlm.build_sft import cached_full

    tok = processor.tokenizer
    outs = []
    for i in range(0, len(qs), batch_size):
        b = qs[i:i + batch_size]
        if ctx is not None:
            ctx.heartbeat(step=i, phase=f"mmad {i}/{len(qs)}")
        texts = [processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt(q)}]}],
            add_generation_prompt=True, tokenize=False) for q in b]
        imgs = [[Image.open(cached_full(q["image_path"])).convert("RGB")] for q in b]
        enc = processor(text=texts, images=imgs, return_tensors="pt", padding=True, padding_side="left").to(model.device)
        gen = model.generate(**enc, max_new_tokens=4, do_sample=False, pad_token_id=tok.pad_token_id)
        for g in gen[:, enc.input_ids.shape[1]:]:
            m = re.search(r"[A-E]", tok.decode(g, skip_special_tokens=True))
            outs.append(m.group(0) if m else None)
    return outs


def run(args, ctx=None):
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    qs = questions(args.fold)
    processor = AutoProcessor.from_pretrained(args.model)
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model, dtype=torch.bfloat16,
                                                            attn_implementation="sdpa", device_map={"": 0})
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    preds = answer(model, processor, qs, args.batch_size, ctx)
    by = defaultdict(lambda: [0, 0])
    for q, p in zip(qs, preds):
        by[q["type"]][0] += p == q["answer"]
        by[q["type"]][1] += 1
    total = sum(v[0] for v in by.values()) / max(sum(v[1] for v in by.values()), 1)
    res = {"model": args.model, "adapter": args.adapter, "fold": args.fold, "n_questions": len(qs),
           "accuracy": total, "by_type": {k: {"acc": v[0] / v[1], "n": v[1]} for k, v in sorted(by.items())},
           "unparsed": sum(p is None for p in preds) / max(len(preds), 1)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    with open(Path(args.out).with_suffix(".preds.jsonl"), "w", encoding="utf-8") as f:
        for q, p in zip(qs, preds):
            f.write(json.dumps({"image_id": q["image_id"], "type": q["type"], "answer": q["answer"], "pred": p},
                               ensure_ascii=False) + "\n")
    print(json.dumps(res, ensure_ascii=False, indent=1))


def main():
    from ops.train_hooks import RunContext, run_main

    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True, help="학습 fold (평가는 다른 fold 이미지)")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    def task(a):
        ctx = RunContext.from_env("mmad_eval")
        ctx.heartbeat(step=0, force=True, phase="generate")
        run(a, ctx)
        ctx.mark_done()

    run_main(task, args)


if __name__ == "__main__":
    main()
