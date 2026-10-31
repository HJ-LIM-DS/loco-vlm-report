"""Qwen3-VL LoRA(bf16) / QLoRA(NF4) SFT. 감독자(ops.supervisor) 규약을 따른다.

- 손실은 정답 토큰 위치의 logits만 계산한다 (왼쪽 패딩 + logits_to_keep). 전체 위치를 계산하면
  어휘 15만 x 길이 1.2천의 fp32 logits로 약 1.5GB가 더 든다.
- LoRA: 언어 모델의 q/k/v/o + MLP. 비전 인코더 LoRA는 --vision-lora (PEFT 비교 실험).
- QLoRA: 언어 모델만 NF4로 양자화하고 비전 인코더·lm_head는 양자화하지 않는다.
- 체크포인트: 약 save_every_minutes마다 저장, 최근 2개 유지, optimizer·scheduler·RNG·데이터 위치 포함,
  저장이 끝나면 COMPLETE 표식. 재개는 표식이 있고 파일 검증을 통과한 최신 체크포인트에서.
- LOCO_SMOKE=1: 20 step, 10 step마다 저장, 가장 긴 샘플부터 순서대로, accumulation 1.

사용법:
    python -m src.vlm.train_lora --data data/sft/m1_gt_f0 --model Qwen/Qwen3-VL-2B-Instruct --epochs 3
"""

import argparse
import contextlib
import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, SequentialSampler

from ops.train_hooks import RunContext, hf_callback, latest_valid_checkpoint, run_main

ROOT = Path(__file__).resolve().parents[2]
LM_TARGETS = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
VISION_TARGETS = r".*visual\.blocks.*\.(attn\.qkv|attn\.proj|mlp\.linear_fc1|mlp\.linear_fc2)"


class SFTDataset(Dataset):
    def __init__(self, path: Path, limit: int | None = None):
        self.rows = [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]
        if limit:
            self.rows = self.rows[:limit]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        return {"images": [Image.open(ROOT / p).convert("RGB") for p in r["images"]],
                "prompt": r["prompt"], "target": r["target"], "id": r["id"]}


class Collator:
    """샘플별로 프롬프트를 처리하고 정답 토큰을 이어 붙인 뒤 왼쪽 패딩한다."""

    def __init__(self, processor):
        from src.vlm.prompts import messages

        self.p = processor
        self.messages = messages
        self.tok = processor.tokenizer
        self.end_id = self.tok.convert_tokens_to_ids("<|im_end|>")
        self.pad_id = self.tok.pad_token_id

    def encode(self, ex):
        text = self.p.apply_chat_template(self.messages(ex["prompt"], len(ex["images"])),
                                          add_generation_prompt=True, tokenize=False)
        enc = self.p(text=[text], images=ex["images"], return_tensors="pt")
        tgt = self.tok(ex["target"], add_special_tokens=False)["input_ids"] + [self.end_id]
        return enc, tgt

    def __call__(self, batch):
        encs = [self.encode(ex) for ex in batch]
        lens = [e.input_ids.shape[1] + len(t) for e, t in encs]
        L, B = max(lens), len(batch)
        ids = torch.full((B, L), self.pad_id, dtype=torch.long)
        att = torch.zeros((B, L), dtype=torch.long)
        labels = torch.full((B, L), -100, dtype=torch.long)
        # Qwen3-VL 위치 인코딩(M-RoPE)용 토큰 종류 (이미지 토큰 표시). 정답·패딩 위치는 텍스트(0)
        mm = torch.zeros((B, L), dtype=torch.long)
        for b, ((e, t), n) in enumerate(zip(encs, lens)):
            seq = torch.cat([e.input_ids[0], torch.tensor(t)])
            ids[b, L - n:] = seq
            att[b, L - n:] = 1
            labels[b, L - len(t):] = torch.tensor(t)
            mm[b, L - n:L - len(t)] = e["mm_token_type_ids"][0]
        return {
            "input_ids": ids, "attention_mask": att, "labels": labels, "mm_token_type_ids": mm,
            "pixel_values": torch.cat([e.pixel_values for e, _ in encs]),
            "image_grid_thw": torch.cat([e.image_grid_thw for e, _ in encs]),
            "n_keep": max(len(t) for _, t in encs),
        }


@contextlib.contextmanager
def trusted_checkpoint(checkpoint, output_dir):
    """이 run이 직접 저장한 체크포인트를 재개할 때만 torch 버전 검사를 건너뛴다.

    transformers는 torch < 2.6에서 torch.load(weights_only=True)도 막는다(CVE-2025-32434: 조작된 파일 위험).
    또 torch 2.5의 weights_only 허용 목록에는 RNG 상태 파일의 numpy 배열이 없다.
    이 환경은 torch 2.5.1이고, 읽는 파일은 같은 run의 출력 폴더 안에 이 run이 직접 저장한 optimizer·scheduler·RNG
    상태뿐이므로, 경로가 출력 폴더 안일 때 해당 호출 동안만 검사를 끄고 전체 로드를 허용한다.
    torch를 2.6 이상으로 올리면 검사 우회는 필요 없다.
    """
    import transformers.trainer as tt

    ckpt = Path(checkpoint).resolve()
    if Path(output_dir).resolve() not in ckpt.parents:
        raise ValueError(f"출력 폴더 밖 체크포인트는 신뢰하지 않음: {ckpt}")
    original_check, original_load = tt.check_torch_load_is_safe, torch.load

    def trusted_load(f, *a, **kw):
        if Path(f).resolve().parent != ckpt:
            raise ValueError(f"재개 중 체크포인트 밖 파일 로드 시도: {f}")
        kw["weights_only"] = False
        return original_load(f, *a, **kw)

    tt.check_torch_load_is_safe = lambda: None
    torch.load = trusted_load
    try:
        yield
    finally:
        tt.check_torch_load_is_safe = original_check
        torch.load = original_load


def make_trainer_cls():
    from transformers import Trainer

    class AnswerOnlyTrainer(Trainer):
        """정답 토큰 위치의 logits만 계산해 손실을 구한다."""

        def __init__(self, *a, smoke: bool = False, **kw):
            super().__init__(*a, **kw)
            self.smoke = smoke
            # 손실을 (정답 토큰 합 / accumulation 창 전체 정답 토큰 수)로 직접 정규화하므로
            # Trainer가 accumulation 횟수로 한 번 더 나누지 않게 한다
            self.model_accepts_loss_kwargs = True

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")
            k = int(inputs.pop("n_keep"))
            out = model(**inputs, logits_to_keep=k + 1)
            logits = out.logits[:, :-1, :].float()
            tgt = labels[:, -k:].to(logits.device)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1),
                                   ignore_index=-100, reduction="sum")
            denom = num_items_in_batch if num_items_in_batch is not None else (tgt != -100).sum()
            loss = loss / denom
            return (loss, out) if return_outputs else loss

        def _load_optimizer_and_scheduler(self, checkpoint):
            if checkpoint is None:
                return super()._load_optimizer_and_scheduler(checkpoint)
            with trusted_checkpoint(checkpoint, self.args.output_dir):
                return super()._load_optimizer_and_scheduler(checkpoint)

        def _load_rng_state(self, checkpoint):
            if checkpoint is None:
                return super()._load_rng_state(checkpoint)
            with trusted_checkpoint(checkpoint, self.args.output_dir):
                return super()._load_rng_state(checkpoint)

        def _get_train_sampler(self, *a, **kw):
            if self.smoke:  # 가장 긴 샘플부터 순서대로 (OOM을 첫 step에서 드러냄)
                return SequentialSampler(self.train_dataset)
            return super()._get_train_sampler(*a, **kw)

    return AnswerOnlyTrainer


def load_model(args):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(args.model)
    kw = {"dtype": torch.bfloat16, "attn_implementation": "sdpa", "device_map": {"": 0}}
    if args.quant == "nf4":
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16, llm_int8_skip_modules=["visual", "lm_head"])
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model, **kw)
    if args.quant == "nf4":
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    targets = LM_TARGETS if not args.vision_lora else f"({LM_TARGETS})|({VISION_TARGETS})"
    model = get_peft_model(model, LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
                                             target_modules=targets, task_type="CAUSAL_LM"))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    return model, processor


def validate_checkpoint(path: Path) -> None:
    """파일이 잘리거나 깨진 체크포인트를 거른다 (F5)."""
    from safetensors import safe_open

    json.loads((path / "trainer_state.json").read_text(encoding="utf-8"))
    with safe_open(str(path / "adapter_model.safetensors"), framework="pt") as f:
        keys = list(f.keys())
        f.get_tensor(keys[-1])
    for name in ("optimizer.pt", "scheduler.pt"):
        if (path / name).stat().st_size == 0:
            raise ValueError(f"{name} 크기 0")


def train(args):
    from transformers import TrainingArguments

    ctx = RunContext.from_env(f"vlm_{Path(args.data).name}")
    if ctx.smoke:
        os.environ["MLFLOW_EXPERIMENT_NAME"] = "loco-vlm-smoke"
    torch.manual_seed(args.seed)
    train_ds = SFTDataset(Path(args.data) / "train.jsonl")
    val_ds = SFTDataset(Path(args.data) / "val.jsonl")
    model, processor = load_model(args)
    collator = Collator(processor)
    if ctx.smoke:
        lens = []
        for i in range(len(train_ds)):
            enc, t = collator.encode(train_ds[i])
            lens.append(enc.input_ids.shape[1] + len(t))
        order = sorted(range(len(train_ds)), key=lambda i: -lens[i])
        train_ds.rows = [train_ds.rows[i] for i in order]

    ck = latest_valid_checkpoint(ctx.ckpt_dir, validate_checkpoint)
    ctx.note_resume(ck)
    ta = TrainingArguments(
        output_dir=str(ctx.ckpt_dir),
        num_train_epochs=args.epochs,
        max_steps=20 if ctx.smoke else (args.max_steps or -1),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=1 if ctx.smoke else args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=0 if ctx.smoke else args.warmup_steps,
        weight_decay=0.0,
        bf16=True,
        logging_steps=1 if ctx.smoke else args.logging_steps,
        save_strategy="steps",
        # 실제 학습은 콜백이 시간 기준으로 저장을 지시한다 (--save-steps는 고장 주입 시험용)
        save_steps=10 if ctx.smoke else (args.save_steps or 10**9),
        save_total_limit=2,
        eval_strategy="no" if (ctx.smoke or not args.eval_loss) else "epoch",
        dataloader_num_workers=0 if ctx.smoke else args.workers,
        dataloader_persistent_workers=False,
        remove_unused_columns=False,
        # 스모크도 실제 설정대로 MLflow를 켜서 연동 오류를 미리 잡는다 (실험 이름만 분리)
        report_to=["mlflow"] if args.mlflow else [],
        run_name=f"{ctx.name}-a{ctx.attempt}",
        seed=args.seed,
        gradient_checkpointing=False,  # load_model에서 이미 켬
    )
    trainer = make_trainer_cls()(
        model=model, args=ta, train_dataset=train_ds,
        eval_dataset=None if (ctx.smoke or not args.eval_loss) else val_ds,
        data_collator=collator, smoke=ctx.smoke,
        callbacks=[hf_callback(ctx, save_every_minutes=None if ctx.smoke else args.save_every_minutes)],
    )
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    result = trainer.train(resume_from_checkpoint=str(ck) if ck else None)
    train_s = time.time() - t0

    ctx.heartbeat(step=trainer.state.global_step, force=True, phase="save_final")
    ctx.final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ctx.final_dir)
    processor.save_pretrained(ctx.final_dir)
    evals = [h for h in trainer.state.log_history if "eval_loss" in h]
    summary = {
        "model": args.model, "quant": args.quant, "vision_lora": args.vision_lora, "data": args.data,
        "train_samples": len(train_ds), "global_step": trainer.state.global_step,
        "train_seconds_this_attempt": round(train_s, 1),
        "train_runtime": result.metrics.get("train_runtime"),
        "peak_vram_alloc_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
        "peak_vram_reserved_gb": round(torch.cuda.max_memory_reserved() / 1024**3, 2),
        "eval_loss_by_epoch": [round(h["eval_loss"], 4) for h in evals],
        "train_loss_by_step": {int(h["step"]): h["loss"] for h in trainer.state.log_history if "loss" in h},
        "args": vars(args),
    }
    # 생성 전에 학습용 VRAM을 돌려준다. optimizer 상태, 기울기, allocator 캐시가 남으면 예약 메모리가 학습 최고치(8B는 11.5GB)에
    # 머문다. 그 상태에서 데스크톱의 GPU 사용(약 1GB)과 겹치면 Windows 그래픽 드라이버(WDDM)의 공유 메모리로 밀려 생성이
    # 2~4배 느려졌다. 8B 실측: 학습 프로세스 안 32.5초/장, 별도 프로세스 7.3초/장(ops/gen_probe.py).
    import gc

    gstep = trainer.state.global_step
    trainer.optimizer = trainer.lr_scheduler = None
    del trainer
    model.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    summary["vram_reserved_before_generation_gb"] = round(torch.cuda.memory_reserved() / 1024**3, 2)
    if not ctx.smoke and args.eval_generate:
        from src.vlm.infer import evaluate_records
        ctx.heartbeat(step=gstep, force=True, phase="eval_generate")
        model.config.use_cache = True
        summary["val_generation"] = evaluate_records(
            model, processor, val_ds.rows, ctx.final_dir / "val_predictions.jsonl", batch_size=args.gen_batch_size,
            on_batch=lambda i, n: ctx.heartbeat(step=gstep, phase=f"eval_generate {i}/{n}"))
    if not ctx.smoke and args.predict_heldout:
        # held-out fold 예측은 최종 평가용으로 한 번만 남긴다 (설정 선택에는 inner-val만 쓴다)
        from src.vlm.infer import evaluate_records
        ctx.heartbeat(step=gstep, force=True, phase="predict_heldout")
        model.config.use_cache = True
        held = SFTDataset(Path(args.data) / "heldout.jsonl").rows
        summary["heldout_generation"] = evaluate_records(
            model, processor, held, ctx.final_dir / "heldout_predictions.jsonl", batch_size=args.gen_batch_size,
            on_batch=lambda i, n: ctx.heartbeat(step=gstep, phase=f"predict_heldout {i}/{n}"))
    (ctx.final_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    ctx.mark_done(**{k: summary[k] for k in ("global_step", "peak_vram_alloc_gb", "peak_vram_reserved_gb",
                                              "train_runtime")})


def build_parser() -> argparse.ArgumentParser:
    """대기열 명령 검사 테스트(tests/test_ops.py)도 이 파서를 쓴다."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    ap.add_argument("--quant", choices=["none", "nf4"], default="none")
    ap.add_argument("--vision-lora", action="store_true")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--warmup-steps", type=int, default=5)
    ap.add_argument("--save-every-minutes", type=float, default=15)
    ap.add_argument("--save-steps", type=int, default=None, help="고장 주입 시험용 step 단위 저장")
    ap.add_argument("--logging-steps", type=int, default=5)
    ap.add_argument("--eval-loss", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--predict-heldout", action="store_true", help="학습 후 held-out fold 예측 저장")
    ap.add_argument("--gen-batch-size", type=int, default=4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    # 설치된 MLflow 3.17이 파일 저장소를 거부하고, SQLite 백엔드는 패키지 안의 모듈 누락으로 import에 실패해 기본값은 끈다.
    # 학습 기록은 summary.json에 남는다.
    ap.add_argument("--mlflow", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--eval-generate", action=argparse.BooleanOptionalAction, default=True)
    return ap


def main():
    args = build_parser().parse_args()
    os.environ.setdefault("MLFLOW_TRACKING_URI", (ROOT / "mlruns").as_uri())
    os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", "loco-vlm")
    run_main(train, args)


if __name__ == "__main__":
    main()
