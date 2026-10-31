"""Gradio 데모: held-out 갤러리 + 캐시.

기본(캐시 모드)은 GPU 없이 동작한다. 각 이미지는 그 이미지를 학습하지 않은 fold 모델(Ours 4B seed 0)의
저장된 예측과 SALAD 캐스케이드 단계(tau_low 학습 측 재현율 0.95)를 보여 주고, 비-VLM 기준선 B1을 같은 캐스케이드에
넣은 출력과 정답을 나란히 둔다(표1과 같은 비교). 세 칸 모두 그룹 정답 여부를 표시한다.
--live를 주면 같은 어댑터를 올려 선택한 이미지를 실제로 생성한다(학습 대기열이 GPU를 쓰는 중에는 쓰지 않는다).

    python -m app.demo                 # 캐시 모드, http://127.0.0.1:7860 (?id=<image_id>로 특정 이미지를 바로 연다)
    python -m app.demo --live          # 4B QLoRA 어댑터로 실시간 생성 (VRAM 약 6GB)
"""

import argparse
import json
import os
import time
from pathlib import Path

import gradio as gr
import pandas as pd
from PIL import Image

from src.eval.cascade import cascade, group_correct
from src.kb import load_kb
from src.report.m2_report import heldout
from src.report.m4_figures import CAT_KO, overlay
from src.vlm.prompts import IMAGE_ROLES

ROOT = Path(__file__).resolve().parents[1]
STAGE_KO = {"pass": "통과(점수 < τ_low): 무출력", "judge": "판정 구간: VLM이 정상/이상 판정",
            "describe": "설명 구간(점수 ≥ τ_high): VLM 설명, 거부 불가"}
LABEL_KO = {"good": "정상", "logical": "논리 이상", "structural": "구조 이상"}
NO_OUTPUT = "(무출력: 정상으로 판정)"


def load_state() -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """이미지별 캐스케이드 결과(Ours 4B s0), SFT 레코드(VLM 입력), B1을 같은 캐스케이드에 넣은 결과."""
    kb = load_kb()
    index = pd.read_parquet(ROOT / "data" / "index.parquet")
    splits = pd.read_csv(ROOT / "data" / "splits.csv")
    df = cascade({f: heldout(f"m2_4b_f{f}_s0") for f in (0, 1)}, 0.95, kb, index, splits).set_index("id")
    recs = {}
    for f in (0, 1):  # fold f 모델의 held-out = 다른 fold 이미지
        for line in (ROOT / "data" / "sft" / f"m2_salad_f{f}" / "heldout.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            recs[r["id"]] = {**r, "model_fold": f}
    b1 = cascade({f: ROOT / "outputs" / "baselines" / f"b1_f{f}_heldout.jsonl" for f in (0, 1)}, 0.95, kb, index, splits)
    return df, recs, b1.set_index("id")


def verdict(row: pd.Series) -> str:
    """최종 출력의 정답 여부 한 줄: 정상 이미지는 무출력이 정답, 이상 이미지는 결함 그룹 일치가 정답."""
    if row["label"] == "good":
        return "정답 (무출력)" if row["final"] == "" else "오경보"
    if row["final"] == "":
        return "놓침 (무출력)"
    return "그룹 정답" if group_correct(row["final"], row["gt_report"]) else "그룹 오답"


class LiveModel:
    """--live: fold별 어댑터를 필요할 때 올린다 (한 번에 하나)."""

    def __init__(self, run_fmt: str, model: str, quant: str):
        self.run_fmt, self.model_id, self.quant = run_fmt, model, quant
        self.fold, self.model, self.processor = None, None, None

    def get(self, fold: int):
        if self.fold == fold:
            return self.model, self.processor
        import torch
        from peft import PeftModel
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

        self.model = None
        torch.cuda.empty_cache()
        kw = {"dtype": torch.bfloat16, "attn_implementation": "sdpa", "device_map": {"": 0}}
        if self.quant == "nf4":
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16, llm_int8_skip_modules=["visual", "lm_head"])
        base = Qwen3VLForConditionalGeneration.from_pretrained(self.model_id, **kw)
        adapter = ROOT / "runs" / self.run_fmt.format(f=fold) / "final"
        self.model = PeftModel.from_pretrained(base, str(adapter)).eval()
        self.processor = AutoProcessor.from_pretrained(str(adapter))
        self.fold = fold
        return self.model, self.processor


def build(live: LiveModel | None) -> gr.Blocks:
    df, recs, b1 = load_state()
    idx = pd.read_parquet(ROOT / "data" / "index.parquet").set_index("image_id")
    fold_of = pd.read_csv(ROOT / "data" / "splits.csv").set_index("image_id")["fold"]
    kb = load_kb()

    def ids_for(cat: str, lab: str) -> list:
        sel = df[(df["category"] == cat) & ((df["label"] == lab) if lab != "전체" else True)]
        return sorted(sel.index)

    def show(iid: str):
        if not iid:
            return None, [], "", "", "", "", "", ""
        r, rec = df.loc[iid], recs[iid]
        det = rec.get("detector") or {}
        info = (f"**{CAT_KO[r['category']]}** `{iid}` ({LABEL_KO[r['label']]})  \n"
                f"SALAD 점수 백분위 {det.get('percentile', 0):.0f} (학습 측 정상 대비), 의심 위치 {det.get('location', '-')}  \n"
                f"단계: {STAGE_KO[r['stage']]}  \n예측 모델: fold {rec['model_fold']} 학습 모델 (이 이미지는 학습에 쓰지 않음)")
        gallery = [(str(ROOT / p), role) for p, role in zip(rec["images"], IMAGE_ROLES)]
        gt = r["gt_report"] if r["label"] != "good" else "정상 (출력 없음이 정답)"
        ours = f"{r['final'] or NO_OUTPUT}\n[{verdict(r)}]"
        if iid in b1.index:
            rb = b1.loc[iid]
            b1_text = f"{rb['final'] or NO_OUTPUT}\n[{verdict(rb)}]"
        else:
            b1_text = "-"
        return overlay(iid, fold_of, idx), gallery, info, ours, r["pred"], b1_text, gt, (b1.loc[iid, "pred"] if iid in b1.index else "-")

    def run_live(iid: str):
        if live is None or not iid:
            return "실시간 생성 불가: --live로 실행했을 때만 가능"
        from src.vlm.guard import guard
        from src.vlm.infer import generate

        rec = recs[iid]
        model, proc = live.get(rec["model_fold"])
        t0 = time.time()
        raw = generate(model, proc, [rec], batch_size=1)[0]
        out, steps = guard(raw, kb, rec["category"], (rec.get("detector") or {}).get("location"))
        return f"{out}\n\n(원출력: {raw})\n생성 {time.time() - t0:.1f}초, Guard 단계: {', '.join(steps) or '없음'}"

    cats = sorted(df["category"].unique())
    with gr.Blocks(title="LOCO 이상 리포트 데모") as demo:
        gr.Markdown("## MVTec LOCO: SALAD 탐지기 + QLoRA VLM 2줄 이상 리포트\n"
                    "held-out 이미지만 보여 준다. 주황 윤곽 = GT 결함, 파랑 상자 = 탐지기 crop 영역. "
                    "가운데 줄은 같은 SALAD 캐스케이드에 넣은 Ours(VLM)와 B1(비-VLM 기준선)의 최종 출력을 정답과 나란히 둔 것이다.")
        with gr.Row():
            cat = gr.Dropdown(cats, value=cats[0], label="카테고리")
            lab = gr.Dropdown(["전체", "good", "logical", "structural"], value="logical", label="라벨")
            first = ids_for(cats[0], "logical")
            iid = gr.Dropdown(first, value=first[0], label="이미지")
        with gr.Row():
            img = gr.Image(label="검사 이미지", type="numpy", height=360)
            info = gr.Markdown()
        with gr.Row(equal_height=True):
            with gr.Column():
                final = gr.Textbox(label="Ours: 캐스케이드 + 4B QLoRA VLM + Guard", lines=3)
                raw = gr.Textbox(label="VLM 원출력", lines=2)
            with gr.Column():
                b1_box = gr.Textbox(label="B1 비-VLM: DINOv2 + SALAD 점수 → 로지스틱, 같은 캐스케이드", lines=3)
                b1_raw = gr.Textbox(label="B1 원출력", lines=2)
            with gr.Column():
                gt = gr.Textbox(label="정답", lines=3)
        gal = gr.Gallery(label="VLM 입력 이미지 3장 (검사, 탐지기 crop, 최근접 정상 참조)", columns=3, height=300,
                         object_fit="contain")
        with gr.Row():
            btn = gr.Button("실시간 생성 (--live)", interactive=live is not None)
            live_out = gr.Textbox(label="실시간 생성 결과", lines=4)

        def refresh(c, lb):
            ids = ids_for(c, lb)
            return gr.Dropdown(choices=ids, value=ids[0] if ids else None)

        def on_load(request: gr.Request):
            # ?id=<image_id>: README 링크와 GIF 녹화용. 없거나 모르는 id면 기본 첫 이미지
            q = request.query_params.get("id") if request else None
            if q not in df.index:
                return (cats[0], "logical", gr.Dropdown(choices=first, value=first[0]), *show(first[0]))
            c, lb = df.loc[q, "category"], df.loc[q, "label"]
            return (c, lb, gr.Dropdown(choices=ids_for(c, lb), value=q), *show(q))

        # 카테고리·라벨은 사용자 입력에만 반응(.input): on_load가 값을 바꿀 때 이미지 목록이 다시 첫 장으로 돌아가지 않게
        outs = [img, gal, info, final, raw, b1_box, gt, b1_raw]
        cat.input(refresh, [cat, lab], iid)
        lab.input(refresh, [cat, lab], iid)
        iid.change(show, iid, outs)
        btn.click(run_live, iid, live_out)
        demo.load(on_load, None, [cat, lab, iid, *outs])
    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--run", default="m2_4b_f{f}_s0", help="--live 어댑터 run 이름 형식")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--quant", default="nf4", choices=["none", "nf4"])
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 7860)))
    a = ap.parse_args()
    live = LiveModel(a.run, a.model, a.quant) if a.live else None
    build(live).launch(server_name="127.0.0.1", server_port=a.port)


if __name__ == "__main__":
    main()
