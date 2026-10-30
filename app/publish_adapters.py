"""LoRA 어댑터 공개 준비: 모델 카드를 만들고(기본), 지시가 있으면 Hugging Face Hub에 올린다(--upload).

- 카드: runs/<run>/final/summary.json·adapter_config.json, data/sft/<fold>/meta.json, reports/tables/table1_main.json에서
  학습 설정·표본 수·held-out 지표·비용을 채워 reports/adapters/<run>/README.md로 저장한다.
- 라이선스: 학습 데이터(MVTec LOCO AD)가 CC BY-NC-SA 4.0이므로 어댑터도 같은 조건(비상업·동일 조건)으로 표기한다.
- manifest: 어댑터 파일의 sha256·크기를 reports/adapters/manifest.json에 적어 올린 가중치를 대조할 수 있게 한다.
- 업로드는 사용자가 직접 실행한다. 토큰은 환경 변수 HF_TOKEN(또는 huggingface-cli login 캐시)에서 읽고 출력하지 않는다.

    python -m app.publish_adapters                                  # 카드·manifest만 생성
    python -m app.publish_adapters --upload --namespace <hf_id>     # 각 run을 <hf_id>/loco-vlm-<run>으로 올림
"""

import argparse
import json
import os
from pathlib import Path

from src.report.publish_predictions import sha256

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "adapters"
RUNS = ["m2_2b_f0_s0", "m2_2b_f1_s0", "m2_4b_f0_s0", "m2_4b_f1_s0", "m2_4b_f0_s1", "m2_4b_f1_s1",
        "m2_4b_f0_s2", "m2_4b_f1_s2", "p2_8b_f0", "p2_8b_f1"]
FAMILY = {"m2_2b": "Ours 2B", "m2_4b": "Ours 4B", "p2_8b": "P2: 8B"}
UPLOAD_FILES = ("adapter_config.json", "adapter_model.safetensors", "chat_template.jinja", "processor_config.json",
                "tokenizer.json", "tokenizer_config.json")
LICENSE = "cc-by-nc-sa-4.0"


def repo_id(namespace: str, run: str) -> str:
    return f"{namespace}/loco-vlm-{run}"


def load_inputs(run: str, root: Path = ROOT) -> dict:
    final = root / "runs" / run / "final"
    s = json.loads((final / "summary.json").read_text(encoding="utf-8"))
    cfg = json.loads((final / "adapter_config.json").read_text(encoding="utf-8"))
    fold = int(s["data"].rstrip("/")[-1])
    meta_p = root / s["data"] / "meta.json"
    meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}
    t1_p = root / "reports" / "tables" / "table1_main.json"
    t1 = json.loads(t1_p.read_text(encoding="utf-8")) if t1_p.exists() else {}
    fam = FAMILY.get(run[:5])
    return {"run": run, "fold": fold, "summary": s, "config": cfg, "meta": meta, "family": fam,
            "table1": t1.get(fam) if fam else None}


def pct(v) -> str:
    return "-" if v is None else f"{100 * float(v):.1f}"


def card(inp: dict, repo_url: str = "", hub_id: str = "") -> str:
    s, cfg, meta, run, fold = inp["summary"], inp["config"], inp["meta"], inp["run"], inp["fold"]
    a = s.get("args", {})
    base = s["model"]
    quant = "4비트 NF4(QLoRA, 이중 양자화, bf16 연산)" if s.get("quant") == "nf4" else "bf16(양자화 없음)"
    counts = meta.get("counts", {})
    tr, va = counts.get("train", {}), counts.get("val", {})
    hg = s.get("heldout_generation") or {}
    raw = hg.get("raw", {})
    t1 = inp.get("table1") or {}
    m1 = t1.get("metrics", {})
    e2e = t1.get("end_to_end_group_f1", {})
    hours = s.get("train_runtime", 0) / 3600
    nf4_snippet = ("    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type=\"nf4\",\n"
                   "        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,\n"
                   "        llm_int8_skip_modules=[\"visual\", \"lm_head\"]),\n") if s.get("quant") == "nf4" else ""
    hub = hub_id or f"<namespace>/loco-vlm-{run}"
    lines = [
        "---",
        f"base_model: {base}",
        "library_name: peft",
        f"license: {LICENSE}",
        "language:",
        "- ko",
        "pipeline_tag: image-text-to-text",
        "tags:",
        "- lora", "- qwen3-vl", "- anomaly-detection", "- industrial-inspection", "- mvtec-loco",
        f"- base_model:adapter:{base}",
        "---",
        "",
        f"# LOCO VLM 이상 리포트 LoRA 어댑터: `{run}`",
        "",
        f"English summary: a rank-{cfg.get('r')} LoRA adapter for `{base}` that turns an industrial inspection image "
        "(plus a detector crop and a normal reference) into a fixed two-line Korean anomaly report on MVTec LOCO AD. "
        f"Trained on the training side of fold {fold}; evaluate it only on the other fold. Non-commercial (CC BY-NC-SA 4.0) "
        "because the training data carries that license.",
        "",
        "## 무엇을 하는 어댑터인가",
        "",
        "- 입력: 이미지 3장(검사 이미지, 탐지기 의심 영역 crop에 윤곽선 표시, 최근접 정상 참조; 각 약 256 토큰)과",
        "  제품·정상 규격·결함 그룹 목록·탐지기 백분위를 담은 한국어 프롬프트(저장소의 `src/vlm/prompts.py`).",
        "- 출력: 정상이면 `정상` 한 단어, 이상이면 고정 2줄 `이상: [논리|구조] <결함 그룹>[ - <세부>][ (<위치>)]` / `조치: <조치>`.",
        f"- 학습 데이터: MVTec LOCO AD test 이미지의 2-fold 분할 중 fold {fold} 학습 측"
        f"(학습 정상 {tr.get('good', '-')} / 이상 {tr.get('anomaly', '-')}장, inner-val 정상 {va.get('good', '-')} / 이상 {va.get('anomaly', '-')}장,"
        f" 정상 하드 네거티브 {meta.get('hard_neg', '-')}장, 목록 드롭아웃 {meta.get('dropout', '-')}). "
        f"fold {1 - fold} 이미지는 학습에 쓰지 않았으므로 평가는 그 fold로만 한다.",
        "- 탐지기: SALAD(ICCV 2025) 공식 가중치 재현 캐시. crop과 백분위는 이 캐시에서 나온다.",
        "",
        "## 학습 설정",
        "",
        "| 항목 | 값 |",
        "|---|---|",
        f"| 기반 모델 | `{base}` ({quant}) |",
        f"| LoRA | r={cfg.get('r')}, alpha={cfg.get('lora_alpha')}, dropout={cfg.get('lora_dropout')}, 언어 모델 q/k/v/o·gate/up/down, 비전 인코더 동결 |",
        f"| 최적화 | lr {a.get('lr')}, epoch {a.get('epochs')}, 배치 {a.get('batch_size')} x 누적 {a.get('grad_accum')}, warmup {a.get('warmup_steps')} step, seed {a.get('seed')} |",
        f"| 손실 | 정답 토큰(assistant 턴) logits만 계산 |",
        f"| 비용 | 학습 {hours:.1f}h (RTX 3060 12GB), 최대 예약 VRAM {s.get('peak_vram_reserved_gb', '-')}GB, "
        f"생성 {hg.get('seconds_per_image', float('nan')):.1f}초/장(배치 {a.get('gen_batch_size', '-')}) |",
    ]
    if run.startswith("p2_8b"):
        lines += ["", "생성 지연은 학습 프로세스 안에서 예약 VRAM 11.5GB가 유지된 채 잰 값이라 과대 측정이다. 별도 프로세스에서 다시 재면 "
                  "약 7.5초/장이다(저장소 `reports/ops/gen_probe_desktop.json`)."]
    lines += [
        "",
        f"## 결과 (held-out fold {1 - fold}, 원출력, n={hg.get('n', '-')})",
        "",
        "| 형식 준수 | 정상/이상 판정 | 결함 그룹 | 논리/구조 유형 | 세부 결함 | 위치 |",
        "|---|---|---|---|---|---|",
        f"| {pct(raw.get('format_ok'))} | {pct(raw.get('normal_decision_acc'))} | {pct(raw.get('group_acc'))} | "
        f"{pct(raw.get('type_acc'))} | {pct(raw.get('detail_exact'))} | {pct(raw.get('location_exact'))} |",
        "",
    ]
    if t1:
        lines += [
            f"두 fold 합산 표1 기준(`{inp['family']}`{', seed ' + str(t1.get('n_seeds')) + '개 평균' if t1.get('n_seeds', 1) > 1 else ''}): "
            f"그룹 {pct(m1.get('group_acc'))}, 유형 {pct(m1.get('type_acc'))}, 세부 {pct(m1.get('detail_exact'))}, "
            f"위치 적중 {pct(m1.get('location_hit'))}, 정상 거부 {pct(m1.get('normal_correct'))}, "
            f"종단 그룹 F1 {pct(e2e.get('test'))} (π=1%: {pct(e2e.get('pi=0.01'))}).",
            "",
        ]
    lines += [
        "주의: 같은 입력을 쓰는 비-VLM 기준선(DINOv2 + SALAD 점수 -> 로지스틱 회귀)이 그룹 정확도 67.0으로 이 어댑터보다 높다. "
        "자세한 비교와 한계는 저장소 README를 본다.",
        "",
        "## 사용법",
        "",
        "```python",
        "import torch",
        "from peft import PeftModel",
        "from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration",
        "",
        f"base = Qwen3VLForConditionalGeneration.from_pretrained(\"{base}\", dtype=torch.bfloat16,",
        nf4_snippet + "    attn_implementation=\"sdpa\", device_map={\"\": 0})",
        f"model = PeftModel.from_pretrained(base, \"{hub}\").eval()",
        f"processor = AutoProcessor.from_pretrained(\"{hub}\")",
        "# 메시지: 이미지 3장 + 프롬프트(src/vlm/prompts.py의 user_text), greedy 생성 max_new_tokens=80",
        "```",
        "",
        "## 라이선스와 한계",
        "",
        "- 어댑터 가중치: CC BY-NC-SA 4.0. MVTec LOCO AD(Bergmann et al., IJCV 2022, CC BY-NC-SA 4.0)로 학습했으므로 비상업·동일 조건을 따른다."
        " 기반 모델 Qwen3-VL은 Apache-2.0.",
        "- 조치 문장은 포트폴리오용 가상 SOP이며 실제 현장 절차가 아니다.",
        "- 결함 그룹 36개 중 여럿이 10장 미만이고, 공개 데이터셋이라 기반 모델의 사전학습 노출 가능성을 배제하지 못한다.",
        "- 대상 fold가 아닌 이미지(학습에 쓴 fold)에 대한 수치는 과대평가다.",
    ]
    if repo_url:
        lines += ["", f"저장소: {repo_url}"]
    return "\n".join(lines) + "\n"


def build(runs: list, root: Path = ROOT, out: Path = OUT, repo_url: str = "", namespace: str = "") -> dict:
    manifest = {"license": LICENSE, "runs": {}}
    for run in runs:
        final = root / "runs" / run / "final"
        if not (final / "adapter_model.safetensors").exists():
            print(f"건너뜀(어댑터 없음): {run}")
            continue
        inp = load_inputs(run, root)
        d = out / run
        d.mkdir(parents=True, exist_ok=True)
        hub = repo_id(namespace, run) if namespace else ""
        (d / "README.md").write_text(card(inp, repo_url, hub), encoding="utf-8")
        files = {n: {"bytes": (final / n).stat().st_size, "sha256": sha256(final / n)} for n in UPLOAD_FILES if (final / n).exists()}
        manifest["runs"][run] = {"repo_id": hub or None, "base_model": inp["summary"]["model"], "fold": inp["fold"], "files": files}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    return manifest


def upload(manifest: dict, namespace: str, root: Path = ROOT, out: Path = OUT, private: bool = False) -> list:
    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN"))  # 없으면 huggingface-cli login 캐시를 쓴다
    done = []
    for run, info in manifest["runs"].items():
        rid = repo_id(namespace, run)
        final = root / "runs" / run / "final"
        api.create_repo(rid, exist_ok=True, private=private)
        ops = [CommitOperationAdd(path_in_repo=n, path_or_fileobj=str(final / n)) for n in info["files"]]
        ops.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=str(out / run / "README.md")))
        api.create_commit(repo_id=rid, operations=ops, commit_message=f"{run}: LoRA adapter, processor, model card")
        done.append(rid)
        print(f"올림: {rid}")
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=RUNS)
    ap.add_argument("--namespace", default="", help="Hugging Face 사용자·조직 이름")
    ap.add_argument("--repo-url", default="", help="카드에 적을 코드 저장소 URL")
    ap.add_argument("--upload", action="store_true", help="실제로 Hub에 올린다 (HF_TOKEN 필요)")
    ap.add_argument("--private", action="store_true")
    a = ap.parse_args()
    if a.upload and not a.namespace:
        raise SystemExit("--upload에는 --namespace가 필요하다")
    m = build(a.runs, repo_url=a.repo_url, namespace=a.namespace)
    total = sum(f["bytes"] for r in m["runs"].values() for f in r["files"].values())
    print(f"카드 {len(m['runs'])}개 -> {OUT}, 올릴 파일 합계 {total / 1e9:.2f}GB")
    if a.upload:
        upload(m, a.namespace, private=a.private)


if __name__ == "__main__":
    main()
