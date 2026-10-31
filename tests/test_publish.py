"""예측 원본 공개(export/restore/verify) 왕복 시험."""

import json

import pandas as pd
import pytest

from src.report import publish_predictions as pp


@pytest.fixture
def fake_root(tmp_path):
    root = tmp_path / "root"
    (root / "runs" / "m2_4b_f0_s0" / "final").mkdir(parents=True)
    (root / "runs" / "m2_4b_f0_s0" / "final" / "heldout_predictions.jsonl").write_text(
        '{"id": "a/test/good/000", "target": "정상", "pred": "정상"}\n', encoding="utf-8")
    (root / "runs" / "m2_4b_f0_s0" / "final" / "summary.json").write_text('{"train_runtime": 1}', encoding="utf-8")
    (root / "runs" / "_smoke" / "final").mkdir(parents=True)  # 스모크는 제외
    (root / "runs" / "_smoke" / "final" / "heldout_predictions.jsonl").write_text("{}\n", encoding="utf-8")
    (root / "runs" / "m2_4b_f0_s1").mkdir()  # 예측 없는 run은 제외
    (root / "outputs" / "baselines" / "b3_zero_f0").mkdir(parents=True)
    (root / "outputs" / "baselines" / "b1_f0_heldout.jsonl").write_text("{}\n", encoding="utf-8")
    (root / "outputs" / "baselines" / "b3_zero_f0" / "strict.jsonl").write_text("{}\n", encoding="utf-8")
    (root / "outputs" / "detector" / "salad" / "pushpins").mkdir(parents=True)
    (root / "outputs" / "detector" / "salad" / "pushpins" / "scores.csv").write_text("image_id,score\nx,1\n", encoding="utf-8")
    (root / "outputs" / "detector" / "salad" / "pushpins" / "meta.json").write_text("{}", encoding="utf-8")
    (root / "data" / "sft" / "m2_salad_f0").mkdir(parents=True)
    (root / "data" / "sft" / "m2_salad_f0" / "meta.json").write_text('{"counts": {}}', encoding="utf-8")
    (root / "data" / "sft" / "m2_salad_f0" / "train.jsonl").write_text("{}\n", encoding="utf-8")  # 프롬프트는 제외
    (root / "data" / "sft" / "m2_salad_f0" / "heldout.jsonl").write_text(
        '{"id": "a/test/good/000", "label": "good", "prompt": "p"}\n', encoding="utf-8")  # 힌트 없음 -> 공개 안 함
    (root / "data" / "sft" / "p3_hints_f0").mkdir(parents=True)
    (root / "data" / "sft" / "p3_hints_f0" / "heldout.jsonl").write_text(
        '{"id": "a/test/good/000", "label": "good", "gt_report": "정상", "hints": [["정상", 0.9]], "prompt": "비공개", '
        '"images": ["x.png"]}\n'
        '{"id": "a/test/bad/000", "label": "logical_anomalies", "gt_report": "이상: x", "hints": null, "prompt": "비공개"}\n',
        encoding="utf-8")
    pd.DataFrame({"image_id": ["a/test/good/000"], "category": ["a"], "label": ["good"], "gt_report": ["정상"],
                  "caption_path": ["a.txt"], "qa_anomaly": ["비밀 텍스트"], "qa_json": ["{}"]}).to_parquet(
        root / "data" / "index.parquet", index=False)
    return root


def test_export_restore_roundtrip(fake_root, tmp_path):
    dest = tmp_path / "pub"
    m = pp.export(fake_root, dest)
    paths = {f["path"] for f in m["files"]}
    assert paths == {"runs/m2_4b_f0_s0/heldout_predictions.jsonl", "runs/m2_4b_f0_s0/summary.json",
                     "baselines/b1_f0_heldout.jsonl", "baselines/b3_zero_f0/strict.jsonl",
                     "detector/salad/pushpins/scores.csv", "detector/salad/pushpins/meta.json",
                     "sft_meta/m2_salad_f0/meta.json", "sft_hints/p3_hints_f0/heldout_hints.jsonl", "index_slim.parquet"}
    # 힌트 기록은 id·라벨·정답·힌트만 남고(프롬프트·이미지 경로 제외), 힌트 없는 줄은 빠진다
    hint_rows = [json.loads(x) for x in
                 (dest / "sft_hints" / "p3_hints_f0" / "heldout_hints.jsonl").read_text(encoding="utf-8").splitlines()]
    assert hint_rows == [{"id": "a/test/good/000", "label": "good", "gt_report": "정상", "hints": [["정상", 0.9]]}]
    # MMAD QA 텍스트와 캡션 경로는 빠지고 라벨 열은 남는다
    slim = pd.read_parquet(dest / "index_slim.parquet")
    assert list(slim.columns) == ["image_id", "category", "label", "gt_report"] == m["index_columns"]
    assert pp.verify(dest) == []
    assert m["total_bytes"] == sum(f["bytes"] for f in m["files"])

    other = tmp_path / "other"
    r = pp.restore(other, dest)
    assert len(r["copied"]) == 9 and r["skipped"] == []
    assert (other / "data" / "sft" / "p3_hints_f0" / "heldout_hints.jsonl").exists()
    assert (other / "runs" / "m2_4b_f0_s0" / "final" / "heldout_predictions.jsonl").read_text(encoding="utf-8") \
        == (fake_root / "runs" / "m2_4b_f0_s0" / "final" / "heldout_predictions.jsonl").read_text(encoding="utf-8")
    assert (other / "outputs" / "detector" / "salad" / "pushpins" / "scores.csv").exists()
    assert (other / "data" / "sft" / "m2_salad_f0" / "meta.json").exists()
    assert "qa_anomaly" not in pd.read_parquet(other / "data" / "index.parquet").columns

    # 있는 파일은 건드리지 않고, force면 덮어쓴다
    target = other / "outputs" / "baselines" / "b1_f0_heldout.jsonl"
    target.write_text("local\n", encoding="utf-8")
    r2 = pp.restore(other, dest)
    assert "baselines/b1_f0_heldout.jsonl" in r2["skipped"] and target.read_text(encoding="utf-8") == "local\n"
    pp.restore(other, dest, force=True)
    assert target.read_text(encoding="utf-8") == "{}\n"


def test_verify_detects_change(fake_root, tmp_path):
    dest = tmp_path / "pub"
    pp.export(fake_root, dest)
    (dest / "baselines" / "b1_f0_heldout.jsonl").write_text("changed\n", encoding="utf-8")
    assert pp.verify(dest) == ["baselines/b1_f0_heldout.jsonl"]


def _fake_run(root, run, model, quant, fold):
    final = root / "runs" / run / "final"
    final.mkdir(parents=True)
    (final / "summary.json").write_text(json.dumps({
        "model": model, "quant": quant, "data": f"data/sft/m2_salad_f{fold}", "train_runtime": 4901.6,
        "peak_vram_reserved_gb": 6.32,
        "args": {"lr": 1e-4, "epochs": 3.0, "batch_size": 1, "grad_accum": 8, "warmup_steps": 5, "seed": 0, "gen_batch_size": 2},
        "heldout_generation": {"n": 783, "seconds_per_image": 5.7, "raw": {
            "format_ok": 0.999, "normal_decision_acc": 0.9, "group_acc": 0.562, "type_acc": 0.715,
            "detail_exact": 0.44, "location_exact": 0.6}}}), encoding="utf-8")
    (final / "adapter_config.json").write_text(json.dumps({"r": 16, "lora_alpha": 32, "lora_dropout": 0.05}), encoding="utf-8")
    (final / "adapter_model.safetensors").write_bytes(b"x" * 10)
    (final / "tokenizer.json").write_text("{}", encoding="utf-8")
    meta = root / "data" / "sft" / f"m2_salad_f{fold}"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "meta.json").write_text(json.dumps({"counts": {"train": {"good": 149, "anomaly": 418},
                                                           "val": {"good": 43, "anomaly": 79}},
                                                "hard_neg": 150, "dropout": 0.12}), encoding="utf-8")


def test_adapter_cards(tmp_path):
    from app import publish_adapters as pa

    _fake_run(tmp_path, "m2_4b_f0_s0", "Qwen/Qwen3-VL-4B-Instruct", "nf4", 0)
    _fake_run(tmp_path, "m2_2b_f1_s0", "Qwen/Qwen3-VL-2B-Instruct", "none", 1)
    m = pa.build(["m2_4b_f0_s0", "m2_2b_f1_s0", "missing_run"], root=tmp_path, out=tmp_path / "cards", namespace="someone")
    assert list(m["runs"]) == ["m2_4b_f0_s0", "m2_2b_f1_s0"]
    assert m["runs"]["m2_4b_f0_s0"]["repo_id"] == "someone/loco-vlm-m2_4b_f0_s0"
    assert set(m["runs"]["m2_4b_f0_s0"]["files"]) == {"adapter_config.json", "adapter_model.safetensors", "tokenizer.json"}
    assert m["runs"]["m2_4b_f0_s0"]["files"]["adapter_model.safetensors"]["bytes"] == 10
    four = (tmp_path / "cards" / "m2_4b_f0_s0" / "README.md").read_text(encoding="utf-8")
    assert four.startswith("---\nbase_model: Qwen/Qwen3-VL-4B-Instruct\n") and "license: cc-by-nc-sa-4.0" in four
    assert "someone/loco-vlm-m2_4b_f0_s0" in four and "fold 0 학습 측" in four and "418장" in four
    assert "| 99.9 | 90.0 | 56.2 | 71.5 | 44.0 | 60.0 |" in four and "bnb_4bit_quant_type" in four
    assert "held-out fold 1" in four
    two = (tmp_path / "cards" / "m2_2b_f1_s0" / "README.md").read_text(encoding="utf-8")
    assert "fold 1 학습 측" in two and "held-out fold 0" in two and "bnb_4bit" not in two and "bf16(양자화 없음)" in two
    assert json.loads((tmp_path / "cards" / "manifest.json").read_text(encoding="utf-8"))["license"] == "cc-by-nc-sa-4.0"


def test_target_of_rejects_unknown(tmp_path):
    with pytest.raises(ValueError):
        pp.target_of(tmp_path, "somewhere/else.txt")
    assert pp.target_of(tmp_path, "runs/x/summary.json") == tmp_path / "runs" / "x" / "final" / "summary.json"
    m = json.loads('{"path": "index_slim.parquet"}')
    assert pp.target_of(tmp_path, m["path"]) == tmp_path / "data" / "index.parquet"
