"""결과 요약 표 (Markdown). 결과 파일만 읽어 reports/tables/에 쓴다.

    python -m src.report.tables m1
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "tables"


def _read(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _pct(v):
    return "-" if v is None else f"{100 * v:.1f}"


def m1_tables() -> Path:
    lines = ["# 선행 실험 결과 요약 (EfficientAD + 2B, 2개 카테고리)", "", "## 탐지기 (EfficientAD-M, test 전체)", "",
             "| 카테고리 | AUROC 전체 | 논리 | 구조 | 지연(ms/장) | 학습 step |", "|---|---|---|---|---|---|"]
    for cat in ("breakfast_box", "screw_bag"):
        s = _read(ROOT / "runs" / f"m1_effad_{cat}" / "final" / "summary.json")
        if s:
            lines.append(f"| {cat} | {_pct(s.get('auroc_all'))} | {_pct(s.get('auroc_logical'))} | "
                         f"{_pct(s.get('auroc_structural'))} | {s.get('latency_ms_median')} | {s.get('steps')} |")
    lines += ["", "## VLM 학습 (1회 학습 시간·VRAM 실측)", "",
              "| run | 모델 | 양자화 | 학습 샘플 | step | 학습 시간(분) | 최대 VRAM 할당/예약(GB) |",
              "|---|---|---|---|---|---|---|"]
    for run in ("m1_vlm2b_f0", "m1_vlm4b_bench"):
        s = _read(ROOT / "runs" / run / "final" / "summary.json")
        if s:
            mins = (s.get("train_runtime") or 0) / 60
            lines.append(f"| {run} | {Path(s['model']).name} | {s['quant']} | {s['train_samples']} | "
                         f"{s['global_step']} | {mins:.1f} | {s['peak_vram_alloc_gb']} / {s['peak_vram_reserved_gb']} |")
    s = _read(ROOT / "runs" / "m1_vlm2b_f0" / "final" / "summary.json")
    if s and s.get("val_generation"):
        g = s["val_generation"]
        lines += ["", f"## inner-val 생성 평가 (n={g['n']}, 원출력 / Guard 후)", "",
                  "| 지표 | 원출력 | Guard 후 |", "|---|---|---|"]
        for k in ("format_ok", "normal_decision_acc", "group_acc", "type_acc", "detail_exact", "location_exact"):
            lines.append(f"| {k} | {_pct(g['raw'][k])} | {_pct(g['guarded'][k])} |")
    p = _read(ROOT / "outputs" / "pipeline" / "m1_f0_heldout.summary.json")
    if p:
        lines += ["", f"## 종단 (fold 1 held-out, 이상 {p['n_anomaly']} / 정상 {p['n_normal']})", "",
                  "| 구성 | 재현율 | FPR |", "|---|---|---|",
                  f"| 탐지기 단독 | {_pct(p['detector_only']['recall'])} | {_pct(p['detector_only']['fpr'])} |",
                  f"| 탐지기 + VLM 거부 | {_pct(p['cascade']['recall'])} | {_pct(p['cascade']['fpr'])} |",
                  "", f"- 이상 이미지의 종단 그룹 정확도: {_pct(p['group_acc_on_anomalies_end_to_end'])}%",
                  f"- 출력 형식 준수: {_pct(p['format_ok'])}%",
                  f"- VLM 지연: {p['vlm_seconds_per_image']}초/장"]
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "m1_summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["m1"])
    a = ap.parse_args()
    print("저장:", {"m1": m1_tables}[a.what]())


if __name__ == "__main__":
    main()
