"""PDF 포트폴리오 빌드: reports/portfolio/portfolio.html.j2 + 결과 JSON -> HTML -> Edge 헤드리스 인쇄 -> PDF.

- 원고의 모든 수치는 reports/tables의 결과 파일에서 채운다(계획 v5 검증 5: PDF 수치 = reports/tables).
  템플릿에 직접 쓰는 숫자는 설정 상수(r=16, 256토큰, 하드 네거티브 150장 등)뿐이다.
- 결과가 아직 없는 절(P2, 4B 확인)은 "진행 중" 문장으로 대신한다.
- pandoc·LaTeX가 없고 weasyprint는 GTK 라이브러리가 없어, Windows 기본 Edge의 --print-to-pdf로 인쇄한다(새 설치 없음).
- 4쪽을 넘으면 경고하고 종료 코드 1.

    python -m src.report.build_portfolio            # reports/portfolio.pdf
"""

import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import jinja2

from ops.envctl import idle_metrics

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "reports" / "tables"
SRC = ROOT / "reports" / "portfolio" / "portfolio.html.j2"
HTML = ROOT / "reports" / "portfolio" / "portfolio.html"
PDF = ROOT / "reports" / "portfolio.pdf"
EDGE = [Path("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"),
        Path("C:/Program Files/Microsoft/Edge/Application/msedge.exe")]
MAX_PAGES = 4
CATS = ["breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors"]
# 표1에 보일 시스템과 표시 이름
T1_ROWS = [("B0 최빈값", "B0 최빈값"), ("B1 비-VLM", "B1 비-VLM 진단기"), ("B2 VLM 단독", "B2 VLM 단독(4B)"),
           ("B3 zero-shot (엄격)", "B3 zero-shot(4B)"), ("B3 few-shot (엄격)", "B3 few-shot(4B)"),
           ("Ours 2B", "Ours 2B LoRA"), ("Ours 4B", "Ours 4B QLoRA"), ("P2: 8B", "P2: 8B QLoRA")]


def load(name: str):
    p = TABLES / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


# --- 숫자 형식 (템플릿 필터) ---
def pct(v, d=1):
    return "-" if v is None else f"{100 * v:.{d}f}"


def pfmt(p):
    return "p<0.001" if p < 0.001 else f"p={p:.2g}"


def ci(d):
    lo, hi = d["ci95"]
    return f"[{100 * lo:+.1f}, {100 * hi:+.1f}]"


def ci_neg(d):
    """b - a 방향으로 뒤집은 CI (기준선이 앞선 크기를 양수로 쓸 때)."""
    lo, hi = d["ci95"]
    return f"[{-100 * hi:.1f}, {-100 * lo:.1f}]"


def idle_gaps(events: list) -> list:
    """유휴 구간 각각의 길이(분). 합계 규칙은 ops.envctl.idle_metrics와 같다."""
    gaps, start = [], None
    for ev in events:
        k = ev["kind"]
        if k in ("run_done", "run_failed", "run_retry", "heartbeat_timeout", "run_paused"):
            start = ev["ts"]
        elif k == "run_start" and start is not None:
            gaps.append((ev["ts"] - start) / 60)
            start = None
        elif k == "queue_finished":
            start = None
    return gaps


def ops_metrics() -> dict:
    """보관된 대기열 이벤트 로그 전체의 실행 시간, 유휴 시간, 가장 긴 유휴 구간."""
    hours = idle = 0.0
    gaps = []
    for p in sorted((ROOT / "reports" / "ops").glob("events_*.jsonl")):
        ev = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
        if ev:
            hours += (ev[-1]["ts"] - ev[0]["ts"]) / 3600
            idle += idle_metrics(ev)["idle_minutes"]
            gaps += idle_gaps(ev)
    return {"hours": hours, "idle": idle, "largest_gap": max(gaps, default=0.0)}


def context() -> dict:
    t1, m2, m3 = load("table1_main.json"), load("m2_summary.json"), load("m3_summary.json")
    cas, fail, p2 = load("m3_cascade.json"), load("m3_failure.json"), load("p2_summary.json")
    judge = load("m3_cascade_b1judge.json")  # 2차 개선 2: 판정 구간 심판을 B1로 바꾼 캐스케이드 (없으면 문단 생략)
    judge_key, judge_met = None, 0
    if judge and judge.get("seeds"):
        judge_key = next(k for k in judge["seeds"]["0"]["j1"]["vs_vlm"] if k != "pi=0.01")
        judge_met = sum(bool(s["j1"]["summary"]["rq4_hypothesis_met"]) for s in judge["seeds"].values())
    casm = cas["by_recall_target"]["0.95"]
    n_anom, n_norm = casm["n_anomaly"], casm["n_normal"]
    test_key = next(k for k in cas["end_to_end"]["ours_minus_b2"] if k not in ("pi=0.01", "pi=0.05"))
    seeds = []
    for suf in ("", "_s1", "_s2"):
        c = load(f"m3_cascade{suf}.json")
        if c and "ours_minus_b2" in c.get("end_to_end", {}):
            seeds.append(f"{100 * c['end_to_end']['ours_minus_b2'][test_key]['f1_group']['diff']:+.1f}")
    dc = cas["decomposition"]
    a_items, n_items = list(dc["anomaly"].items()), list(dc["normal"].items())
    k = max(len(a_items), len(n_items))
    pad = lambda xs: xs + [("", None)] * (k - len(xs))  # noqa: E731
    correct = dc["anomaly"]["정답"]
    n_err = n_anom - correct
    vlm_group = dc["anomaly"]["그룹 오답(판정 구간)"] + dc["anomaly"]["그룹 오답(설명 구간)"]
    crop = {r["bin"]: r["개수 계열"] for r in fail["crop"]}
    det = [json.loads((ROOT / "outputs" / "detector" / "salad" / c / "meta.json").read_text(encoding="utf-8"))
           for c in CATS]
    git_name = subprocess.run(["git", "config", "user.name"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    rel = lambda name: f"../figures/{name}"  # noqa: E731
    return {
        "author": git_name or "-", "date": time.strftime("%Y-%m-%d"), "build_time": time.strftime("%Y-%m-%d %H:%M"),
        "t1": t1, "t1_rows": [(disp, t1[key]) for key, disp in T1_ROWS if key in t1],
        "ours": t1["Ours 4B"], "b1": t1["B1 비-VLM"], "m2": m2, "m3": {**m3, "confirm_4b": m3.get("confirm_4b")},
        "b1_key": "B1 비-VLM", "zs_key": "B3 zero-shot(엄격)",
        "zs_acc": t1["B3 zero-shot (엄격)"]["metrics"]["group_acc"], "zs_fmt": t1["B3 zero-shot (엄격)"]["metrics"]["format_ok"],
        "b1_logic": m2["B1 비-VLM"]["raw"]["by_type"]["logical"]["group_acc"],
        "ours_logic": m2["Ours 4B s0"]["raw"]["by_type"]["logical"]["group_acc"],
        "mmad": ", ".join(f"{pct(v['base'])} → {pct(v['lora'])}" for v in m2["mmad"].values()),
        "e2e_b2": cas["end_to_end"]["ours_minus_b2"][test_key]["f1_group"], "e2e_seeds": seeds,
        "abl": m3["ablation_2b"], "cas": cas, "casm": casm, "test_pi": n_anom / (n_anom + n_norm),
        "decomp_rows": list(zip(pad(a_items), pad(n_items))), "n_anom": n_anom, "n_norm": n_norm, "n_err": n_err,
        "vlm_err_share": vlm_group / n_err, "det_err_share": dc["anomaly"]["탐지기 놓침(통과 구간)"] / n_err,
        "fk": fail["by_kind"], "gap0": fail["by_group"][0], "gap1": fail["by_group"][1],
        "conf0": fail["confusions_s0"][0], "conf1": fail["confusions_s0"][1],
        "cover_full": crop["100%"]["ours"], "cover_low": crop["<50%"]["ours"],
        "det": {"mean": sum(d["official_auroc"]["auc"] for d in det) / len(det),
                "latency": t1["detector"]["latency_ms_median"]},
        "p2": p2, "ops": ops_metrics(),
        "judge": judge if judge and judge.get("seeds") else None, "judge_key": judge_key, "judge_met": judge_met,
        "figs": {"system": rel("system.png"), "pi": rel("pi_call_rate.png"), "ablation": rel("ablation_effects.png"),
                 "cases": rel("cases.png"), "curve": rel("label_curve.png")},
    }


def render() -> Path:
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(SRC.parent), undefined=jinja2.StrictUndefined,
                             autoescape=False)
    env.filters.update({
        "pct": pct, "pct0": lambda v: pct(v, 0), "spct": lambda v: f"{100 * v:+.1f}", "pfmt": pfmt, "ci": ci,
        "ci_neg": ci_neg, "p2": lambda p: f"{p:.2g}", "f0": lambda v: f"{v:.0f}", "f1": lambda v: f"{v:.1f}",
        "f2": lambda v: f"{v:.2f}"})
    HTML.write_text(env.get_template(SRC.name).render(**context()), encoding="utf-8")
    return HTML


def print_pdf(html: Path, pdf: Path) -> None:
    edge = next((p for p in EDGE if p.exists()), None)
    if edge is None:
        sys.exit("Edge를 찾지 못함")
    # 열려 있는 Edge와 섞이지 않도록 별도 프로필. 출력 경로에 공백이 있으면 Edge가 파일을 쓰지 못해
    # (프로젝트 폴더 이름에 공백) 임시 폴더에 인쇄한 뒤 옮긴다.
    tmp = Path(tempfile.mkdtemp(prefix="loco_pdf_"))
    try:
        out = tmp / "out.pdf"
        subprocess.run([str(edge), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                        f"--user-data-dir={tmp / 'profile'}", f"--print-to-pdf={out}", html.resolve().as_uri()],
                       check=True, timeout=120, capture_output=True)
        # Edge는 종료 코드를 돌려준 뒤에 파일을 쓰기도 한다: 파일이 생기고 크기가 멈출 때까지 기다린다
        size, t0 = -1, time.time()
        while time.time() - t0 < 60:
            cur = out.stat().st_size if out.exists() else -1
            if cur > 0 and cur == size:
                break
            size = cur
            time.sleep(1)
        else:
            sys.exit("Edge가 PDF를 만들지 못함")
        shutil.move(str(out), pdf)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    html = render()
    print_pdf(html, PDF)
    import pymupdf
    n = pymupdf.open(PDF).page_count
    print(f"저장: {PDF} ({n}쪽)")
    if n > MAX_PAGES:
        print(f"경고: {MAX_PAGES}쪽 초과")
        sys.exit(1)


if __name__ == "__main__":
    main()
