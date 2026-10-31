"""추가 실험 2(입력 표현) 결과 집계. 가설과 판정 기준은 결과를 보기 전에 적어 두었다.

1단계 선별: 2B fold 0 모델로 fold 1 이미지를 채점. 기준 m2_2b_f0_s0 대비 변형의 쌍대 효과(적용 - 미적용).
  변형 A(구성 신호): 탐지기 줄에 SALAD 세 항 백분위. 가설 A: 기타 논리에서 효과 > 0 (+5%p 이상이면 지지).
  변형 B(B1 후보 힌트): 상위 2그룹 후보. 가설 B: B1과의 격차(fold 1 이미지)를 절반 이상 줄인다.
  부가 지표: 변형 B의 힌트 1위 복사율, 힌트 1위가 틀렸을 때/맞았을 때 정답률.
  4B 확인 대상: 그룹 전체 효과가 양수이고 가장 큰 변형 1개 (둘 다 0 이하이면 생략).
2단계 확인: 4B 2-fold(기준 Ours 4B seed 0). 그룹 전체 쌍대 차이 95% CI가 0을 제외하고 양수면 재현.
있는 결과만 집계하므로 대기열 중간에도 다시 실행할 수 있다.

    python -m src.report.p3_report
"""

import json
from pathlib import Path

import pandas as pd

from src.eval.explanation import evaluate, load_preds
from src.eval.stats import holm, mcnemar, paired_bootstrap
from src.output_spec import NORMAL, is_normal, parse_report
from src.report.m2_report import correct_vector, heldout
from src.report.m3_report import normal_vector
from src.report.p2_report import KINDS, kinds

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "tables"
BASE = ROOT / "outputs" / "baselines"
BASE_2B = "m2_2b_f0_s0"
BASE_4B = "m2_4b_f{f}_s0"
B1_F1 = BASE / "b1_f0_heldout.jsonl"  # fold 0 모델의 held-out = fold 1 이미지
B1_BOTH = [BASE / f"b1_f{f}_heldout.jsonl" for f in (0, 1)]  # 4B 확인의 탐색적 B1 대비(두 fold 합산)
# (표시 이름, 키, 2B 선별 run, 4B 확인 run 형식, SFT 이름)
VARIANTS = [("구성 신호 (A)", "terms", "p3_2b_f0_terms", "p3_4b_f{f}_terms", "p3_terms"),
            ("B1 후보 힌트 (B)", "hints", "p3_2b_f0_hints", "p3_4b_f{f}_hints", "p3_hints")]
H_A_MIN = 0.05  # 가설 A 지지 기준: 기타 논리 +5%p 이상


def pct(v, sign=False):
    return "-" if v is None else (f"{100 * v:+.1f}" if sign else f"{100 * v:.1f}")


def ci(bt):
    return f"[{100 * bt['ci95'][0]:+.1f}, {100 * bt['ci95'][1]:+.1f}]"


def effect(base_paths: list, var_paths: list) -> dict:
    """변형 - 기준: 그룹 정답(전체·결함 종류별)과 정상 거부의 쌍대 효과, 원출력 지표."""
    b = correct_vector(base_paths)
    v = correct_vector(var_paths).reindex(b.index)
    if v.isna().any():
        raise ValueError("변형 예측에 기준 이미지가 빠져 있음")
    out = {"all": {**paired_bootstrap(v.values, b.values), **mcnemar(v.values, b.values), "n": int(len(b)),
                   "base_acc": float(b.mean()), "variant_acc": float(v.mean())}}
    k = kinds(b.index)
    for kd in KINDS:
        m = (k == kd).to_numpy()
        out[kd] = {**paired_bootstrap(v.values[m], b.values[m]), **mcnemar(v.values[m], b.values[m]), "n": int(m.sum()),
                   "base_acc": float(b.values[m].mean()), "variant_acc": float(v.values[m].mean())}
    nb = normal_vector(base_paths)
    nv = normal_vector(var_paths).reindex(nb.index)
    out["normal"] = {**paired_bootstrap(nv.values, nb.values), **mcnemar(nv.values, nb.values), "n": int(len(nb))}
    mb, mv = evaluate(base_paths)["raw"], evaluate(var_paths)["raw"]
    out["metrics"] = {m: {"base": mb[m], "variant": mv[m]} for m in ("group_acc", "type_acc", "detail_exact", "location_hit",
                                                                     "normal_correct", "format_ok")}
    return out


def hint_rows(sft_dir: Path, pred_path: Path) -> list:
    """held-out 이상 이미지별 (힌트 1위 복사 여부, 힌트 1·2위 정오, 출력 정오, 출력 = 2위, 정상 출력) 행."""
    recs = {}
    src = sft_dir / "heldout.jsonl"
    if not src.exists():  # 공개 사본(reports/predictions)에서 복원한 환경: 힌트 필드만 있는 슬림 기록
        src = sft_dir / "heldout_hints.jsonl"
    for line in src.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            recs[r["id"]] = r
    pr = load_preds([pred_path]).set_index("id")["pred"]
    rows = []
    for iid, rec in recs.items():
        if rec["label"] == "good" or not rec.get("hints") or iid not in pr.index:
            continue
        hints = rec["hints"]
        hint1, hint2 = hints[0][0], (hints[1][0] if len(hints) > 1 else None)
        gt = parse_report(rec["gt_report"]).group
        p = pr.loc[iid] or ""
        out_group = NORMAL if is_normal(p) else (parse_report(p).group if parse_report(p) else None)
        rows.append({"copy": out_group == hint1, "hint_right": hint1 == gt, "hint2_right": hint2 == gt,
                     "out_hint2": out_group == hint2, "correct": out_group == gt, "out_normal": out_group == NORMAL})
    return rows


def hint_diagnostics(pairs: list) -> dict:
    """변형 B 복사 여부(결과 전에 정한 지표): 힌트 1위와 출력 그룹의 일치율, 힌트 1위 정오별 정답률.
    힌트 1위가 틀린 이미지는 (2위로 맞춤, 힌트 밖으로 맞춤, 틀린 1위 복사, 정상 출력=놓침)으로 나눈다. pairs = [(SFT 디렉터리, 예측)]."""
    df = pd.DataFrame([r for sft_dir, pred_path in pairs for r in hint_rows(sft_dir, pred_path)])
    if df.empty:
        return {}
    hr = df["hint_right"]
    w = df.loc[~hr]
    return {"n": int(len(df)), "copy_rate": float(df["copy"].mean()), "hint1_acc": float(hr.mean()),
            "acc_when_hint_right": float(df.loc[hr, "correct"].mean()) if hr.any() else None,
            "acc_when_hint_wrong": float(w["correct"].mean()) if len(w) else None,
            "copy_when_hint_wrong": float(w["copy"].mean()) if len(w) else None,
            "n_hint_wrong": int(len(w)),
            "hint2_right_when_wrong": float(w["hint2_right"].mean()) if len(w) else None,
            "correct_via_hint2_when_wrong": float((w["correct"] & w["out_hint2"]).mean()) if len(w) else None,
            "correct_outside_hints_when_wrong": float((w["correct"] & ~w["out_hint2"]).mean()) if len(w) else None,
            "missed_when_wrong": float(w["out_normal"].mean()) if len(w) else None}


def diag_line(label: str, d: dict) -> str:
    return (f"- {label}(이상 {d['n']}장): 출력 그룹 = 힌트 1위 {pct(d['copy_rate'])}%, 힌트 1위 정확도 {pct(d['hint1_acc'])}%, "
            f"힌트가 맞을 때 정답률 {pct(d['acc_when_hint_right'])}%, 틀릴 때 {pct(d['acc_when_hint_wrong'])}% "
            f"(틀린 힌트를 그대로 복사 {pct(d['copy_when_hint_wrong'])}%). 힌트 1위가 틀린 {d['n_hint_wrong']}장의 분해: "
            f"2위로 맞춤 {pct(d['correct_via_hint2_when_wrong'])}%(2위가 정답인 비율 {pct(d['hint2_right_when_wrong'])}%), "
            f"힌트 밖으로 맞춤 {pct(d['correct_outside_hints_when_wrong'])}%, 정상으로 출력(놓침) {pct(d['missed_when_wrong'])}%")


def screening_section(res: dict) -> list:
    base = [heldout(BASE_2B)]
    rows = {key: effect(base, [heldout(run)]) for _, key, run, _, _ in VARIANTS if heldout(run).exists()}
    if not rows:
        return ["## 1. 2B 선별 (대기열 진행 중: 결과 없음)"]
    adj = holm({f"{k}|{kd}": r[kd]["p"] for k, r in rows.items() for kd in KINDS})
    b1 = correct_vector([B1_F1])
    base_acc = rows[next(iter(rows))]["all"]["base_acc"]
    b1_acc = float(b1.reindex(correct_vector(base).index).mean())
    gap_base = b1_acc - base_acc
    res["screening"] = {"base_run": BASE_2B, "b1_acc_fold1": b1_acc, "base_acc": base_acc, "gap_base": gap_base, "variants": {}}
    lines = ["## 1. 선별 (2B, fold 0 모델로 fold 1 이미지 채점, 기준 m2_2b_f0_s0)", "",
             f"효과 = 변형 - 기준(%p). 결함 종류별 p는 Holm 보정({len(adj)}개 검정). 기준 그룹 정확도 {pct(base_acc)}, "
             f"같은 이미지에서 B1 {pct(b1_acc)} (격차 {pct(gap_base)}%p).", "",
             "| 변형 | 그룹 전체 [95% CI], p | 개수 계열 (Holm p) | 기타 논리 (Holm p) | 구조 (Holm p) | 정상 거부 (p) | "
             "B1 격차 | 가설 판정 | 4B 확인 |", "|---|---|---|---|---|---|---|---|---|"]
    pos = {k: r["all"]["diff"] for k, r in rows.items() if r["all"]["diff"] > 0}
    selected = max(pos, key=pos.get) if pos else None
    for name, key, run, _, sft in VARIANTS:
        if key not in rows:
            continue
        r = rows[key]
        gap = b1_acc - r["all"]["variant_acc"]
        if key == "terms":
            e, p = r["기타 논리"]["diff"], adj[f"{key}|기타 논리"]
            v = ("지지" if e >= H_A_MIN and p < 0.05 else "방향 일치, 기준(+5%p·p<0.05) 미달" if e > 0
                 else "효과 없음" if e == 0 else "반대 방향")
            verdict = f"가설 A 기타 논리 {pct(e, True)}: {v}"
        else:
            v = "지지" if gap <= gap_base / 2 else ("격차 축소" if gap < gap_base else "격차 유지·확대")
            verdict = f"가설 B 격차 {pct(gap_base)}에서 {pct(gap)}: {v}"
        diag = {}
        if key == "hints" and (ROOT / "data" / "sft" / f"{sft}_f0").exists():
            diag = hint_diagnostics([(ROOT / "data" / "sft" / f"{sft}_f0", heldout(run))])
        res["screening"]["variants"][key] = {**r, "p_holm": {kd: adj[f"{key}|{kd}"] for kd in KINDS}, "gap_b1": gap,
                                             "verdict": verdict, "selected_for_4b": key == selected, "hint_diagnostics": diag}
        a = r["all"]
        lines.append(f"| {name} | {pct(a['diff'], True)} {ci(a)}, {a['p']:.2g} | "
                     + " | ".join(f"{pct(r[kd]['diff'], True)} ({adj[f'{key}|{kd}']:.2g})" for kd in KINDS)
                     + f" | {pct(r['normal']['diff'], True)} ({r['normal']['p']:.2g}) | {pct(gap)}%p | {verdict} | "
                     f"{'선정' if key == selected else ''} |")
    lines += ["", "- 4B 확인 선정 규칙(사전): 그룹 전체 효과가 양수인 변형 중 가장 큰 1개."
              + ("" if selected else " 둘 다 0 이하라 4B 확인을 생략한다.")]
    d = res["screening"]["variants"].get("hints", {}).get("hint_diagnostics")
    if d:
        lines.append(diag_line("변형 B 복사 진단", d))
    res["screening"]["selected"] = selected
    return lines


def confirm_section(res: dict) -> list:
    base = [heldout(BASE_4B.format(f=f)) for f in (0, 1)]
    lines, extra = [], []
    for name, key, _, fmt, sft in VARIANTS:
        paths = [heldout(fmt.format(f=f)) for f in (0, 1)]
        if not all(p.exists() for p in paths):
            continue
        r = effect(base, paths)
        adj = holm({kd: r[kd]["p"] for kd in KINDS})
        rep = r["all"]["ci95"][0] > 0
        res.setdefault("confirm_4b", {})[key] = {**r, "p_holm": adj, "reproduced": rep}
        if key == "hints":
            # 복사 진단을 4B 확인에도 같은 정의로 적용(두 fold 합산). B1 대비는 결과 전에 정한 비교가 아닌 사후 수치.
            sft_dirs = [ROOT / "data" / "sft" / f"{sft}_f{f}" for f in (0, 1)]
            if all(d.exists() for d in sft_dirs):
                d = hint_diagnostics(list(zip(sft_dirs, paths)))
                res["confirm_4b"][key]["hint_diagnostics"] = d
                extra.append(diag_line("4B 복사 진단(두 fold 합산)", d))
            if all(p.exists() for p in B1_BOTH):
                vb = effect(B1_BOTH, paths)
                res["confirm_4b"][key]["vs_b1"] = vb
                a = vb["all"]
                extra.append(f"- 사후 비교(결과 전에 정한 비교 아님) B1 대비: 그룹 {pct(a['variant_acc'])} vs {pct(a['base_acc'])}, "
                             f"{pct(a['diff'], True)} {ci(a)} p={a['p']:.2g}; "
                             + ", ".join(f"{kd} {pct(vb[kd]['diff'], True)}" for kd in KINDS)
                             + f"; 정상 거부 {pct(vb['normal']['diff'], True)} (p={vb['normal']['p']:.2g}). "
                             "힌트는 B1의 그룹 합산 확률이므로 이 차이는 B1 판정을 VLM이 고친 몫이다.")
        if not lines:
            lines = ["", "## 2. 4B 2-fold 확인 (기준 Ours 4B seed 0, 두 held-out fold 합산)", "",
                     "| 변형 | 그룹 전체 [95% CI], p | 개수 계열 (Holm p) | 기타 논리 (Holm p) | 구조 (Holm p) | 정상 거부 (p) | 판정 |",
                     "|---|---|---|---|---|---|---|"]
        a = r["all"]
        lines.append(f"| {name} | {pct(a['diff'], True)} {ci(a)}, {a['p']:.2g} | "
                     + " | ".join(f"{pct(r[kd]['diff'], True)} ({adj[kd]:.2g})" for kd in KINDS)
                     + f" | {pct(r['normal']['diff'], True)} ({r['normal']['p']:.2g}) | "
                     f"{'재현(CI가 0을 제외하고 양수)' if rep else '재현 안 됨'} |")
    if extra:
        lines += [""] + extra
    return lines or ["", "## 2. 4B 2-fold 확인 (아직 결과 없음)"]


def main():
    res = {}
    lines = ["# 추가 실험 2 결과 요약: 입력 표현 (가설·판정 기준은 docs/preregistration.md 6절)", ""]
    lines += screening_section(res)
    lines += confirm_section(res)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "p3_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "p3_summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print("\n".join(lines))
    print(f"저장: {OUT / 'p3_summary.md'}")


if __name__ == "__main__":
    main()
