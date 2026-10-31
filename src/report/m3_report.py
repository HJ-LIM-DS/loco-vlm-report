"""분석 단계 결과 집계: 입력 ablation(2B로 선별한 뒤 상위 2개를 4B 2-fold로 확인), RQ3 open-set, RQ4·B2 종단 비교,
사전학습 노출 점검, 실패 분석 요약, 성공 기준 점검.

- ablation 효과는 "적용 - 미적용"으로 통일한다. 입력 제거 ablation은 기본 - 제거(그 입력의 기여),
  비전 인코더 LoRA는 추가 - 기본. 양수면 그 입력·기법이 정확도를 올린다.
- 2B 선별은 fold 0 모델로 fold 1 이미지를 채점하고, Holm 보정은 7종 x 논리/구조 14개 검정을 한 묶음으로 한다.
- 가설 열은 결과를 보기 전(2026-10-24)에 적어 둔 내용만 옮긴다. 가설이 없으면 "-".
- 선행 단계 산출물(src.eval.cascade, src.eval.failure)을 읽고, 있는 결과만 집계하므로 대기열 중간에도 다시 실행할 수 있다.

    python -m src.eval.cascade && python -m src.eval.failure && python -m src.report.m3_report --with-figures

대기열 마지막 작업(m3_report)이 모든 분석 단계 학습이 끝난 뒤 이 집계를 다시 실행한다.
"""

import json
from pathlib import Path

import pandas as pd

from src.eval.explanation import evaluate, load_preds
from src.eval.failure import COUNT_GROUPS
from src.eval.openset import score as openset_score
from src.eval.stats import holm, mcnemar, paired_bootstrap
from src.output_spec import is_normal
from src.report.m2_report import correct_vector, heldout

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "tables"
BASE = ROOT / "outputs" / "baselines"

# (이름, run, 부호: +1 = 기본 - 변형(입력 제거), -1 = 변형 - 기본(기법 추가), 결과 전 가설, 가설을 판정할 부분집합)
ABLATIONS_2B = [
    ("정상 참조 제거", "m2_2b_f0_noref", 1, "RQ2: 참조는 논리 결함(개수·배치)에 기여", "logical"),
    ("crop 제거", "m2_2b_f0_nocrop", 1, "RQ2: crop은 작은 구조 결함에 기여", "structural"),
    ("규격 텍스트 제거", "m3_2b_f0_nospec", 1, "RQ2: 규격은 논리 결함(개수·배치)에 기여", "logical"),
    ("탐지기 텍스트 제거", "m3_2b_f0_nodet", 1, "-", None),
    ("정상 네거티브 제거", "m3_2b_f0_noneg", 1, "RQ4: 오탐 거부 학습(정상 거부율)", "normal"),
    ("목록 드롭아웃 제거", "m3_2b_f0_nodrop", 1, "RQ3: open-set 표에서 검증", None),
    ("비전 인코더 LoRA", "m3_2b_f0_vislora", -1, "작은 부품 개수 세기에 필요", "count"),
]
SUBSET_KO = {"logical": "논리", "structural": "구조", "normal": "정상 거부", "count": "개수 계열"}
BASE_2B = "m2_2b_f0_s0"
CONFIRM_4B = [("crop 제거", "m3_4b_f{f}_nocrop", 1), ("비전 인코더 LoRA", "m3_4b_f{f}_vislora", -1)]
BASE_4B = "m2_4b_f{f}_s0"
OPENSET = [("구조형(이물 혼입 제외)", "m3_open_struct_4b_f0:m3_open_struct_f0", "m3_open_struct_nodrop_4b_f0:m3_open_struct_nodrop_f0"),
           ("논리형(카테고리별 1그룹 제외)", "m3_open_logic_4b_f0:m3_open_logic_f0", "m3_open_logic_nodrop_4b_f0:m3_open_logic_nodrop_f0")]
TYPES = (("logical", "논리"), ("structural", "구조"))


def pct(v, sign=False):
    return "-" if v is None else (f"{100 * v:+.1f}" if sign else f"{100 * v:.1f}")


def ci(bt):
    return f"[{100 * bt['ci95'][0]:+.1f}, {100 * bt['ci95'][1]:+.1f}]"


def normal_vector(paths) -> pd.Series:
    idx = pd.read_parquet(ROOT / "data" / "index.parquet").set_index("image_id")
    pr = load_preds(paths).set_index("id")
    pr = pr[idx.loc[pr.index, "label"] == "good"]
    return pr["pred"].map(lambda t: is_normal(t or ""))


def verdict(eff: dict, p: float) -> str:
    if eff["diff"] > 0:
        return "지지" if p < 0.05 else "방향 일치, 유의하지 않음"
    return "효과 없음" if eff["diff"] == 0 else "반대 방향"


def paired_effect(base_paths, var_paths, sign: int) -> dict:
    """이상 이미지 그룹 정답(전체·논리·구조·개수 계열)과 정상 거부의 쌍대 효과 (적용 - 미적용)."""
    idx = pd.read_parquet(ROOT / "data" / "index.parquet").set_index("image_id")
    b = correct_vector(base_paths)
    v = correct_vector(var_paths).reindex(b.index)
    on, off = (b, v) if sign > 0 else (v, b)
    out = {"all": {**paired_bootstrap(on.values, off.values), **mcnemar(on.values, off.values), "n": len(b)}}
    is_count = [(c, g[0]) in COUNT_GROUPS for c, g in zip(idx.loc[b.index, "category"], idx.loc[b.index, "groups"])]
    for lab, m in [(lab, (idx.loc[b.index, "label"] == lab).to_numpy()) for lab, _ in TYPES] + [("count", is_count)]:
        m = pd.Series(m).to_numpy(bool)
        out[lab] = {**paired_bootstrap(on.values[m], off.values[m]), **mcnemar(on.values[m], off.values[m]), "n": int(m.sum())}
    nb = normal_vector(base_paths)
    nv = normal_vector(var_paths).reindex(nb.index)
    n_on, n_off = (nb, nv) if sign > 0 else (nv, nb)
    out["normal"] = {**paired_bootstrap(n_on.values, n_off.values), **mcnemar(n_on.values, n_off.values), "n": len(nb)}
    mb, mv = evaluate(base_paths)["raw"], evaluate(var_paths)["raw"]
    out["metrics"] = {k: {"base": mb[k], "variant": mv[k]} for k in ("group_acc", "type_acc", "detail_exact", "location_hit",
                                                                     "normal_correct", "format_ok")}
    return out


def ablation_section(res: dict) -> list:
    base = [heldout(BASE_2B)]
    rows = {}
    for name, run, sign, hyp, key in ABLATIONS_2B:
        if heldout(run).exists():
            rows[name] = (paired_effect(base, [heldout(run)], sign), hyp, key)
    if not rows:
        return []
    ps = {f"{n}|{lab}": r[lab]["p"] for n, (r, _, _) in rows.items() for lab, _ in TYPES}
    adj = holm(ps)
    top = sorted(rows, key=lambda n: -abs(rows[n][0]["all"]["diff"]))[:2]
    res["ablation_2b"] = {}
    lines = ["## 1. 입력 ablation 선별 (2B, fold 0 모델로 fold 1 이미지 채점, 이상 496 / 정상 287)", "",
             "효과 = 적용 - 미적용(%p). 그룹 정확도 쌍대 차이, 논리·구조 p는 Holm 보정(14개 검정). 정상 거부·개수 계열(83장)은 "
             "2차 지표(보정 없음). 가설 판정은 가설이 가리키는 부분집합의 효과로 한다.", "",
             "| 설정 | 계획 가설(결과 전) | 그룹 전체 [95% CI], p | 논리 (Holm p) | 구조 (Holm p) | 개수 계열 (p) | 정상 거부 (p) | 위치 적중 기본 → 변형 | 가설 판정 | 4B 확인 |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for n, (r, h, key) in rows.items():
        a = r["all"]
        p_key = None if key is None else (adj[f"{n}|{key}"] if key in ("logical", "structural") else r[key]["p"])
        v = "-" if key is None else f"{SUBSET_KO[key]} {pct(r[key]['diff'], True)}: {verdict(r[key], p_key)}"
        res["ablation_2b"][n] = {**r, "hypothesis": h, "hypothesis_subset": key, "verdict": v,
                                 "p_holm": {lab: adj[f"{n}|{lab}"] for lab, _ in TYPES}, "selected_for_4b": n in top}
        lines.append(
            f"| {n} | {h} | {pct(a['diff'], True)} {ci(a)}, {a['p']:.2g} | "
            + " | ".join(f"{pct(r[lab]['diff'], True)} ({adj[f'{n}|{lab}']:.2g})" for lab, _ in TYPES)
            + f" | {pct(r['count']['diff'], True)} ({r['count']['p']:.2g}) | {pct(r['normal']['diff'], True)} ({r['normal']['p']:.2g}) | "
              f"{pct(r['metrics']['location_hit']['base'])} -> {pct(r['metrics']['location_hit']['variant'])} | {v} | "
              f"{'선정' if n in top else ''} |")
    lines += ["", "- 4B 확인 선정 규칙(사전 규칙 '효과가 큰 상위 2개'): 그룹 전체 효과의 절댓값 상위 2개. "
              "정상 네거티브 제거는 정상 거부가 0%가 되는 설계상 자명한 효과라 그룹 효과 기준에서 자연히 빠진다."]
    return lines


def confirm_section(res: dict) -> list:
    base = [heldout(BASE_4B.format(f=f)) for f in (0, 1)]
    lines = ["", "## 2. 상위 2개 4B 2-fold 확인 (비교 기준 Ours 4B seed 0, 두 held-out fold 합산)", ""]
    rows = {}
    for name, run, sign in CONFIRM_4B:
        paths = [heldout(run.format(f=f)) for f in (0, 1)]
        if all(p.exists() for p in paths):
            rows[name] = paired_effect(base, paths, sign)
        else:
            lines.append(f"- {name}: 대기열 진행 중 (" + ", ".join(run.format(f=f) for f in (0, 1)) + ")")
    if rows:
        ps = {f"{n}|{lab}": r[lab]["p"] for n, r in rows.items() for lab, _ in TYPES}
        adj = holm(ps)
        res["confirm_4b"] = {n: {**r, "p_holm": {lab: adj[f"{n}|{lab}"] for lab, _ in TYPES}} for n, r in rows.items()}
        lines += ["| 설정 | 그룹 전체 [95% CI], p | 논리 [95% CI] (Holm p) | 구조 [95% CI] (Holm p) | 정상 거부 (p) | 2B 선별과 방향 일치 |",
                  "|---|---|---|---|---|---|"]
        for n, r in rows.items():
            a, sel = r["all"], res.get("ablation_2b", {}).get(n, {}).get("all", {})
            same = "-" if not sel else ("예" if (a["diff"] > 0) == (sel["diff"] > 0) else "아니오")
            lines.append(f"| {n} | {pct(a['diff'], True)} {ci(a)}, {a['p']:.2g} | "
                         + " | ".join(f"{pct(r[lab]['diff'], True)} {ci(r[lab])} ({adj[f'{n}|{lab}']:.2g})" for lab, _ in TYPES)
                         + f" | {pct(r['normal']['diff'], True)} ({r['normal']['p']:.2g}) | {same} |")
    return lines


def openset_section(res: dict) -> list:
    lines = ["", "## 3. RQ3 open-set (4B, fold 0 모델 → fold 1 이미지)", "",
             "가설: 목록 드롭아웃 없이 학습하면 미등록 결함을 기존 그룹으로 잘못 분류한다. 핵심어 일치는 사전에 정한 핵심어"
             "(이물 혼입 → \"이물\", 칸막이 누락 → \"칸막이\" 등)가 미분류 서술에 들어간 비율.", "",
             "| 설정 | 드롭아웃 | 미등록 n | 미분류 응답 | 기존 그룹 오분류 | 정상 응답 | 핵심어 일치 | 알려진 결함 과잉 미분류 | 알려진 결함 그룹 정확도 | 정상 거부 |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    res["openset"] = {}
    pend = []
    for name, drop, nodrop in OPENSET:
        got = {}
        for tag, spec in (("있음", drop), ("없음", nodrop)):
            run, sft = spec.split(":")
            if (ROOT / "runs" / run / "final" / "heldout_predictions.jsonl").exists():
                got[tag] = openset_score(run, sft)
            else:
                pend.append(run)
        for tag, r in got.items():
            lines.append(f"| {name} | {tag} | {r['n_unseen']} | {pct(r['unseen_unknown_rate'])} | {pct(r['unseen_to_known_group'])} | "
                         f"{pct(r['unseen_said_normal'])} | {pct(r['unseen_keyword_hit'])} | {pct(r['known_false_unknown'])} | "
                         f"{pct(r['known_group_acc'])} | {pct(r['normal_correct'])} |")
        if len(got) == 2:
            a = pd.DataFrame(got["있음"]["unseen_per_image"]).set_index("id")
            b = pd.DataFrame(got["없음"]["unseen_per_image"]).set_index("id").reindex(a.index)
            mc_unk = mcnemar(a["unknown"].to_numpy(), b["unknown"].to_numpy())
            mc_known = mcnemar(b["known_group"].to_numpy(), a["known_group"].to_numpy())
            # 탐색(사후 분석): 드롭아웃이 알려진 결함의 그룹 정확도를 깎는가
            ka = pd.DataFrame(got["있음"]["known_per_image"]).set_index("id")
            kb = pd.DataFrame(got["없음"]["known_per_image"]).set_index("id").reindex(ka.index)
            mc_kacc = mcnemar(kb["correct"].to_numpy(), ka["correct"].to_numpy())
            got["paired"] = {"unknown_drop_vs_nodrop": mc_unk, "known_group_nodrop_vs_drop": mc_known,
                             "known_acc_nodrop_vs_drop": mc_kacc,
                             "by_group": {g: {"drop": got["있음"]["by_group"][g]["unknown_rate"],
                                              "nodrop": got["없음"]["by_group"][g]["unknown_rate"],
                                              "n": got["있음"]["by_group"][g]["n"]} for g in got["있음"]["by_group"]}}
        res["openset"][name] = {k: ({kk: vv for kk, vv in v.items() if not kk.endswith("_per_image")} if k != "paired" else v)
                                for k, v in got.items()}
    for name, r in res["openset"].items():
        if "paired" in r:
            p = r["paired"]
            lines.append(f"- {name}: 미분류 응답 드롭아웃 있음만 {p['unknown_drop_vs_nodrop']['n10']}장 / 없음만 "
                         f"{p['unknown_drop_vs_nodrop']['n01']}장 (McNemar p={p['unknown_drop_vs_nodrop']['p']:.2g}). 그룹별 미분류 응답(있음/없음): "
                         + ", ".join(f"{g} {pct(v['drop'])}/{pct(v['nodrop'])} (n={v['n']})" for g, v in p["by_group"].items()))
            k = p["known_acc_nodrop_vs_drop"]
            lines.append(f"  - 탐색(사후 분석): 알려진 결함 그룹 정답이 드롭아웃 없음만 {k['n10']}장 / 있음만 {k['n01']}장 "
                         f"(McNemar p={k['p']:.2g})")
    if pend:
        lines.append("- 대기열 진행 중: " + ", ".join(pend))
    return lines


def cascade_section(res: dict) -> list:
    lines = ["", "## 4. RQ4 캐스케이드와 B2 VLM 단독 (tau_low 학습 측 재현율 목표 0.95, 두 fold 합산)", "",
             "| 4B seed | 탐지기 단독 재현율 / FPR | 캐스케이드 재현율 / FPR | 재현율 손실(%p) | 오경보 감소 | 종단 탐지 F1 Ours / B2 (차이 [95% CI]) | 종단 그룹 F1 Ours / B2 (차이 [95% CI]) | π=1% 그룹 F1 Ours / B2 |",
             "|---|---|---|---|---|---|---|---|"]
    res["cascade"] = {}
    for s, suf in ((0, ""), (1, "_s1"), (2, "_s2")):
        p = OUT / f"m3_cascade{suf}.json"
        if not p.exists():
            continue
        c = json.loads(p.read_text(encoding="utf-8"))
        m = c["by_recall_target"]["0.95"]
        e = c.get("end_to_end", {})
        res["cascade"][s] = {"main": m, "end_to_end": e, "decomposition": c.get("decomposition")}
        if "ours_minus_b2" not in e:
            continue
        tkey = next(k for k in e["ours_minus_b2"] if k not in ("pi=0.01", "pi=0.05"))
        d, d1 = e["ours_minus_b2"][tkey], e["ours_minus_b2"]["pi=0.01"]
        lines.append(f"| {s} | {pct(m['detector_only']['recall'])} / {pct(m['detector_only']['fpr'])} | {pct(m['cascade']['recall'])} / "
                     f"{pct(m['cascade']['fpr'])} | {m['recall_loss_pp']:.1f} | {100 * m['false_alarm_reduction']:.0f}% | "
                     f"{pct(d['f1_detect']['a'])} / {pct(d['f1_detect']['b'])} ({pct(d['f1_detect']['diff'], True)} {ci(d['f1_detect'])}) | "
                     f"{pct(d['f1_group']['a'])} / {pct(d['f1_group']['b'])} ({pct(d['f1_group']['diff'], True)} {ci(d['f1_group'])}) | "
                     f"{pct(d1['f1_group']['a'])} / {pct(d1['f1_group']['b'])} |")
    b2 = [heldout(f"m3_b2_4b_f{f}") for f in (0, 1)]
    if all(p.exists() for p in b2):
        mb = evaluate(b2)
        res["b2_explanation"] = mb
        lines += ["", f"- B2 VLM 단독(탐지기·crop 없음) 설명 품질(이상 이미지, 원출력): 그룹 {pct(mb['raw']['group_acc'])}, "
                      f"유형 {pct(mb['raw']['type_acc'])}, 세부 {pct(mb['raw']['detail_exact'])}, 이상을 정상이라 답함 "
                      f"{pct(mb['raw']['normal_miss'])}, 정상 거부 {pct(mb['raw']['normal_correct'])}"]
    if 0 in res["cascade"] and res["cascade"][0]["decomposition"]:
        dc = res["cascade"][0]["decomposition"]
        lines += ["- 오류 분해(seed 0, 이상 993장): " + ", ".join(f"{k} {v}" for k, v in dc["anomaly"].items())
                  + f". 탐지기 놓침 중 VLM 원출력 정답 {dc['anomaly_missed_but_vlm_raw_correct']}장",
                  "- 정상 575장: " + ", ".join(f"{k} {v}" for k, v in dc["normal"].items()),
                  "- 세부 표(백분위 구간별 거부율 포함): `reports/tables/m3_cascade.md`"]
    return lines


def exposure_section(res: dict) -> list:
    rows = []
    for mode in ("zero", "few"):
        for f in (0, 1):
            p = BASE / f"b3_{mode}_f{f}" / "meta.json"
            if p.exists():
                m = json.loads(p.read_text(encoding="utf-8"))
                rows.append((mode, f, m.get("n"), m.get("english_defect_name_rate")))
    res["exposure"] = [{"mode": m, "fold": f, "n": n, "english_defect_name_rate": r} for m, f, n, r in rows]
    if not rows:
        return []
    return ["", "## 5. 사전학습 노출 점검 (B3 4B 무학습 출력에 영문 LOCO 결함명이 그대로 나오는 비율)", "",
            "| 프롬프트 | fold | 이미지 수 | 영문 결함명 비율 |", "|---|---|---|---|"]\
        + [f"| {m} | {f} | {n} | {pct(r)} |" for m, f, n, r in rows]\
        + ["", "- 한계: 프롬프트가 한국어 출력을 요구하므로 영문 이름이 나오지 않는 것은 노출이 없다는 증거로는 약하다."]


def failure_section(res: dict) -> list:
    p = OUT / "m3_failure.json"
    if not p.exists():
        return []
    f = json.loads(p.read_text(encoding="utf-8"))
    res["failure"] = {k: f[k] for k in ("by_kind", "count_errors_s0", "ablation_2b_fold1")}
    k = f["by_kind"]
    return ["", "## 6. 실패 분석 요약 (세부: `reports/tables/m3_failure.md`)", "",
            "| 결함 종류 | n | Ours 4B(seed 평균) | B1 | 차이(%p) |", "|---|---|---|---|---|"]\
        + [f"| {n} | {v['n']} | {pct(v['ours'])} | {pct(v['b1'])} | {pct(v['ours'] - v['b1'], True)} |" for n, v in k.items()]


def criteria_section(res: dict) -> list:
    m2 = json.loads((OUT / "m2_summary.json").read_text(encoding="utf-8")) if (OUT / "m2_summary.json").exists() else {}
    o = m2.get("Ours 4B s0", {}).get("raw", {})
    rows = [("결함 그룹 Top-1 85% 이상 (oracle 게이팅)", pct(o.get("group_acc")), o.get("group_acc", 0) >= 0.85),
            ("논리/구조 구분 90% 이상", pct(o.get("type_acc")), o.get("type_acc", 0) >= 0.90),
            ("형식 준수 99% 이상", pct(o.get("format_ok")), o.get("format_ok", 0) >= 0.99)]
    zs = m2.get("pairwise", {}).get("B3 zero-shot(엄격)")
    if zs:
        rows.append(("zero-shot 대비 유의한 개선 (McNemar p<0.05)", f"{pct(zs['diff'], True)}%p, p={zs['p']:.2g}", zs["p"] < 0.05 and zs["diff"] > 0))
    c0 = res.get("cascade", {}).get(0)
    if c0 and "ours_minus_b2" in c0["end_to_end"]:
        d = c0["end_to_end"]["ours_minus_b2"]
        tkey = next(k for k in d if k not in ("pi=0.01", "pi=0.05"))
        ok = all(v["f1_group"]["ci95"][0] >= 0 or v["f1_group"]["diff"] >= 0 for v in d.values())
        rows.append(("종단 F1 Ours >= B2 (π = test, 1%, 5%)", f"그룹 F1 차이 {pct(d[tkey]['f1_group']['diff'], True)}%p (test 비율)", ok))
        call = c0["main"]["vlm_call_rate"]["pi=0.01"]
        rows.append(("π=1%에서 VLM 호출률 10% 이하", f"{pct(call)}%", call <= 0.10))
        rl = c0["main"]["recall_loss_pp"]
        rows.append(("RQ4: 재현율 손실 1%p 이하로 오경보 절반 이상 감소", f"손실 {rl:.1f}%p, 감소 {100 * c0['main']['false_alarm_reduction']:.0f}%",
                     c0["main"]["rq4_hypothesis_met"]))
    res["criteria"] = [{"criterion": a, "value": b, "met": bool(c)} for a, b, c in rows]
    return ["", "## 7. 성공 기준·가설 점검 (Ours 4B seed 0)", "", "| 기준 | 값 | 충족 |", "|---|---|---|"]\
        + [f"| {a} | {b} | {'예' if c else '아니오'} |" for a, b, c in rows]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-figures", action="store_true", help="그림3(ablation 효과)도 다시 그린다")
    a = ap.parse_args()
    res = {}
    lines = ["# 분석 단계 결과 요약", ""]
    for sec in (ablation_section, confirm_section, openset_section, cascade_section, exposure_section, failure_section,
                criteria_section):
        lines += sec(res)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "m3_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "m3_summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"저장: {OUT / 'm3_summary.md'}")
    if a.with_figures:
        from src.report.figures import fig_ablation
        from src.report.style import setup
        setup()
        print(f"저장: {fig_ablation()}")


if __name__ == "__main__":
    main()
