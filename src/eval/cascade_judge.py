"""캐스케이드 심판 교체 실험: 판정 구간의 심판을 VLM에서 B1(비-VLM 진단기)로 바꿨을 때의 RQ4 지표. 가설과 판정 기준은 결과를 보기 전에 적어 두었다.

    J1 (주): 판정 구간에서 B1 argmax가 "good"이면 거부. 매개변수 없음.
    J2 (부, 탐색): B1 p(good) >= t면 거부. t는 fold 학습 측 5-fold OOF에서 판정 구간 이상의 97%가 통과하도록 정함.
    VLM 심판(기존)과 Ours 4B seed 0·1·2 각각에서 비교. held-out 1회 평가.

필요 파일: outputs/baselines/b1_f{f}_probs.jsonl (+ .oof.jsonl) <- python -m src.baselines.crop_classifier --probs-out
출력: reports/tables/m3_cascade_b1judge.md / .json

    python -m src.eval.cascade_judge
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.cascade import PIS, cascade, compare_e2e, end_to_end, load_judge, oof_threshold, summarize
from src.kb import load_kb
from src.output_spec import is_normal
from src.report.m2_report import heldout

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs" / "baselines"
RECALL = 0.95
SEEDS = (0, 1, 2)


def pct(v, sign=False):
    return "-" if v is None else f"{100 * v:+.1f}" if sign else f"{100 * v:.1f}"


def agreement(df: pd.DataFrame) -> dict:
    """판정 구간에서 VLM과 B1의 거부 판단 일치 (정상·이상 각각): 둘 다 / VLM만 / B1만 / 둘 다 아님."""
    j = df[df["stage"] == "judge"]
    out = {}
    for name, m in (("normal", j["label"] == "good"), ("anomaly", j["label"] != "good")):
        g = j[m]
        v, b = g["vlm_normal"].to_numpy(bool), g["judge_normal"].to_numpy(bool)
        out[name] = {"n": int(len(g)), "both_reject": int((v & b).sum()), "vlm_only": int((v & ~b).sum()),
                     "b1_only": int((~v & b).sum()), "neither": int((~v & ~b).sum())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="m2_4b_f{f}_s{s}")
    ap.add_argument("--probs", default=str(BASE / "b1_f{f}_probs.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "reports" / "tables" / "m3_cascade_b1judge"))
    a = ap.parse_args()
    kb = load_kb()
    index = pd.read_parquet(ROOT / "data" / "index.parquet")
    splits = pd.read_csv(ROOT / "data" / "splits.csv")
    judge = {f: load_judge(Path(a.probs.format(f=f))) for f in (0, 1)}
    thr = {f: oof_threshold(Path(a.probs.format(f=f)).with_name(Path(a.probs.format(f=f)).stem + ".oof.jsonl"),
                            f, RECALL, index, splits) for f in (0, 1)}
    # B1 재학습 argmax와 기존 B1 예측(표1)의 정상/이상 불일치 장수
    old = pd.concat([pd.read_json(BASE / f"b1_f{f}_heldout.jsonl", lines=True) for f in (0, 1)]).set_index("id")
    new_norm = pd.concat([judge[f]["argmax"] == "good" for f in (0, 1)])
    old_norm = old["pred"].map(is_normal).reindex(new_norm.index)
    mismatch = int((new_norm != old_norm).sum())

    res = {"recall_target": RECALL, "j2_threshold": {str(f): thr[f] for f in thr},
           "b1_retrain_normal_mismatch_vs_table1": mismatch, "seeds": {}}
    for s in SEEDS:
        models = {f: heldout(a.run.format(f=f, s=s)) for f in (0, 1)}
        if not all(p.exists() for p in models.values()):
            continue
        base = cascade(models, RECALL, kb, index, splits)
        j1 = cascade(models, RECALL, kb, index, splits, judge=judge, judge_rule="argmax")
        j2 = cascade(models, RECALL, kb, index, splits, judge=judge, judge_rule="threshold",
                     judge_thr={f: thr[f]["threshold"] for f in thr})
        test_pi = float((base["label"] != "good").mean())
        pis = (test_pi, 0.01)
        r = {"test_pi": test_pi}
        for name, df in (("vlm", base), ("j1", j1), ("j2", j2)):
            r[name] = {"summary": summarize(df), "end_to_end": end_to_end(df, pis)}
            if name != "vlm":
                r[name]["vs_vlm"] = compare_e2e(df, base, pis)
                r[name]["agreement"] = agreement(df)
        res["seeds"][str(s)] = r
        print(f"seed {s}: VLM 손실 {r['vlm']['summary']['recall_loss_pp']:.1f}%p → J1 {r['j1']['summary']['recall_loss_pp']:.1f}%p, "
              f"오경보 감소 {r['vlm']['summary']['false_alarm_reduction']:.0%} → {r['j1']['summary']['false_alarm_reduction']:.0%}")

    lines = [f"# 캐스케이드 심판 교체 실험 (판정 구간 거부를 B1에 맡김, τ_low 재현율 목표 {RECALL}, 두 held-out fold 합산)", "",
             "결과를 보기 전에 적어 둔 규칙: J1 = B1 argmax가 정상이면 거부(매개변수 없음), J2 = p(good) 임계값(학습 측 out-of-fold 확률에서 "
             "판정 구간 이상 97% 통과, 탐색). 가설 J: J1은 재현율 손실 1%p 이하·오경보 감소 50% 이상(RQ4 기준)을 충족하고 "
             "종단 정답 그룹 F1이 VLM 심판 이상이다.", "",
             f"- J2 임계값 p(good) >= t: fold 0 t={thr[0]['threshold']:.3f} (OOF 판정 구간 이상 {thr[0]['n_judge_anomaly']}장), "
             f"fold 1 t={thr[1]['threshold']:.3f} ({thr[1]['n_judge_anomaly']}장)",
             f"- B1 재학습(확률용)의 정상/이상 argmax가 표1의 B1 예측과 다른 이미지: {mismatch}장 (표1은 기존 파일 그대로)", "",
             "| 4B seed | 심판 | 캐스케이드 재현율 / FPR | 재현율 손실(%p) | 오경보 감소 | RQ4 기준 | 종단 그룹 정확도 | "
             "종단 그룹 F1 (test / π=1%) | F1 차이 vs VLM 심판 [95% CI] (test / π=1%) | VLM 호출률 π=1% |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for s, r in res["seeds"].items():
        for name, ko in (("vlm", "VLM(기존)"), ("j1", "B1 argmax (J1)"), ("j2", "B1 p(good) 임계값 (J2, 탐색)")):
            sm, e = r[name]["summary"], r[name]["end_to_end"]
            c = sm["cascade"]
            pi_t = f"pi={r['test_pi']:g}"
            diff = "-"
            if name != "vlm":
                v = r[name]["vs_vlm"]
                d1, d2 = v[pi_t]["f1_group"], v["pi=0.01"]["f1_group"]
                diff = (f"{pct(d1['diff'], True)} [{pct(d1['ci95'][0], True)}, {pct(d1['ci95'][1], True)}] / "
                        f"{pct(d2['diff'], True)} [{pct(d2['ci95'][0], True)}, {pct(d2['ci95'][1], True)}]")
            lines.append(f"| {s} | {ko} | {pct(c['recall'])} / {pct(c['fpr'])} | {sm['recall_loss_pp']:.1f} | "
                         f"{sm['false_alarm_reduction']:.0%} | {'충족' if sm['rq4_hypothesis_met'] else '미충족'} | "
                         f"{pct(sm['end_to_end_group_acc'])} | {pct(e[pi_t]['f1_group'])} / {pct(e['pi=0.01']['f1_group'])} | {diff} | "
                         f"{pct(sm['vlm_call_rate']['pi=0.01'])} |")
    s0 = res["seeds"].get("0")
    if s0:
        lines += ["", "판정 구간 거부 판단 일치 (seed 0, VLM vs B1 argmax): "]
        for name, ko in (("normal", "정상"), ("anomaly", "이상")):
            g = s0["j1"]["agreement"][name]
            lines.append(f"- {ko} {g['n']}장: 둘 다 거부 {g['both_reject']}, VLM만 {g['vlm_only']}, B1만 {g['b1_only']}, 둘 다 아님 {g['neither']}")
        jb = s0["vlm"]["summary"]["judge_band"]
        jb1 = s0["j1"]["summary"]["judge_band"]
        lines.append(f"- 판정 구간 정상 거부율 {pct(jb['normal_rejected'])} → {pct(jb1['normal_rejected'])}, "
                     f"이상 오거부율 {pct(jb['anomaly_wrongly_rejected'])} → {pct(jb1['anomaly_wrongly_rejected'])}")
        # 탐색(사후 산술): 둘 다 정상이라 할 때만 거부(AND)하면 어떻게 되는지 일치표에서 계산
        sm = s0["vlm"]["summary"]
        ag = s0["j1"]["agreement"]
        n_a, n_n = sm["n_anomaly"], sm["n_normal"]
        loss_and = ag["anomaly"]["both_reject"] / n_a
        det_out_n = sm["detector_only"]["fpr"] * n_n
        fpr_and = (det_out_n - ag["normal"]["both_reject"]) / n_n
        red_and = 1 - fpr_and / sm["detector_only"]["fpr"]
        res["exploratory_and_rule_seed0"] = {"recall_loss_pp": 100 * loss_and, "fpr": fpr_and, "false_alarm_reduction": red_and,
                                             "rq4_met": loss_and * 100 <= 1.0 and red_and >= 0.5}
        lines.append(f"- 탐색(사후 비교): VLM과 B1이 둘 다 정상이라 할 때만 거부(AND)하면 seed 0 기준 재현율 손실 {100 * loss_and:.1f}%p, "
                     f"FPR {pct(fpr_and)}, 오경보 감소 {red_and:.0%} → RQ4 기준 {'충족' if res['exploratory_and_rule_seed0']['rq4_met'] else '미충족'}"
                     "(일치표의 산술값이며 새 설정으로 채택하지 않는다)")
        met = [bool(res["seeds"][k]["j1"]["summary"]["rq4_hypothesis_met"]) for k in res["seeds"]]
        ci_pos = [res["seeds"][k]["j1"]["vs_vlm"][f"pi={res['seeds'][k]['test_pi']:g}"]["f1_group"]["ci95"][0] > 0 for k in res["seeds"]]
        res["verdict"] = {"rq4_met_all_seeds": all(met), "f1_group_ci_excludes_zero_all_seeds": all(ci_pos)}
        lines += ["", f"가설 J 판정: RQ4 기준 충족 seed {sum(met)}/{len(met)}, 종단 그룹 F1 차이 95% CI가 0을 넘는 seed {sum(ci_pos)}/{len(ci_pos)}"
                  + (" → 지지" if all(met) else " → 미충족(기준 미달 seed 있음)")]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"저장: {out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
