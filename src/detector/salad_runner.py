"""SALAD(ICCV 2025) 사전학습 가중치로 이미지별 점수 3항과 맵을 저장한다 (SALAD 전용 가상환경에서 실행).

공식 test 스크립트(external/SALAD/test_salad.py)는 AUROC 4종만 출력하고 이미지별 점수·맵을 남기지 않는다.
여기서는 공식 코드의 정규화·추론 함수를 그대로 불러 같은 프로세스·같은 정규화 값으로
1) 공식 test()를 실행해 AUROC 4종을 받고, 2) 같은 계산을 이미지별로 반복해 저장한 뒤,
3) 저장값으로 다시 계산한 AUROC 4종이 1)과 같은지 확인한다 (다르면 실패 처리).

이미지 점수 = 국소 맵 최댓값 + Mahalanobis 점수 + 합성 맵 최댓값 (train_salad.py test()와 동일).
AUROC는 공식과 같이 논리·구조 각각을 정상과 비교한 값의 평균이다.

저장 (src.detector.cache 형식)
    outputs/detector/salad/<category>/scores.csv   image_id, score, s_local, s_mah, s_comp
    outputs/detector/salad/<category>/maps/        국소 맵 + 합성 맵(256x256, crop용)
    outputs/detector/salad/<category>/maps_local/, maps_comp/   분석용 개별 맵
    outputs/detector/salad/<category>/meta.json    공식·재계산 AUROC, 지연

정규화 분위수는 validation/good으로 계산되므로(공식 코드) validation 점수는 in-sample이다. test만 저장한다.

사용법:
    E:\\venvs\\loco_salad\\Scripts\\python.exe -m src.detector.salad_runner --category breakfast_box
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from ops.train_hooks import RunContext, run_main

ROOT = Path(__file__).resolve().parents[2]
SALAD_DIR = ROOT / "external" / "SALAD"
DATA = ROOT / "MVTec-LOCO"
OUT_ROOT = ROOT / "outputs" / "detector" / "salad"


def map_key(image_id: str) -> str:
    return image_id.replace("/", "__")


def auroc_pair(y, s):
    """공식 방식: (정상 vs 논리), (정상 vs 구조) AUROC의 평균과 각 값."""
    from sklearn.metrics import roc_auc_score

    out = {}
    for lab in ("logical_anomalies", "structural_anomalies"):
        ss = s["good"] + s[lab]
        out[lab] = roc_auc_score([0] * len(y["good"]) + [1] * len(y[lab]), ss)
    return 0.5 * (out["logical_anomalies"] + out["structural_anomalies"]), out


def run(args):
    ctx = RunContext.from_env(f"salad_{args.category}")
    sys.path.insert(0, str(SALAD_DIR))
    import train_salad as ts
    from salad_dataset import (ImageFolderWithoutTarget, ImageFolderWithoutTargetWithSeg, ImageFolderWithPath,
                               ImageFolderWithPathWithSeg)
    from torch.utils.data import DataLoader
    from torchvision import transforms

    for f in (random.seed, np.random.seed, torch.manual_seed):
        f(args.seed)
    cat = args.category
    seg_root = SALAD_DIR / "data" / "mvtec_loco_composition_maps"
    wdir = Path(args.weights_dir) / cat
    ctx.heartbeat(step=0, force=True, phase="load")

    def pair(split, transform, with_path=False):
        img = (ImageFolderWithPath if with_path else ImageFolderWithoutTarget)(str(DATA / cat / split), transform=transform)
        seg = ImageFolderWithoutTarget(str(seg_root / cat / split), transform=transform)
        img.seg, seg.seg = False, True
        return (ImageFolderWithPathWithSeg if with_path else ImageFolderWithoutTargetWithSeg)(img, seg)

    train_set = pair("train", transforms.Lambda(ts.train_transform))
    validation_set = pair("validation", transforms.Lambda(ts.train_transform))
    test_set = pair("test", ts.default_transform, with_path=True)
    g = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=0, generator=g)
    validation_loader = DataLoader(validation_set, batch_size=1)

    # SALAD 가중치는 모듈 전체를 pickle로 저장한 공식 파일이다 (공식 저장소에서 내려받은 파일)
    nets = {n: torch.load(wdir / f"{n}_final.pth", weights_only=False).cuda().eval()
            for n in ("teacher", "autoencoder", "student", "comp_autoencoder", "comp_unet")}
    teacher, autoencoder, student = nets["teacher"], nets["autoencoder"], nets["student"]
    comp_ae, comp_unet = nets["comp_autoencoder"], nets["comp_unet"]

    ctx.heartbeat(step=1, force=True, phase="normalization")
    teacher_mean, teacher_std = ts.teacher_normalization(teacher, train_loader)
    q_st_start, q_st_end, q_ae_start, q_ae_end = ts.map_normalization(
        validation_loader=validation_loader, teacher=teacher, student=student, autoencoder=autoencoder,
        teacher_mean=teacher_mean, teacher_std=teacher_std, desc="map normalization")
    q_eff_start, q_eff_end, q_seg_start, q_seg_end = ts.score_normalization(
        validation_loader=validation_loader, teacher=teacher, student=student, comp_ae=comp_ae,
        comp_unet=comp_unet, autoencoder=autoencoder, teacher_mean=teacher_mean, teacher_std=teacher_std,
        q_st_start=q_st_start, q_st_end=q_st_end, q_ae_start=q_ae_start, q_ae_end=q_ae_end,
        desc="score normalization")
    ctx.heartbeat(step=2, force=True, phase="mahalanobis")
    fm, fc, fms, fcs, fma, fca = ts.extract_features_mahalanobis(train_loader, train_set, student, teacher_mean,
                                                                  teacher_std)
    q_start_mah, q_end_mah = ts.map_normalization_mahalanobis(validation_loader, student, teacher_mean, teacher_std,
                                                              fc, fm, fcs, fms, fca, fma)
    common = dict(teacher_mean=teacher_mean, teacher_std=teacher_std)

    ctx.heartbeat(step=3, force=True, phase="official_test")
    official = ts.test(
        test_set=test_set, teacher=teacher, student=student, comp_ae=comp_ae, comp_unet=comp_unet,
        autoencoder=autoencoder, feature_vectors_covinv=fc, feature_vectors_mean=fm,
        feature_vectors_covinv_seg=fcs, feature_vectors_mean_seg=fms, feature_vectors_covinv_seg_area=fca,
        feature_vectors_mean_seg_area=fma, q_st_start=q_st_start, q_st_end=q_st_end, q_ae_start=q_ae_start,
        q_eff_start=q_eff_start, q_eff_end=q_eff_end, q_ae_end=q_ae_end, q_seg_start=q_seg_start,
        q_seg_end=q_seg_end, q_start_mah=q_start_mah, q_end_mah=q_end_mah, desc="official test", **common)

    out = OUT_ROOT / cat
    for d in ("maps", "maps_local", "maps_comp"):
        (out / d).mkdir(parents=True, exist_ok=True)
    rows, times = [], []
    y = {k: [] for k in ("good", "logical_anomalies", "structural_anomalies")}
    terms = {t: {k: [] for k in y} for t in ("all", "local", "mah", "comp")}
    for i, (image, seg, target, path) in enumerate(test_set):
        ctx.heartbeat(step=4 + i, phase="export")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        image, seg = image.unsqueeze(0).cuda(), seg.cuda()
        mc, _, _ = ts.predict(image=image, teacher=teacher, student=student, autoencoder=autoencoder,
                              q_st_start=q_st_start, q_st_end=q_st_end, q_ae_start=q_ae_start,
                              q_ae_end=q_ae_end, **common)
        mc = ((mc - q_eff_start) / q_eff_end)[0, 0].cpu().numpy()
        mp = ts.predict_comp_map(seg, comp_ae, comp_unet).unsqueeze(0)
        mp = ((mp - q_seg_start) / q_seg_end)[0, 0].cpu().numpy()
        mah = ts.predict_mahalanobis(image=image, seg=seg, teacher=student, teacher_mean=teacher_mean,
                                     feature_vectors_covinv=fc, feature_vectors_mean=fm,
                                     feature_vectors_covinv_seg=fcs, feature_vectors_mean_seg=fms,
                                     feature_vectors_covinv_seg_area=fca, feature_vectors_mean_seg_area=fma,
                                     teacher_std=teacher_std, q_start=q_start_mah, q_end=q_end_mah)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        mah = float(mah)
        s_local, s_comp = float(np.max(mc)), float(np.max(mp))
        score = s_local + mah + s_comp
        sub = os.path.basename(os.path.dirname(path))
        image_id = f"{cat}/test/{sub}/{Path(path).stem}"
        key = map_key(image_id)
        mp256 = cv2.resize(mp.astype(np.float32), (mc.shape[1], mc.shape[0]), interpolation=cv2.INTER_LINEAR)
        np.save(out / "maps_local" / f"{key}.npy", mc.astype(np.float16))
        np.save(out / "maps_comp" / f"{key}.npy", mp.astype(np.float16))
        np.save(out / "maps" / f"{key}.npy", (mc + mp256).astype(np.float16))
        rows.append((image_id, score, s_local, mah, s_comp))
        y[sub].append(0 if sub == "good" else 1)
        for t, v in (("all", score), ("local", s_local), ("mah", mah), ("comp", s_comp)):
            terms[t][sub].append(v)

    with open(out / "scores.csv", "w", encoding="utf-8") as f:
        f.write("image_id,score,s_local,s_mah,s_comp\n")
        for r in rows:
            f.write(",".join([r[0]] + [f"{v:.8g}" for v in r[1:]]) + "\n")
    recomputed, detail = {}, {}
    for t, name in (("all", "auc"), ("local", "auc_img"), ("mah", "auc_maha"), ("comp", "auc_comp")):
        avg, per = auroc_pair(y, terms[t])
        recomputed[name] = 100 * avg
        detail[name] = {k: 100 * v for k, v in per.items()}
    official = dict(zip(("auc", "auc_img", "auc_maha", "auc_comp"), map(float, official)))
    diffs = {k: abs(official[k] - recomputed[k]) for k in official}
    meta = {"category": cat, "official_auroc": official, "recomputed_auroc": recomputed,
            "recomputed_detail": detail, "max_abs_diff": max(diffs.values()),
            "n_test": len(rows), "latency_ms_median": round(1000 * float(np.median(times[1:])), 2),
            "seed": args.seed}
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=1))
    if meta["max_abs_diff"] > 1e-4:
        raise RuntimeError(f"저장값 재계산 AUROC가 공식 출력과 다름: {diffs}")
    ctx.final_dir.mkdir(parents=True, exist_ok=True)
    (ctx.final_dir / "summary.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    ctx.mark_done(auc=recomputed["auc"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", required=True)
    ap.add_argument("--weights-dir", default=str(SALAD_DIR / "results"))
    ap.add_argument("--seed", type=int, default=42)
    run_main(run, ap.parse_args())


if __name__ == "__main__":
    main()
