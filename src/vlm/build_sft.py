"""VLM SFT 데이터 생성 (fold 단위).

    train   : fold f 중 inner-val이 아닌 이미지 (이상 + 정상 하드 네거티브, 목록 드롭아웃 적용)
    val     : fold f의 inner-val (설정 선택 전용, 드롭아웃 없음)
    heldout : 다른 fold (최종 1회 평가)

이미지 3장(검사, crop, 정상 참조)은 약 256토큰 크기로 줄여 data/vlm_cache/에 캐시한다.
crop 출처
- gt       : GT bbox를 1.5배 확장 (탐지 캐시가 생기기 전 스모크·고장 주입 시험용, oracle)
- <탐지기> : 탐지 맵에서 기준을 넘는 최대 영역의 bbox를 1.5배 확장, 영역 윤곽선 표시.
             기준을 넘는 영역이 없으면 축소한 전체 이미지. 탐지기 점수 백분위·의심 위치를 프롬프트에 넣는다.
정상 참조: train/good에서 DINOv2 CLS 최근접 1장 (data/emb_dinov2_base_cls.npz).
하드 네거티브(--hard-neg N): 학습 측 정상 중 탐지 점수 상위 N장만 train에 넣는다(카테고리 정상 수 비율로 배분).
  N=0이면 정상 이미지를 학습에 넣지 않는다 (오탐 거부 학습 ablation).
입력 ablation(--drop-image ref|crop): 해당 이미지를 빼고 프롬프트의 이미지 설명도 맞춘다.
텍스트 ablation: --no-spec(정상 규격 줄 제거), --no-detector-text(탐지기 백분위·의심 위치 줄 제거).
open-set(--exclude-groups): 지정 그룹 이미지를 train·val에서 빼고 모든 프롬프트 목록에서도 지운다.
  held-out의 해당 이미지는 정답을 미분류 템플릿으로 두고 openset=true로 표시한다 (RQ3).
B2 VLM 단독(--crop-source none): 탐지기 없이 검사 이미지 + 정상 참조 2장, 탐지기 줄 없음, 학습 측 정상 전부 사용.
라벨 수 곡선(--label-frac): 하드 네거티브 선택 뒤 train·val을 (역할, 카테고리, 정상/그룹) 층별로 줄인다
  (층마다 최소 1장, 비율 간 중첩). B1도 같은 SFT 폴더로 학습하므로 같은 부분집합을 쓴다 (P2).

사용법:
    python -m src.vlm.build_sft --fold 0 --categories breakfast_box screw_bag --crop-source gt --name m1_gt
"""

import argparse
import json
import os
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from src.data.masks import CATEGORIES
from src.detector.cache import FoldCalibration, map_key
from src.kb import defect_table, load_kb
from src.output_spec import MAX_FREE_DETAIL, NORMAL, format_report, gt_report, parse_report, unknown_report
from src.vlm.prompts import IMAGE_ROLES, image_size_for, user_text

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "MVTec-LOCO"
CACHE = ROOT / "data" / "vlm_cache"
CROP_EXPAND = 1.5
CONTOUR_RGB = (255, 0, 255)  # 색상 결함과 헷갈리지 않도록 마젠타 윤곽선만 그린다 (영역 채색 금지)


def rel(p: Path) -> str:
    return p.resolve().relative_to(ROOT).as_posix()


def save_resized(img: Image.Image, path: Path) -> None:
    """임시 파일에 쓴 뒤 교체한다. 쓰는 도중 프로세스가 종료돼도(일시 정지 등) 빈 파일이 캐시에 남지 않는다.

    캐시는 "파일이 있으면 건너뛴다" 방식이라, 반쯤 쓴 파일이 남으면 이후 실행이 계속 그 파일을 쓴다
    (2026-10-25 일시 정지로 0바이트 PNG가 남아 4B 학습의 held-out 생성이 실패한 사례).
    """
    W, H = image_size_for(*img.size)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.{os.getpid()}.tmp{path.suffix}")
    img.resize((W, H), Image.BICUBIC).save(tmp)
    os.replace(tmp, path)


def cached_full(image_path: str) -> Path:
    out = CACHE / "img" / (image_path.replace("/", "__"))
    if not out.exists():
        save_resized(Image.open(DATA / image_path).convert("RGB"), out)
    return out


def expand_box(x0, y0, x1, y1, w, h, k=CROP_EXPAND):
    cx, cy, bw, bh = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) * k, (y1 - y0) * k
    # 너무 작은 영역은 긴 변의 1/8 이상으로 맞춰 주변 맥락을 남긴다
    m = max(w, h) / 8
    bw, bh = max(bw, m), max(bh, m)
    return (int(max(0, cx - bw / 2)), int(max(0, cy - bh / 2)), int(min(w, cx + bw / 2)), int(min(h, cy + bh / 2)))


def crop_gt(row, out: Path) -> Path:
    if not out.exists():
        img = Image.open(DATA / row.image_path).convert("RGB")
        if row.bbox is None:
            save_resized(img, out)
        else:
            w, h = img.size
            b = row.bbox
            save_resized(img.crop(expand_box(b[0] * w, b[1] * h, b[2] * w, b[3] * h, w, h)), out)
    return out


def crop_detector(row, calib: FoldCalibration, out: Path) -> tuple[Path, str]:
    img = Image.open(DATA / row.image_path).convert("RGB")
    w, h = img.size
    region, cell = calib.region(row.image_id, (h, w))
    if not out.exists():
        if region is None:
            save_resized(img, out)
        else:
            arr = np.array(img)
            cnts, _ = cv2.findContours(region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(arr, cnts, -1, CONTOUR_RGB, max(2, round(max(w, h) / 400)))
            ys, xs = np.nonzero(region)
            box = expand_box(xs.min(), ys.min(), xs.max() + 1, ys.max() + 1, w, h)
            save_resized(Image.fromarray(arr).crop(box), out)
    return out, cell


def nearest_reference(emb: dict, train_good: dict, image_id: str, category: str) -> str:
    ids, mat = train_good[category]
    sims = mat @ emb[image_id]
    if image_id in ids:  # train/good 자신은 제외 (현재 SFT 대상은 test뿐이라 실제로는 발생하지 않음)
        sims[ids.index(image_id)] = -1
    return ids[int(sims.argmax())]


def free_text(kb, category, defects) -> str:
    """목록 드롭아웃 정답의 자유 서술: 첫 세부 결함 ko (20자 이내)."""
    t = defect_table(kb, category)
    d = sorted((t[n] for n in defects), key=lambda x: x["order"])[0]
    return d["ko"][:MAX_FREE_DETAIL]


def select_hard_negatives(test: pd.DataFrame, calibs: dict, total: int) -> set:
    """학습 측(inner-val 제외) 정상 중 탐지 점수 백분위 상위를 카테고리 정상 수 비율대로 고른다.

    validation/good과 train/good은 탐지기에 in-sample이라 후보에서 제외된다 (test/good만 대상).
    """
    cand = test[(test["role"] == "train") & (test["label"] == "good")]
    n_all = len(cand)
    keep = set()
    for cat, g in cand.groupby("category"):
        k = min(len(g), round(total * len(g) / n_all))
        pct = g["image_id"].map(calibs[cat].percentile)
        keep.update(g.assign(p=pct.values).sort_values("p", ascending=False)["image_id"].head(k))
    return keep


def subsample_labels(pool: pd.DataFrame, frac: float, seed: int = 0) -> set | None:
    """라벨 수 곡선(P2): 학습 측(train·val) 이미지를 (역할, 카테고리, 정상/그룹) 층마다 frac만 남긴다.

    - 층마다 최소 1장을 남겨 모든 그룹이 학습에 나타나게 한다(곡선이 그룹 수가 아니라 그룹당 표본 수를 재도록).
    - 층마다 frac과 무관한 고정 순열의 앞부분을 쓰므로 25% 집합 ⊂ 50% 집합 ⊂ 전체가 된다(중첩).
    - held-out은 건드리지 않는다. frac >= 1이면 None(전부 사용).
    """
    if frac >= 1:
        return None
    rng = np.random.default_rng(seed)
    side = pool[pool["role"] != "heldout"]
    key = side["groups"].map(lambda g: g[0] if len(g) else "good")
    keep = set()
    for _, g in side.groupby([side["role"], side["category"], key], sort=True):
        ids = np.array(sorted(g["image_id"]))
        order = rng.permutation(len(ids))
        keep.update(ids[order[:max(1, round(frac * len(ids)))]].tolist())
    return keep


def group_hints(kb: dict, category: str, probs: dict, allowed: set | None = None, k: int = 2) -> list:
    """P3 변형 B: B1 클래스 확률({'good' 또는 'a+b': p})을 결함 그룹으로 합산해 상위 k개 [(그룹 ko_label|정상, p)].

    allowed가 있으면 그 그룹(프롬프트 목록에 보이는 그룹 + 정상)만 후보로 둔다: 목록 드롭아웃으로 목록에서 뺀 그룹을
    힌트가 다시 알려 주지 않게 해 학습·held-out의 힌트 분포를 같게 유지한다. 확률은 다시 정규화하지 않는다."""
    d2g = {}
    for g in kb[category]["groups"].values():
        for dname in g["defects"]:
            d2g[dname] = g["ko_label"]
    acc = {}
    for cls, p in probs.items():
        if cls == "good":
            label = NORMAL
        else:
            names = cls.split("+")
            table = defect_table(kb, category)
            first = sorted((n for n in names if n in table), key=lambda n: table[n]["order"])
            label = d2g.get(first[0] if first else names[0])
            if label is None:
                raise KeyError(f"{category}: KB에 없는 B1 클래스 {cls!r}")
        acc[label] = acc.get(label, 0.0) + float(p)
    items = [(g, p) for g, p in acc.items() if allowed is None or g in allowed]
    items.sort(key=lambda x: (-x[1], x[0]))
    return [(g, round(p, 2)) for g, p in items[:k]]


def load_hint_probs(path: Path) -> dict:
    """crop_classifier --probs-out 파일(held-out)과 같은 이름의 .oof.jsonl(학습 측 OOF·드롭아웃) -> {id: probs}."""
    out = {}
    for p in (path, path.with_name(path.stem + ".oof.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                out[r["id"]] = r["probs"]
    return out


def parse_excluded(spec: str, categories: list) -> set:
    """'all:contamination,breakfast_box:fruit_count' -> {(카테고리, 그룹ID)}. 없는 그룹이면 오류."""
    kb = load_kb()
    out = set()
    for item in filter(None, (x.strip() for x in spec.split(","))):
        cat, gid = item.split(":")
        for c in (categories if cat == "all" else [cat]):
            if gid not in kb[c]["groups"]:
                raise ValueError(f"KB에 없는 그룹: {c}:{gid}")
            out.add((c, gid))
    return out


def build_parser() -> argparse.ArgumentParser:
    """대기열 명령 검사 테스트(tests/test_ops.py)도 이 파서를 쓴다."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--categories", nargs="+", default=CATEGORIES)
    ap.add_argument("--crop-source", default="gt", help="gt 또는 탐지기 이름 (outputs/detector/<이름>)")
    ap.add_argument("--name", required=True, help="data/sft/<이름>_f<fold>/에 저장")
    ap.add_argument("--dropout", type=float, default=0.12, help="목록 드롭아웃 비율 (train 이상 샘플)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hard-neg", type=int, default=None, help="train 정상 하드 네거티브 총수 (탐지기 crop일 때)")
    ap.add_argument("--drop-image", choices=["ref", "crop"], default=None, help="입력 ablation")
    ap.add_argument("--no-spec", action="store_true", help="정상 규격 줄 제거")
    ap.add_argument("--no-detector-text", action="store_true", help="탐지기 백분위·의심 위치 줄 제거")
    ap.add_argument("--exclude-groups", default="",
                    help="open-set 제외 그룹: '카테고리:그룹ID' 쉼표 구분, 카테고리 자리에 all 가능")
    ap.add_argument("--label-frac", type=float, default=1.0,
                    help="라벨 수 곡선(P2): 학습 측(train·val) 이미지를 층별로 이 비율만 사용")
    ap.add_argument("--det-terms", action="store_true",
                    help="P3 변형 A: 탐지기 줄에 SALAD 세 항(국소/전역/구성)의 정상 대비 백분위를 추가")
    ap.add_argument("--b1-hints", default=None,
                    help="P3 변형 B: crop_classifier --probs-out 파일. 상위 2그룹 후보와 확률을 프롬프트에 추가")
    return ap


def main():
    args = build_parser().parse_args()
    excluded = parse_excluded(args.exclude_groups, args.categories)

    kb = load_kb()
    df = pd.read_parquet(ROOT / "data" / "index.parquet")
    sp = pd.read_csv(ROOT / "data" / "splits.csv")
    df = df.merge(sp[["image_id", "fold", "inner_val"]], on="image_id", how="left")
    df = df[df["category"].isin(args.categories)]
    z = np.load(ROOT / "data" / "emb_dinov2_base_cls.npz")
    emb = dict(zip(z["image_id"], z["emb"]))
    train_good = {}
    for cat, g in df[(df["split"] == "train")].groupby("category"):
        ids = g["image_id"].tolist()
        train_good[cat] = (ids, np.stack([emb[i] for i in ids]))

    test = df[df["split"] == "test"].copy()
    test["role"] = np.where(test["fold"] != args.fold, "heldout", np.where(test["inner_val"], "val", "train"))
    calibs = {}
    use_detector = args.crop_source not in ("gt", "none")
    if use_detector:
        for cat in args.categories:
            normal = test[(test["category"] == cat) & (test["role"] != "heldout") & (test["label"] == "good")]
            calibs[cat] = FoldCalibration(args.crop_source, cat, normal["image_id"].tolist())

    keep_neg = None
    if args.hard_neg == 0:
        keep_neg = set()  # 정상 이미지를 학습에 넣지 않음
    elif args.hard_neg and use_detector:
        keep_neg = select_hard_negatives(test, calibs, args.hard_neg)
    pool = test
    if keep_neg is not None:  # 하드 네거티브를 고른 뒤의 학습 측에서 라벨을 줄인다
        pool = test[~((test["role"] == "train") & (test["label"] == "good") & ~test["image_id"].isin(keep_neg))]
    keep_lab = subsample_labels(pool, args.label_frac, args.seed)
    slot = {"ref": 2, "crop": 1}.get(args.drop_image)
    if args.crop_source == "none":
        slot = 1  # 탐지기가 없으면 crop도 없다
    roles = tuple(r for i, r in enumerate(IMAGE_ROLES) if i != slot)
    rng = random.Random(args.seed * 100 + args.fold)
    hint_probs = load_hint_probs(ROOT / args.b1_hints) if args.b1_hints else None
    crop_dir = CACHE / (f"crop_gt" if args.crop_source == "gt" else f"crop_{args.crop_source}_f{args.fold}")
    out_dir = ROOT / "data" / "sft" / f"{args.name}_f{args.fold}"
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {r: open(out_dir / f"{r}.jsonl", "w", encoding="utf-8") for r in ("train", "val", "heldout")}
    n = {r: {"good": 0, "anomaly": 0, "dropped": 0} for r in files}
    n_open = 0
    for row in test.itertuples():
        if keep_neg is not None and row.role == "train" and row.label == "good" and row.image_id not in keep_neg:
            continue
        if keep_lab is not None and row.role != "heldout" and row.image_id not in keep_lab:
            continue
        is_open = row.label != "good" and (row.category, row.groups[0]) in excluded
        if is_open and row.role != "heldout":
            continue  # 미등록 결함은 학습·inner-val에 넣지 않는다
        groups_all = [g["ko_label"] for gid, g in kb[row.category]["groups"].items()
                      if (row.category, gid) not in excluded]
        crop_out = crop_dir / (map_key(row.image_id) + ".png")
        det = None
        if args.crop_source == "gt":
            crop = crop_gt(row, crop_out)
        elif args.crop_source == "none":
            crop = None
        else:
            crop, cell = crop_detector(row, calibs[row.category], crop_out)
            det = {"percentile": calibs[row.category].percentile(row.image_id), "location": cell}
            if args.det_terms:
                det["terms"] = calibs[row.category].term_percentiles(row.image_id)
        ref_id = nearest_reference(emb, train_good, row.image_id, row.category)
        ref_path = df.loc[df["image_id"] == ref_id, "image_path"].iloc[0]
        images = [rel(cached_full(row.image_path)), rel(crop) if crop else None, rel(cached_full(ref_path))]
        if slot is not None:
            images.pop(slot)

        shown, dropped = groups_all, False
        if row.label == "good":
            target = NORMAL
        elif is_open:
            # 목록에 없는 결함: 미분류로 답하고 세부를 서술하는 것이 정답 (채점은 src.eval.openset)
            loc = None if pd.isna(row.location) else row.location
            target = format_report(unknown_report(loc, row.defect_type, free_text(kb, row.category, row.defects)))
            n_open += 1
        else:
            target = row.gt_report
            if row.role == "train" and rng.random() < args.dropout:
                gl = kb[row.category]["groups"][row.groups[0]]["ko_label"]
                shown = [g for g in groups_all if g != gl]
                dropped = True
                # 위치 생략(instances) 결함은 parquet에서 None이 NaN으로 읽히므로 None으로 되돌린다
                loc = None if pd.isna(row.location) else row.location
                target = format_report(unknown_report(loc, row.defect_type,
                                                      free_text(kb, row.category, row.defects)))
        # 모든 정답은 출력 명세 파서를 통과해야 한다 (잘못된 학습 목표 방지)
        if target != NORMAL and parse_report(target) is None:
            raise ValueError(f"{row.image_id}: 출력 명세에 맞지 않는 정답 {target!r}")
        hints = None
        if hint_probs is not None:
            if row.image_id not in hint_probs:
                raise KeyError(f"{row.image_id}: B1 확률 파일에 없음 ({args.b1_hints})")
            hints = group_hints(kb, row.category, hint_probs[row.image_id], set(shown) | {NORMAL})
        rec = {
            "id": row.image_id, "category": row.category, "role": row.role, "label": row.label,
            "group": row.groups[0] if len(row.groups) else None, "dropped": dropped, "openset": is_open,
            "images": images, "reference_id": ref_id, "detector": det, "hints": hints,
            "prompt": user_text(kb, row.category, shown, None if args.no_detector_text else det, roles,
                                spec=not args.no_spec, hints=hints),
            "target": target,
            "gt_report": row.gt_report,
        }
        files[row.role].write(json.dumps(rec, ensure_ascii=False) + "\n")
        k = n[row.role]
        k["good" if row.label == "good" else "anomaly"] += 1
        k["dropped"] += dropped
    for f in files.values():
        f.close()
    meta = {"fold": args.fold, "categories": args.categories, "crop_source": args.crop_source,
            "dropout": args.dropout, "seed": args.seed, "counts": n, "hard_neg": args.hard_neg,
            "drop_image": args.drop_image, "no_spec": args.no_spec, "no_detector_text": args.no_detector_text,
            "excluded_groups": sorted(f"{c}:{g}" for c, g in excluded), "n_openset_heldout": n_open,
            "label_frac": args.label_frac, "det_terms": args.det_terms, "b1_hints": args.b1_hints,
            "map_thresholds": {c: cb.map_thr for c, cb in calibs.items()}}
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"저장: {out_dir}")
    print(json.dumps(n, ensure_ascii=False))


if __name__ == "__main__":
    main()
