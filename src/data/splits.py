"""test 셋을 VLM 학습/평가용 2-fold로 층화 분할하고, fold마다 inner-val을 떼어 둔다.

층(stratum) = 카테고리 + 라벨(good/logical/structural) + 결함 그룹 조합.
샘플이 2개 미만인 층은 카테고리 + 라벨 층으로 합친다.
모델 k는 fold k로 학습하고 다른 fold(held-out)에서 1회만 평가한다.

근접 중복: src.data.near_dups가 만든 쌍 목록(코사인 유사도 >= 임계값)이 있으면
연결 성분 단위로 같은 fold에 모은다(성분의 다수 fold로 이동).

inner-val: fold마다 학습 측의 15%를 (카테고리, 라벨, 그룹) 단위로 층화해 뗀다.
하이퍼파라미터·조기 종료·프롬프트·거부 임계값 선택 전용이며, 그 fold에 2장 이하인
그룹은 inner-val에 넣지 않는다. 근접 중복 성분은 통째로 inner-val 또는 학습에 둔다.

사용법:
    python -m src.data.splits --index data/index.parquet --out data/splits.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

N_FOLDS = 2
INNER_VAL_FRAC = 0.15
INNER_VAL_MIN_GROUP = 3


def make_strata(test: pd.DataFrame) -> pd.Series:
    base = test["category"] + "|" + test["label"]
    fine = base + "|" + test["groups"].map(lambda g: "+".join(g))
    counts = fine.map(fine.value_counts())
    return fine.where(counts >= N_FOLDS, base)


def dup_components(image_ids: pd.Series, pairs: pd.DataFrame) -> pd.Series:
    """근접 중복 쌍으로 연결 성분 번호를 매긴다 (중복이 없는 이미지는 자기 혼자 성분)."""
    pos = {iid: i for i, iid in enumerate(image_ids)}
    parent = list(range(len(image_ids)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in zip(pairs["image_a"], pairs["image_b"]):
        if a in pos and b in pos:
            parent[find(pos[a])] = find(pos[b])
    return pd.Series([find(i) for i in range(len(image_ids))], index=image_ids.index)


def merge_dup_folds(fold: pd.Series, comp: pd.Series) -> int:
    """fold에 걸친 성분을 다수 fold로 옮긴다(동률이면 작은 fold 번호). 옮긴 이미지 수 반환."""
    moved = 0
    for _, idx in comp.groupby(comp).groups.items():
        f = fold.loc[idx]
        if f.nunique() > 1:
            target = f.value_counts().sort_index().idxmax()
            moved += int((f != target).sum())
            fold.loc[idx] = target
    return moved


def select_inner_val(test: pd.DataFrame, fold: pd.Series, comp: pd.Series, seed: int) -> pd.Series:
    """fold별 (카테고리, 라벨, 그룹) 단위로 15%를 inner-val로 고른다. 성분 단위로 뽑는다."""
    rng = np.random.default_rng(seed)
    key = test["category"] + "|" + test["label"] + "|" + test["groups"].map(lambda g: "+".join(g))
    inner = pd.Series(False, index=test.index)
    for (_, _), idx in test.groupby([fold, key]).groups.items():
        n = len(idx)
        if n < INNER_VAL_MIN_GROUP:
            continue
        n_val = max(1, round(INNER_VAL_FRAC * n))
        comps = comp.loc[idx].unique()
        rng.shuffle(comps)
        picked = 0
        for c in comps:
            members = idx[comp.loc[idx].to_numpy() == c]
            # 성분이 커서 목표를 넘기면 건너뛴다 (학습 측에 남김)
            if picked + len(members) > n_val:
                continue
            inner.loc[members] = True
            picked += len(members)
            if picked >= n_val:
                break
    return inner


def main():
    root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, default=root / "data" / "index.parquet")
    ap.add_argument("--dups", type=Path, default=root / "data" / "near_dups.csv")
    ap.add_argument("--out", type=Path, default=root / "data" / "splits.csv")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_parquet(args.index)
    test = df[df["split"] == "test"].reset_index(drop=True)
    strata = make_strata(test)

    fold = pd.Series(-1, index=test.index)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=args.seed)
    for k, (_, idx) in enumerate(skf.split(test, strata)):
        fold.iloc[idx] = k

    if args.dups.exists():
        pairs = pd.read_csv(args.dups)
        comp = dup_components(test["image_id"], pairs)
        moved = merge_dup_folds(fold, comp)
        n_comp = int((comp.value_counts() > 1).sum())
        print(f"근접 중복: 쌍 {len(pairs)}개, 성분 {n_comp}개, fold 이동 {moved}장")
    else:
        comp = pd.Series(np.arange(len(test)), index=test.index)
        print(f"근접 중복 목록 없음({args.dups}) - fold 이동 없이 진행")

    inner = select_inner_val(test, fold, comp, args.seed)

    out = pd.DataFrame({
        "image_id": test["image_id"],
        "stratum": strata,
        "fold": fold,
        "inner_val": inner,
        "dup_group": comp.map(lambda c: c if (comp == c).sum() > 1 else -1),
    })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False, encoding="utf-8")
    print(f"저장: {args.out} ({len(out)}행)")

    m = out.merge(test[["image_id", "category", "label", "groups"]])
    print(m.groupby(["category", "label", "fold"]).size().unstack(fill_value=0))
    print("\nfold별 inner-val 비율:")
    print(m.groupby(["fold", "label"])["inner_val"].agg(["sum", "count", "mean"]).round(3))
    m["group"] = m["groups"].map(lambda g: g[0] if len(g) else "good")
    print("\ninner-val 그룹 분포 (fold, 카테고리, 그룹: 학습/inner-val):")
    tab = m.groupby(["fold", "category", "group"])["inner_val"].agg(
        train=lambda x: int((~x).sum()), val=lambda x: int(x.sum()))
    with pd.option_context("display.max_rows", 300):
        print(tab.to_string())


if __name__ == "__main__":
    main()
