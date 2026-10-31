"""예측 원본·탐지 점수 캐시 공개: GPU와 데이터셋 없이 표1·캐스케이드·P2 표를 다시 계산할 수 있게 한다.

공개 대상(모두 이미지 ID와 수치·텍스트만, 데이터셋 이미지와 MMAD QA 텍스트는 제외):
- runs/<run>/final/heldout_predictions.jsonl, summary.json  -> reports/predictions/runs/<run>/
- outputs/baselines/**  (B0·B1·B3, P2-b B1)                 -> reports/predictions/baselines/
- outputs/detector/salad/<cat>/scores.csv, meta.json        -> reports/predictions/detector/salad/<cat>/
- data/sft/<dir>/meta.json (학습 표본 수)                    -> reports/predictions/sft_meta/<dir>/
- data/sft/<dir>/heldout.jsonl 중 B1 힌트가 있는 기록(id·라벨·정답 리포트·힌트만, P3 복사 진단용)
                                                            -> reports/predictions/sft_hints/<dir>/heldout_hints.jsonl
- data/index.parquet의 라벨 열(qa_*, caption_path 제외)      -> reports/predictions/index_slim.parquet
MANIFEST.json에 파일별 sha256과 크기, 내보낸 커밋을 적는다.

    python -m src.report.publish_predictions export    # 로컬 결과 -> reports/predictions/
    python -m src.report.publish_predictions restore   # reports/predictions/ -> runs/, outputs/, data/ (있는 파일은 건드리지 않음)
    python -m src.report.publish_predictions verify    # MANIFEST의 sha256 대조

복원 뒤 `python -m src.report.m4_tables`, `python -m src.eval.cascade`, `python -m src.report.p2_report`,
`python -m src.report.p3_report`가 GPU 없이 돈다.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "reports" / "predictions"
RUN_FILES = ("heldout_predictions.jsonl", "summary.json")
DETECTOR_FILES = ("scores.csv", "meta.json")
INDEX_DROP_PREFIX = "qa_"
INDEX_DROP = ("caption_path",)
SKIP_RUN_PREFIX = ("_", "f1_", "f2_", "f3_", "f4_", "f5_")  # 스모크·고장 주입 시험
HINT_KEYS = ("id", "label", "gt_report", "hints")  # P3 힌트 기록에서 공개하는 필드 (이미지 경로·프롬프트 제외)
HINT_FILE = "heldout_hints.jsonl"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect(root: Path) -> list:
    """(원본 경로, 공개 상대 경로) 목록. 인덱스는 별도 처리."""
    items = []
    runs = root / "runs"
    if runs.exists():
        for d in sorted(runs.iterdir()):
            if not d.is_dir() or d.name.startswith(SKIP_RUN_PREFIX):
                continue
            if not (d / "final" / RUN_FILES[0]).exists():
                continue
            for name in RUN_FILES:
                p = d / "final" / name
                if p.exists():
                    items.append((p, f"runs/{d.name}/{name}"))
    base = root / "outputs" / "baselines"
    if base.exists():
        for p in sorted(base.rglob("*")):
            if p.is_file() and p.suffix in (".jsonl", ".json"):
                items.append((p, "baselines/" + p.relative_to(base).as_posix()))
    det = root / "outputs" / "detector" / "salad"
    if det.exists():
        for c in sorted(det.iterdir()):
            for name in DETECTOR_FILES:
                p = c / name
                if p.exists():
                    items.append((p, f"detector/salad/{c.name}/{name}"))
    sft = root / "data" / "sft"
    if sft.exists():
        for d in sorted(sft.iterdir()):
            p = d / "meta.json"
            if p.exists():
                items.append((p, f"sft_meta/{d.name}/meta.json"))
    return items


def slim_hints(src: Path, dst: Path) -> int:
    """SFT held-out 기록 중 B1 힌트가 있는 것만 HINT_KEYS 필드로 줄여 쓴다. 쓴 줄 수를 돌려준다(0이면 파일을 만들지 않는다)."""
    rows = []
    for line in src.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("hints"):
                rows.append({k: r.get(k) for k in HINT_KEYS})
    if rows:
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return len(rows)


def target_of(root: Path, rel: str) -> Path:
    """공개 상대 경로 -> 로컬 원래 자리."""
    head, _, rest = rel.partition("/")
    if head == "runs":
        run, _, name = rest.partition("/")
        return root / "runs" / run / "final" / name
    if head == "baselines":
        return root / "outputs" / "baselines" / rest
    if head == "detector":
        return root / "outputs" / "detector" / rest
    if head in ("sft_meta", "sft_hints"):
        return root / "data" / "sft" / rest
    if rel == "index_slim.parquet":
        return root / "data" / "index.parquet"
    raise ValueError(f"알 수 없는 공개 경로: {rel}")


def slim_index(src: Path, dst: Path) -> list:
    """MMAD QA 텍스트(qa_*)와 캡션 경로를 뺀 인덱스. 남긴 열 이름을 돌려준다."""
    import pandas as pd

    df = pd.read_parquet(src)
    keep = [c for c in df.columns if not c.startswith(INDEX_DROP_PREFIX) and c not in INDEX_DROP]
    dst.parent.mkdir(parents=True, exist_ok=True)
    df[keep].to_parquet(dst, index=False)
    return keep


def git_head(root: Path) -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return None


def export(root: Path = ROOT, dest: Path = DEST) -> dict:
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    files = []
    for src, rel in collect(root):
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, out)
        files.append({"path": rel, "sha256": sha256(out), "bytes": out.stat().st_size})
    sft = root / "data" / "sft"
    if sft.exists():
        for d in sorted(sft.iterdir()):
            src = d / "heldout.jsonl"
            if src.exists():
                out = dest / "sft_hints" / d.name / HINT_FILE
                if slim_hints(src, out):
                    files.append({"path": f"sft_hints/{d.name}/{HINT_FILE}", "sha256": sha256(out),
                                  "bytes": out.stat().st_size})
    index = root / "data" / "index.parquet"
    manifest = {"exported_at": time.strftime("%Y-%m-%d %H:%M:%S"), "commit": git_head(root), "files": files}
    if index.exists():
        out = dest / "index_slim.parquet"
        manifest["index_columns"] = slim_index(index, out)
        files.append({"path": "index_slim.parquet", "sha256": sha256(out), "bytes": out.stat().st_size})
    manifest["total_bytes"] = sum(f["bytes"] for f in files)
    (dest / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    return manifest


def load_manifest(dest: Path = DEST) -> dict:
    return json.loads((dest / "MANIFEST.json").read_text(encoding="utf-8"))


def verify(dest: Path = DEST) -> list:
    """MANIFEST와 다른 파일 목록 (빈 목록이면 정상)."""
    bad = []
    for f in load_manifest(dest)["files"]:
        p = dest / f["path"]
        if not p.exists() or sha256(p) != f["sha256"]:
            bad.append(f["path"])
    return bad


def restore(root: Path = ROOT, dest: Path = DEST, force: bool = False) -> dict:
    """공개 파일을 원래 자리로 복사. 이미 있으면 건너뛴다(force=True면 덮어씀)."""
    copied, skipped = [], []
    for f in load_manifest(dest)["files"]:
        src, dst = dest / f["path"], target_of(root, f["path"])
        if dst.exists() and not force:
            skipped.append(f["path"])
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        copied.append(f["path"])
    return {"copied": copied, "skipped": skipped}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["export", "restore", "verify"])
    ap.add_argument("--force", action="store_true", help="restore: 있는 파일도 덮어쓴다")
    a = ap.parse_args()
    if a.action == "export":
        m = export()
        print(f"내보냄: {len(m['files'])}개, {m['total_bytes'] / 1e6:.1f}MB -> {DEST}")
    elif a.action == "verify":
        bad = verify()
        print("MANIFEST 일치" if not bad else "불일치: " + ", ".join(bad))
        raise SystemExit(1 if bad else 0)
    else:
        r = restore(force=a.force)
        print(f"복원 {len(r['copied'])}개, 건너뜀(이미 있음) {len(r['skipped'])}개")


if __name__ == "__main__":
    main()
