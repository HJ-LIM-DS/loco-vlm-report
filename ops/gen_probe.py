"""생성 지연 탐침: 같은 조건에서 어댑터별 held-out 생성 속도와 GPU 메모리 사용을 짧게 다시 잰다.

P2 8B의 held-out 생성이 fold 0은 32.5초/장, fold 1은 12.3초/장으로 2.6배 달랐던 원인을 가리기 위해 쓴다.
fold별 held-out 레코드에서 일정 간격으로 n장을 뽑아(카테고리·라벨이 고르게 섞임) 같은 배치 크기로 생성하고,
5초마다 nvidia-smi로 GPU 전체 사용 메모리와 다른 프로세스의 점유를 기록한다.

    python -m ops.gen_probe --label idle                       # 기본: p2_8b_f{f}, fold 0·1, 40장, 배치 2
    python -m ops.gen_probe --label load --folds 0             # 다른 GPU 부하를 걸어 둔 상태에서 비교

결과: reports/ops/gen_probe_<label>.json
"""

import argparse
import gc
import json
import os
import subprocess
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_EVERY_S = 5.0


def nvidia_smi(args: list) -> str:
    try:
        return subprocess.run(["nvidia-smi", *args], capture_output=True, text=True, timeout=10, check=True).stdout
    except Exception:  # noqa: BLE001
        return ""


class GpuSampler(threading.Thread):
    """GPU 전체 사용 메모리(MB)·이용률과 프로세스별 점유를 주기적으로 기록한다."""

    def __init__(self):
        super().__init__(daemon=True)
        self.samples, self.apps, self._halt = [], {}, threading.Event()  # Thread._stop과 이름이 겹치지 않게

    def run(self):
        while not self._halt.is_set():
            q = nvidia_smi(["--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"])
            if q.strip():
                used, total, util = (float(x) for x in q.strip().split(","))
                self.samples.append({"t": time.time(), "used_mb": used, "total_mb": total, "util": util})
            # Windows(WDDM)에서는 프로세스별 메모리가 [N/A]로 나오므로 pid만 기록한다
            for line in nvidia_smi(["--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"]).splitlines():
                if "," in line:
                    pid, mem = (x.strip() for x in line.split(",", 1))
                    try:
                        val = float(mem)
                    except ValueError:
                        val = None
                    if pid.isdigit():
                        prev = self.apps.get(int(pid))
                        self.apps[int(pid)] = max(prev or 0.0, val) if val is not None else prev
            self._halt.wait(SAMPLE_EVERY_S)

    def stop(self) -> dict:
        self._halt.set()
        self.join(timeout=15)
        used = sorted(s["used_mb"] for s in self.samples)
        me = os.getpid()
        return {"n_samples": len(used), "used_mb_median": used[len(used) // 2] if used else None,
                "used_mb_max": used[-1] if used else None, "total_mb": self.samples[0]["total_mb"] if self.samples else None,
                "util_median": sorted(s["util"] for s in self.samples)[len(used) // 2] if used else None,
                "other_processes_mb": {str(p): m for p, m in self.apps.items() if p != me}}


def pick_rows(path: Path, n: int) -> list:
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    step = max(1, len(rows) // n)
    return rows[::step][:n]


def probe_fold(run_fmt: str, model_id: str, quant: str, fold: int, n: int, batch_size: int) -> dict:
    import torch

    from app.demo import LiveModel
    from src.vlm.infer import generate

    rows = pick_rows(ROOT / "data" / "sft" / f"m2_salad_f{fold}" / "heldout.jsonl", n)
    live = LiveModel(run_fmt, model_id, quant)
    t_load = time.time()
    model, proc = live.get(fold)
    t_load = time.time() - t_load
    torch.cuda.reset_peak_memory_stats()
    sampler = GpuSampler()
    sampler.start()
    marks = []
    t0 = time.time()
    preds = generate(model, proc, rows, batch_size=batch_size, on_batch=lambda i, k: marks.append(time.time()))
    total = time.time() - t0
    marks.append(time.time())
    gpu = sampler.stop()
    batch_s = [b - a for a, b in zip(marks, marks[1:])]
    n_out = sum(len(p) for p in preds)
    res = {"run": run_fmt.format(f=fold), "fold": fold, "n": len(rows), "batch_size": batch_size,
           "load_s": round(t_load, 1), "total_s": round(total, 1), "sec_per_image": round(total / len(rows), 2),
           "sec_per_image_excl_first_batch": round(sum(batch_s[1:]) / max(len(rows) - batch_size, 1), 2) if len(batch_s) > 1 else None,
           "batch_seconds": [round(b, 1) for b in batch_s], "mean_output_chars": round(n_out / len(rows), 1),
           "labels": {k: sum(r["label"] == k for r in rows) for k in ("good", "logical", "structural")},
           "torch_max_reserved_gb": round(torch.cuda.max_memory_reserved() / 1024**3, 2),
           "torch_max_allocated_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2), "gpu": gpu}
    del model, proc, live
    gc.collect()
    torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="p2_8b_f{f}")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--quant", default="nf4", choices=["none", "nf4"])
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--label", default="idle")
    a = ap.parse_args()
    out = ROOT / "reports" / "ops" / f"gen_probe_{a.label}.json"
    res = {"label": a.label, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "args": vars(a),
           "gpu_before": nvidia_smi(["--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader"]).strip(), "folds": []}
    for f in a.folds:
        r = probe_fold(a.run, a.model, a.quant, f, a.n, a.batch_size)
        res["folds"].append(r)
        print(f"fold {f}: {r['sec_per_image']}초/장 (첫 배치 제외 {r['sec_per_image_excl_first_batch']}), "
              f"GPU 사용 중앙값 {r['gpu']['used_mb_median']}MB / 최대 {r['gpu']['used_mb_max']}MB, "
              f"다른 프로세스 {r['gpu']['other_processes_mb']}", flush=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    res["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
