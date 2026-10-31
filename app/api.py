"""FastAPI 시연 서버: 면접장 데모용 REST API + 단일 페이지 화면.

캐시 모드(기본)는 GPU 없이 노트북에서 돈다. held-out 이미지마다 그 이미지를 학습하지 않은 fold 모델(Ours 4B seed 0)의
저장된 예측, SALAD 캐스케이드 단계(tau_low 학습 측 재현율 0.95), 비-VLM 기준선 B1을 같은 캐스케이드에 넣은 출력, 정답을
JSON으로 돌려주고 `/`의 화면이 이를 나란히 보여 준다. --live를 주면 같은 4B QLoRA 어댑터를 올려 선택한 이미지를 실제로
생성한다(VRAM 약 6GB, 한 번에 한 요청).

    python -m app.api                      # http://127.0.0.1:8008  (화면: /, 문서: /docs)
    python -m app.api --live               # 실시간 생성 버튼 활성화
    python -m app.api --port 8010 --open   # 포트 지정, 기본 브라우저 열기

주요 엔드포인트
    GET  /api/meta                         카테고리·라벨 목록, 표1 요약 수치, 라이브 가능 여부
    GET  /api/images?category=&label=      이미지 목록(id, 라벨, 단계, Ours/B1 판정)
    GET  /api/predict/{image_id}           한 이미지의 탐지기 점수·단계·Ours·B1·정답·입력 이미지 URL
    GET  /api/image/{image_id}?kind=overlay|inspect|crop|ref   PNG
    POST /api/generate/{image_id}          --live일 때 VLM 실시간 생성(원출력, Guard 후, 소요 시간)
"""

import argparse
import io
import json
import os
import threading
import time
import webbrowser
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response
from PIL import Image

from app.demo import LABEL_KO, LiveModel, STAGE_KO, load_state, verdict
from src.eval.cascade import group_correct
from src.kb import load_kb
from src.output_spec import is_normal
from src.report.m4_figures import CAT_KO, overlay
from src.vlm.prompts import IMAGE_ROLES

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "app" / "static"
TABLES = ROOT / "reports" / "tables"
STAGE_SHORT = {"pass": "통과", "judge": "판정 구간", "describe": "설명 구간"}
IMAGE_KINDS = {"inspect": 0, "crop": 1, "ref": 2}


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def summary_numbers() -> dict:
    """표1·캐스케이드·탐지기 결과 파일에서 화면 상단 요약 수치를 모은다(파일이 없으면 빈 값)."""
    t1 = _load_json(TABLES / "table1_main.json")
    cas = _load_json(TABLES / "m3_cascade.json")
    out = {}

    def grab(name: str, key: str):
        row = t1.get(name) or {}
        if row:
            out[f"{key}_group"] = row["metrics"]["group_acc"]
            out[f"{key}_e2e"] = row["end_to_end_group_f1"]["test"]
            out[f"{key}_e2e_pi1"] = row["end_to_end_group_f1"]["pi=0.01"]
            out[f"{key}_sec_per_image"] = (row.get("cost") or {}).get("sec_per_image")

    grab("Ours 4B", "ours")
    grab("B1 비-VLM", "b1")
    grab("P3: 4B + B1 후보 힌트", "p3")
    out["detector_latency_ms"] = (t1.get("detector") or {}).get("latency_ms_median")
    aucs = []
    det_dir = ROOT / "outputs" / "detector" / "salad"
    if det_dir.exists():
        for c in sorted(det_dir.iterdir()):
            m = _load_json(c / "meta.json")
            if isinstance(m.get("official_auroc"), dict) and "auc" in m["official_auroc"]:
                aucs.append(m["official_auroc"]["auc"])
    out["detector_auroc"] = sum(aucs) / len(aucs) / 100 if aucs else None
    c95 = (cas.get("by_recall_target") or {}).get("0.95") or {}
    if c95:
        out["cascade"] = {"recall": c95["cascade"]["recall"], "fpr": c95["cascade"]["fpr"],
                          "detector_recall": c95["detector_only"]["recall"], "detector_fpr": c95["detector_only"]["fpr"],
                          "false_alarm_reduction": c95["false_alarm_reduction"],
                          "vlm_call_rate_pi1": (c95.get("vlm_call_rate") or {}).get("pi=0.01")}
    return out


class State:
    """서버가 시작할 때 한 번 읽는 결과 캐시."""

    def __init__(self, live: LiveModel | None = None):
        self.df, self.recs, self.b1 = load_state()
        self.idx = pd.read_parquet(ROOT / "data" / "index.parquet").set_index("image_id")
        self.fold_of = pd.read_csv(ROOT / "data" / "splits.csv").set_index("image_id")["fold"]
        self.kb = load_kb()
        self.live = live
        self.lock = threading.Lock()
        self.summary = summary_numbers()
        self._overlay: dict[str, bytes] = {}

    def overlay_png(self, image_id: str) -> bytes:
        if image_id not in self._overlay:
            arr = overlay(image_id, self.fold_of, self.idx)
            buf = io.BytesIO()
            Image.fromarray(arr).save(buf, format="PNG")
            self._overlay[image_id] = buf.getvalue()
        return self._overlay[image_id]

    def row(self, image_id: str):
        if image_id not in self.df.index:
            raise HTTPException(404, f"모르는 image_id: {image_id}")
        return self.df.loc[image_id], self.recs[image_id]

    def predict(self, image_id: str) -> dict:
        r, rec = self.row(image_id)
        det = rec.get("detector") or {}
        b1 = self.b1.loc[image_id] if image_id in self.b1.index else None
        return {
            "id": image_id, "category": r["category"], "category_ko": CAT_KO[r["category"]],
            "label": r["label"], "label_ko": LABEL_KO[r["label"]], "model_fold": rec["model_fold"],
            "detector": {"percentile": det.get("percentile"), "location": det.get("location")},
            "stage": r["stage"], "stage_ko": STAGE_KO[r["stage"]], "stage_short": STAGE_SHORT[r["stage"]],
            "ours": {"final": r["final"], "raw": r["pred"], "verdict": verdict(r)},
            "b1": None if b1 is None else {"final": b1["final"], "raw": b1["pred"], "verdict": verdict(b1)},
            "gt": r["gt_report"] if r["label"] != "good" else "",
            "images": [{"role": role, "url": f"/api/image/{image_id}?kind={kind}"}
                       for role, kind in zip(IMAGE_ROLES, IMAGE_KINDS)],
            "overlay_url": f"/api/image/{image_id}?kind=overlay",
            "live": self.live is not None,
        }


def create_app(state: State) -> FastAPI:
    app = FastAPI(title="LOCO 이상 리포트 시연 API", version="1.0",
                  description="SALAD 탐지기 + QLoRA Qwen3-VL 2줄 이상 리포트. held-out 이미지의 저장된 예측을 돌려준다.")

    @app.get("/")
    def page():
        return FileResponse(STATIC / "demo.html", media_type="text/html")

    @app.get("/api/health")
    def health():
        return {"ok": True, "images": int(len(state.df)), "live": state.live is not None}

    @app.get("/api/meta")
    def meta():
        counts = state.df["label"].value_counts().to_dict()
        cats = sorted(state.df["category"].unique())
        return {"categories": [{"code": c, "ko": CAT_KO[c]} for c in cats],
                "labels": [{"code": k, "ko": v} for k, v in LABEL_KO.items()],
                "n_images": int(len(state.df)), "counts": {k: int(v) for k, v in counts.items()},
                "live": state.live is not None, "summary": state.summary}

    @app.get("/api/images")
    def images(category: str = Query(""), label: str = Query("")):
        sel = state.df
        if category:
            sel = sel[sel["category"] == category]
        if label:
            sel = sel[sel["label"] == label]
        rows = []
        for iid, r in sel.sort_index().iterrows():
            b1v = verdict(state.b1.loc[iid]) if iid in state.b1.index else None
            rows.append({"id": iid, "label": r["label"], "stage": r["stage"], "ours_verdict": verdict(r), "b1_verdict": b1v})
        return rows

    @app.get("/api/predict/{image_id:path}")
    def predict(image_id: str):
        return state.predict(image_id)

    @app.get("/api/image/{image_id:path}")
    def image(image_id: str, kind: str = Query("overlay")):
        _, rec = state.row(image_id)
        if kind == "overlay":
            return Response(state.overlay_png(image_id), media_type="image/png")
        if kind not in IMAGE_KINDS:
            raise HTTPException(400, f"kind는 overlay|inspect|crop|ref 중 하나: {kind}")
        path = ROOT / rec["images"][IMAGE_KINDS[kind]]
        if not path.exists():
            raise HTTPException(404, f"이미지 파일 없음: {path.name}")
        return FileResponse(path, media_type="image/png")

    @app.post("/api/generate/{image_id:path}")
    def generate_live(image_id: str):
        if state.live is None:
            raise HTTPException(503, "실시간 생성은 --live로 서버를 띄웠을 때만 쓸 수 있다.")
        from src.vlm.guard import guard
        from src.vlm.infer import generate

        r, rec = state.row(image_id)
        with state.lock:  # 어댑터 교체와 생성은 한 번에 하나
            model, proc = state.live.get(rec["model_fold"])
            t0 = time.time()
            raw = generate(model, proc, [rec], batch_size=1)[0]
            seconds = time.time() - t0
        out, steps = guard(raw, state.kb, rec["category"], (rec.get("detector") or {}).get("location"))
        if r["label"] == "good":
            correct = is_normal(out)
        else:
            correct = (not is_normal(out)) and group_correct(out, r["gt_report"])
        return {"id": image_id, "raw": raw, "guarded": out, "guard_steps": steps, "seconds": seconds,
                "model_fold": rec["model_fold"], "correct": bool(correct),
                "note": "캐스케이드 단계와 무관하게 VLM을 직접 호출한 결과"}

    return app


def main():
    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--run", default="m2_4b_f{f}_s0", help="--live 어댑터 run 이름 형식")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--quant", default="nf4", choices=["none", "nf4"])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8008)))
    ap.add_argument("--open", action="store_true", help="기본 브라우저로 화면을 연다")
    a = ap.parse_args()
    live = LiveModel(a.run, a.model, a.quant) if a.live else None
    app = create_app(State(live))
    url = f"http://{a.host}:{a.port}/"
    print(f"시연 화면: {url}   API 문서: {url}docs   {'실시간 생성 켜짐' if live else '캐시 모드(GPU 불필요)'}")
    if a.open:
        threading.Timer(1.0, webbrowser.open, args=(url,)).start()
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
