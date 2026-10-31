"""시연 영상(MP4) 녹화: 면접장에서 서버를 띄울 수 없을 때 트는 대체물.

FastAPI 시연 서버(app/api.py, 캐시 모드)를 이 프로세스 안에서 띄우고, Edge 헤드리스를 CDP로 조종해 사례 페이지를
차례로 찍은 뒤 OpenCV로 MP4를 만든다. 소리 없음. 위쪽 띠에 사례 설명을 쓰고, 앞뒤에 제목 카드와 결과 요약 카드를 둔다.
사례는 GIF(app/record_gif.py)와 같은 목록(그림 4의 성공 3, 실패 3에 정상 1)이다.

    python -m app.record_video                 # reports/figures/demo.mp4 (약 1분)
    python -m app.record_video --hold 10       # 사례당 표시 시간(초)
"""

import argparse
import base64
import io
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app.api import State, create_app
from app.record_gif import CDP, FONT, cases, free_port, get_json
from src.report.build_portfolio import EDGE

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "figures" / "demo.mp4"
W, H = 1280, 820  # 브라우저 뷰포트
BANNER_H = 48
FPS = 8
FONT_REG = "C:/Windows/Fonts/malgun.ttf"
BG = "#1f2937"


def pct(v) -> str:
    return "-" if v is None else f"{100 * v:.1f}"


def serve(app, port: int) -> None:
    import uvicorn

    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    get_json(f"http://127.0.0.1:{port}/api/health")


def banner(img: Image.Image, text: str) -> Image.Image:
    out = Image.new("RGB", (W, H + BANNER_H), BG)
    out.paste(img.resize((W, H), Image.LANCZOS), (0, BANNER_H))
    ImageDraw.Draw(out).text((18, 11), text, fill="white", font=ImageFont.truetype(FONT, 22))
    return out


def wrap(d: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, width: int) -> list[str]:
    """폭을 넘지 않게 어절 단위로 줄을 나눈다(빈 줄은 그대로)."""
    if not text:
        return [""]
    out, cur = [], ""
    for word in text.split(" "):
        cand = word if not cur else f"{cur} {word}"
        if d.textlength(cand, font=font) <= width or not cur:
            cur = cand
        else:
            out.append(cur)
            cur = word
    out.append(cur)
    return out


def text_card(title: str, lines: list[str], foot: str = "") -> Image.Image:
    """어두운 배경의 제목·본문 카드(영상 앞뒤). 긴 줄은 자동으로 접는다."""
    img = Image.new("RGB", (W, H + BANNER_H), BG)
    d = ImageDraw.Draw(img)
    margin, width = 80, W - 160
    tf, bf = ImageFont.truetype(FONT, 38), ImageFont.truetype(FONT_REG, 24)
    y = 100
    for line in wrap(d, title, tf, width):
        d.text((margin, y), line, fill="white", font=tf)
        y += 54
    y += 30
    for line in lines:
        for seg in wrap(d, line, bf, width):
            d.text((margin, y), seg, fill="#e5e7eb", font=bf)
            y += 42
        y += 6
    if foot:
        ff = ImageFont.truetype(FONT_REG, 19)
        yf = H - 40
        for seg in wrap(d, foot, ff, width):
            d.text((margin, yf), seg, fill="#9ca3af", font=ff)
            yf += 26
    return img


def capture(cdp: CDP, url: str, iid: str) -> Image.Image:
    cdp.call("Page.navigate", url=url)
    ready = f"document.body.dataset.ready === '1' && document.body.innerText.includes({iid!r})"
    t0 = time.time()
    while not cdp.eval(ready):
        if time.time() - t0 > 60:
            sys.exit(f"페이지 로드 시간 초과: {iid}")
        time.sleep(0.3)
    time.sleep(0.8)
    png = base64.b64decode(cdp.call("Page.captureScreenshot", format="png")["data"])
    return Image.open(io.BytesIO(png)).convert("RGB")


def summary_card(s: dict) -> Image.Image:
    c = s.get("cascade") or {}
    lines = [
        f"탐지기 SALAD: image AUROC {pct(s.get('detector_auroc'))}, {s.get('detector_latency_ms', 0):.0f}ms/장 (공식 가중치 재현)",
        f"캐스케이드(재현율 목표 0.95): 재현율 {pct(c.get('recall'))} / FPR {pct(c.get('fpr'))}, "
        f"탐지기 단독 {pct(c.get('detector_recall'))} / {pct(c.get('detector_fpr'))} → 오경보 {100 * (c.get('false_alarm_reduction') or 0):.0f}% 감소",
        f"VLM 호출률 {pct(c.get('vlm_call_rate_pi1'))} (이상 비율 1% 가정), 생성 {s.get('ours_sec_per_image') or 0:.1f}초/장 (RTX 3060 12GB)",
        f"결함 그룹 정확도: Ours 4B QLoRA {pct(s.get('ours_group'))}  <  B1 비-VLM 기준선 {pct(s.get('b1_group'))}"
        + (f"  <  P3 4B + B1 힌트 {pct(s.get('p3_group'))}" if s.get("p3_group") is not None else ""),
        f"종단 그룹 F1(탐지+진단): Ours {pct(s.get('ours_e2e'))}, B1 {pct(s.get('b1_e2e'))}"
        + (f", P3 {pct(s.get('p3_e2e'))}" if s.get("p3_e2e") is not None else ""),
        "",
        "결론: 비-VLM 분류기가 더 정확함(가설 기각). 용량(8B), 라벨 수, 입력 표현 실험으로 원인 분해. "
        "실용 구성은 '판정은 단순 분류기, 설명과 검증은 VLM'인 계층. 예측 원본과 어댑터 공개로 GPU 없이 표 재계산 가능.",
    ]
    return text_card("결과 요약 (두 held-out fold 합산, 이상 993 / 정상 575)", lines,
                     "수치 출처: reports/tables/table1_main.md, m3_cascade.md. 자세한 내용은 README와 포트폴리오 PDF.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--hold", type=float, default=7.0, help="사례당 표시 시간(초)")
    ap.add_argument("--fps", type=int, default=FPS)
    a = ap.parse_args()

    state = State(None)
    port = free_port()
    serve(create_app(state), port)
    todo = cases(state.df)
    edge = next((p for p in EDGE if p.exists()), None)
    if edge is None:
        sys.exit("Edge를 찾지 못함")
    tmp = Path(tempfile.mkdtemp(prefix="loco_video_"))
    dbg = free_port()
    proc = subprocess.Popen([str(edge), "--headless=new", "--disable-gpu", f"--remote-debugging-port={dbg}",
                             f"--user-data-dir={tmp / 'profile'}", f"--window-size={W},{H}", "about:blank"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    clips: list[tuple[Image.Image, float]] = []
    try:
        get_json(f"http://127.0.0.1:{dbg}/json/version")
        page = next(t for t in get_json(f"http://127.0.0.1:{dbg}/json/list") if t["type"] == "page")
        cdp = CDP(page["webSocketDebuggerUrl"])
        cdp.call("Page.enable")
        cdp.call("Emulation.setDeviceMetricsOverride", width=W, height=H, deviceScaleFactor=1, mobile=False)
        clips.append((text_card(
            "MVTec LOCO AD: 탐지기 + LoRA 적응 VLM 2줄 이상 리포트",
            ["1. SALAD(비지도 탐지기)가 이미지의 이상 점수를 산출",
             "2. 점수에 따라 통과(무출력) / 판정 구간(VLM이 정상·이상 판정) / 설명 구간(VLM 설명)으로 분기",
             "3. QLoRA로 적응시킨 Qwen3-VL-4B가 '이상: 결함 그룹 - 세부 (위치) / 조치: ...' 2줄만 출력, Guard가 KB와 대조",
             "",
             "화면: held-out 이미지만. 각 이미지는 학습에 쓰지 않은 fold 모델의 예측. Ours(VLM)와 B1(비-VLM 기준선)을 같은 "
             "캐스케이드에 넣은 최종 출력을 정답과 나란히 표시. 사례 7개는 규칙으로 정한 목록(성공 3, 정상 1, 실패 3)."],
            "같은 화면: python -m app.api (GPU 불필요). --live면 선택한 이미지를 실시간 생성."), 6.0))
        for k, (iid, label) in enumerate(todo, 1):
            p = state.predict(iid)
            b1v = p["b1"]["verdict"] if p["b1"] else "-"
            cap = (f"{k}/{len(todo)}  {label}   |   {p['category_ko']} · {p['label_ko']} · {p['stage_short']}"
                   f"   |   Ours: {p['ours']['verdict']}   B1: {b1v}")
            shot = capture(cdp, f"http://127.0.0.1:{port}/?id={iid}", iid)
            clips.append((banner(shot, cap), a.hold))
            print(f"{k}/{len(todo)} {iid}")
        clips.append((summary_card(state.summary), 9.0))
    finally:
        proc.terminate()
        proc.wait(timeout=30)
        shutil.rmtree(tmp, ignore_errors=True)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (W, H + BANNER_H))
    total = 0.0
    for img, sec in clips:
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        for _ in range(int(round(sec * a.fps))):
            writer.write(frame)
        total += sec
    writer.release()
    print(f"저장: {out} ({total:.0f}초, {len(clips)}장면, {out.stat().st_size / 1e6:.1f}MB)")


if __name__ == "__main__":
    main()
