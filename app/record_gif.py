"""데모 GIF 녹화 (계획 v5 Phase 4: README 상단 데모 GIF).

캐시 모드 데모를 이 프로세스 안에서 띄우고, Edge 헤드리스를 원격 디버깅 프로토콜(CDP)로 조종해 사례 페이지
(`?id=<image_id>`)를 차례로 찍은 뒤 GIF로 묶는다. 새 설치 없이 Windows 기본 Edge와 websocket-client만 쓴다.

사례는 결과를 보고 고르지 않도록 그림 4와 같은 목록(`reports/tables/figure_cases.json`: 성공 3, 실패 3)에
정상 1장(정상 held-out 중 카테고리 순·id 순으로 첫 무출력 이미지)을 더한다.

    python -m app.record_gif        # reports/figures/demo.gif
"""

import base64
import io
import json
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import websocket
from PIL import Image, ImageDraw, ImageFont

from app.demo import build, load_state
from src.report.build_portfolio import EDGE
from src.report.m4_figures import CAT_KO

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "figures" / "demo.gif"
VIEW_W = 1280  # 브라우저 창 너비(px)
GIF_W = 960  # GIF 너비
FRAME_MS = 4000  # 사례당 표시 시간
FONT = "C:/Windows/Fonts/malgunbd.ttf"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def get_json(url: str, timeout: float = 60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                return json.loads(r.read() or b"null")
        except Exception:
            time.sleep(0.5)
    sys.exit(f"응답 없음: {url}")


class CDP:
    """최소 CDP 클라이언트: 요청 id로 응답을 기다리고 이벤트는 버린다."""

    def __init__(self, ws_url: str):
        # Origin 헤더가 있으면 Edge가 원격 디버깅 연결을 거부하므로 보내지 않는다
        self.ws = websocket.create_connection(ws_url, timeout=60, suppress_origin=True)
        self.n = 0

    def call(self, method: str, **params) -> dict:
        self.n += 1
        self.ws.send(json.dumps({"id": self.n, "method": method, "params": params}))
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get("id") == self.n:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})

    def eval(self, expr: str):
        return self.call("Runtime.evaluate", expression=expr, returnByValue=True)["result"].get("value")


def cases(df) -> list[tuple[str, str]]:
    """(image_id, 띠 문구). 성공 3 -> 정상 1 -> 실패 3."""
    fig = json.loads((ROOT / "reports" / "tables" / "figure_cases.json").read_text(encoding="utf-8"))
    good = df[(df["label"] == "good") & (df["stage"] == "pass")].index
    normal = sorted(good, key=lambda i: (df.loc[i, "category"], i))[0]
    succ = [(c["id"], c["label"]) for c in fig if c["kind"] == "success"]
    fail = [(c["id"], c["label"]) for c in fig if c["kind"] == "fail"]
    return succ + [(normal, "정상: 탐지기 통과, 무출력")] + fail


def banner(img: Image.Image, text: str) -> Image.Image:
    """위쪽 띠에 사례 설명을 쓴다."""
    h = 44
    out = Image.new("RGB", (img.width, img.height + h), "#1f2937")
    out.paste(img, (0, h))
    ImageDraw.Draw(out).text((16, 8), text, fill="white", font=ImageFont.truetype(FONT, 22))
    return out


def capture(cdp: CDP, url: str, iid: str) -> Image.Image:
    cdp.call("Page.navigate", url=url)
    # 정보 문단에 id가 나오고 검사 이미지 1장 + 입력 갤러리 3장이 모두 로드될 때까지
    ready = ("(() => { const im = [...document.images].filter(i => i.src.includes('/file=') || i.src.includes('/gradio_api/'));"
             f" return document.body.innerText.includes({json.dumps(iid)}) && im.length >= 4"
             " && im.every(i => i.complete && i.naturalWidth > 0); })()")
    t0 = time.time()
    while not cdp.eval(ready):
        if time.time() - t0 > 60:
            sys.exit(f"페이지 로드 시간 초과: {iid}")
        time.sleep(0.3)
    time.sleep(1.0)  # 레이아웃 안정화
    h = int(cdp.eval("document.documentElement.scrollHeight"))
    cdp.call("Emulation.setDeviceMetricsOverride", width=VIEW_W, height=h, deviceScaleFactor=1, mobile=False)
    time.sleep(0.5)
    png = base64.b64decode(cdp.call("Page.captureScreenshot", format="png")["data"])
    return Image.open(io.BytesIO(png)).convert("RGB")


def main():
    df, _, _ = load_state()
    todo = cases(df)
    port = free_port()
    demo = build(None)
    demo.launch(server_name="127.0.0.1", server_port=port, prevent_thread_lock=True, quiet=True)
    edge = next((p for p in EDGE if p.exists()), None)
    if edge is None:
        sys.exit("Edge를 찾지 못함")
    tmp = Path(tempfile.mkdtemp(prefix="loco_gif_"))
    dbg = free_port()
    proc = subprocess.Popen([str(edge), "--headless=new", "--disable-gpu", f"--remote-debugging-port={dbg}",
                             f"--user-data-dir={tmp / 'profile'}", f"--window-size={VIEW_W},900", "about:blank"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        get_json(f"http://127.0.0.1:{dbg}/json/version")
        page = next(t for t in get_json(f"http://127.0.0.1:{dbg}/json/list") if t["type"] == "page")
        cdp = CDP(page["webSocketDebuggerUrl"])
        cdp.call("Page.enable")
        cdp.call("Emulation.setDeviceMetricsOverride", width=VIEW_W, height=900, deviceScaleFactor=1, mobile=False)
        frames = []
        for k, (iid, label) in enumerate(todo, 1):
            # 테마를 고정해 시스템 설정과 무관하게 같은 GIF가 나오게 한다
            shot = capture(cdp, f"http://127.0.0.1:{port}/?id={iid}&__theme=dark", iid)
            shot = shot.resize((GIF_W, round(shot.height * GIF_W / shot.width)), Image.LANCZOS)
            frames.append(banner(shot, f"{k}/{len(todo)}  {CAT_KO[df.loc[iid, 'category']]} - {label}"))
            print(f"{k}/{len(todo)} {iid}")
    finally:
        proc.terminate()
        proc.wait(timeout=30)
        demo.close()
        shutil.rmtree(tmp, ignore_errors=True)
    # 프레임 높이를 맞춘 뒤(아래를 배경색으로 채움) 프레임별 팔레트로 양자화
    h = max(f.height for f in frames)
    pal = []
    for f in frames:
        canvas = Image.new("RGB", (GIF_W, h), "white")
        canvas.paste(f, (0, 0))
        pal.append(canvas.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE))
    pal[0].save(OUT, save_all=True, append_images=pal[1:], duration=FRAME_MS, loop=0, optimize=True)
    print(f"저장: {OUT} ({len(frames)}프레임, {OUT.stat().st_size / 1e6:.1f}MB)")


if __name__ == "__main__":
    main()
