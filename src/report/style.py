"""그림 공용 스타일: 한글 폰트(맑은 고딕)와 색 역할.

색은 범주 순서를 고정해 쓴다(순환 금지). 검증: dataviz validate_palette.js로 3색 모두 통과
(인접 CVD ΔE 9.2 이상). 청록(SERIES[2])은 밝은 바탕 대비가 3:1 미만이라 쓸 때 직접 라벨을 붙인다.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

FONT_PATHS = [Path("C:/Windows/Fonts/malgun.ttf"), Path("C:/Windows/Fonts/malgunbd.ttf")]
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # 파랑, 주황, 청록 (고정 순서)
NORMAL_COLOR, ANOMALY_COLOR = SERIES[0], SERIES[1]
FIG_DIR = Path(__file__).resolve().parents[2] / "reports" / "figures"


FONT_CANDIDATES = ["Malgun Gothic", "Noto Sans CJK KR", "NanumGothic", "DejaVu Sans"]


def pick_font() -> str:
    """설치된 첫 후보 폰트. 맑은 고딕이 없는 환경(Linux CI)에서는 경고 없이 다음 후보로 떨어진다."""
    for p in FONT_PATHS:
        if p.exists():
            font_manager.fontManager.addfont(str(p))
    names = {f.name for f in font_manager.fontManager.ttflist}
    return next((n for n in FONT_CANDIDATES if n in names), FONT_CANDIDATES[-1])


def setup() -> None:
    plt.rcParams.update({
        "font.family": pick_font(),
        "axes.unicode_minus": False,  # 한글 폰트에서 음수 기호 깨짐 방지
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "text.color": TEXT, "axes.labelcolor": TEXT_2, "xtick.color": TEXT_2, "ytick.color": TEXT_2,
        "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.axisbelow": True,  # 격자선을 막대·선 뒤로
        "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 2.0, "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
        "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight",
    })


def save(fig, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / name
    fig.savefig(out)
    plt.close(fig)
    return out
