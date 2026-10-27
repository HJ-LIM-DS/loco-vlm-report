"""LOCO VLM 포트폴리오 패키지.

Windows에서 torch를 먼저 불러온 뒤 pyarrow.dataset을 불러오면(transformers.Trainer -> datasets 경로)
DLL 충돌로 접근 위반(0xC0000005, traceback 없이 종료)이 난다. pyarrow를 torch보다 먼저 불러와 피한다.
SALAD 전용 가상환경처럼 pyarrow가 없는 환경에서는 건너뛴다.
"""

try:
    import pyarrow.dataset  # noqa: F401
except ImportError:
    pass
