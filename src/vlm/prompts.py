"""VLM 입력 구성: 이미지 크기 예산과 사용자 프롬프트 (계획 v5 Phase 2 입력 명세).

이미지 3장: 검사 이미지, 의심 영역 확대(crop), 정상 참조. 각 이미지는 종횡비를 유지해 약 256토큰
(Qwen3-VL: 16px 패치 x 2x2 병합 = 토큰당 32px)으로 미리 줄여 캐시한다.
"""

import math

from src.output_spec import CELLS, LOC_GLOBAL, NORMAL, UNKNOWN_LABEL

TOKEN_PX = 32  # patch_size 16 x merge_size 2
DEFAULT_TOKENS = 256

IMAGE_ROLES = ("검사 이미지", "의심 영역 확대(윤곽선 표시)", "정상 참조")


def image_size_for(w: int, h: int, tokens: int = DEFAULT_TOKENS) -> tuple[int, int]:
    """종횡비를 유지하며 토큰 수가 tokens에 가깝도록 32px 배수 크기를 고른다."""
    s = math.sqrt(tokens * TOKEN_PX * TOKEN_PX / (w * h))
    W = max(TOKEN_PX, round(w * s / TOKEN_PX) * TOKEN_PX)
    H = max(TOKEN_PX, round(h * s / TOKEN_PX) * TOKEN_PX)
    while (W // TOKEN_PX) * (H // TOKEN_PX) > tokens * 1.1:
        if W >= H:
            W -= TOKEN_PX
        else:
            H -= TOKEN_PX
    return W, H


TERM_KO = (("local", "국소"), ("mah", "전역"), ("comp", "구성"))  # SALAD 세 항: 패치 국소 / Mahalanobis 전역 / 부품 구성


def detector_line(det: dict | None) -> str | None:
    """det["terms"]({local, mah, comp}: 학습 측 정상 대비 백분위)가 있으면 세 항도 적는다 (P3 변형 A)."""
    if not det:
        return None
    parts = [f"이상 점수 백분위 {det['percentile']:.0f} (정상 이미지 대비)"]
    if det.get("terms"):
        parts.append("세부 백분위 " + " / ".join(f"{ko} {det['terms'][k]:.0f}" for k, ko in TERM_KO))
    if det.get("location"):
        parts.append(f"의심 위치: {det['location']}")
    return "[탐지기] " + ", ".join(parts)


def hint_line(hints: list | None) -> str | None:
    """P3 변형 B: 비-VLM 분류기(B1)의 상위 그룹 후보와 확률. [(그룹 ko_label 또는 정상, 확률), ...]"""
    if not hints:
        return None
    return "[분류기 후보] " + ", ".join(f"{g} {p:.2f}" for g, p in hints)


def user_text(kb: dict, category: str, groups_shown: list, det: dict | None = None,
              roles: tuple = IMAGE_ROLES, spec: bool = True, hints: list | None = None) -> str:
    """사용자 프롬프트. groups_shown은 목록에 보일 그룹 ko_label (목록 드롭아웃이면 정답 그룹이 빠짐).
    roles는 실제로 넣는 이미지의 역할 (입력 ablation에서 정상 참조나 crop을 빼면 줄어듦).
    spec=False면 정상 규격 줄을 뺀다 (규격 텍스트 ablation). hints는 B1 후보 줄 (P3 변형 B)."""
    c = kb[category]
    lines = [f"[제품] {c['ko_name']}"]
    if spec:
        lines.append(f"[정상 규격] {c['spec']}")
    lines.append("[결함 그룹 목록] " + " / ".join(groups_shown))
    dl = detector_line(det)
    if dl:
        lines.append(dl)
    hl = hint_line(hints)
    if hl:
        lines.append(hl)
    lines += [
        "[이미지] " + ", ".join(f"{i + 1}: {r}" for i, r in enumerate(roles)),
        "[출력 규칙]",
        f'정상이면 "{NORMAL}" 한 단어만 출력한다.',
        "이상이면 아래 2줄만 출력한다.",
        "이상: [논리|구조] <결함 그룹>[ - <세부 결함>][ (<위치>)]",
        "조치: <조치>",
        f"목록에 없는 결함이면 그룹 자리에 \"{UNKNOWN_LABEL}\"을 쓰고 세부 결함 자리에 20자 이내로 서술한다.",
        "위치: " + ", ".join(CELLS) + f", {LOC_GLOBAL} 중 하나. 같은 부품 여러 개가 모두 해당하면 생략.",
    ]
    return "\n".join(lines)


def messages(text: str, n_images: int = len(IMAGE_ROLES), target: str | None = None) -> list:
    content = [{"type": "image"} for _ in range(n_images)] + [{"type": "text", "text": text}]
    msgs = [{"role": "user", "content": content}]
    if target is not None:
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": target}]})
    return msgs
