"""SALAD 배포 가중치 점검·복구.

공식 salad_loco.zip(2025-10 확인)의 breakfast_box/autoencoder_final.pth에는 모델 기록(autoencoder_final/) 외에
다른 카테고리 가중치 파일 24개가 같은 zip 안에 덧붙어 있어(1.2GB) torch.load가 실패한다.
CRC는 모두 정상이라, 파일 이름과 같은 접두사 아래 기록만 남겨 다시 묶으면 정상 로드된다(4.4MB, 다른 카테고리와 같은 구조).
원본은 <파일>.orig로 보존한다.

사용법 (SALAD 전용 가상환경):
    E:\\venvs\\loco_salad\\Scripts\\python.exe -m src.detector.fix_salad_weights
"""

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "external" / "SALAD" / "results"


def check_and_fix(path: Path) -> str:
    prefix = path.stem + "/"
    with zipfile.ZipFile(path) as z:
        infos = z.infolist()
        extra = [i for i in infos if not i.filename.startswith(prefix)]
        if not extra:
            return "정상"
        bad = z.testzip()
        if bad is not None:
            raise RuntimeError(f"{path}: CRC 오류 {bad}")
        keep = [(i.filename, z.read(i)) for i in infos if i.filename.startswith(prefix)]
    orig = path.with_name(path.name + ".orig")
    path.rename(orig)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as out:
        for name, data in keep:
            out.writestr(name, data)
    return f"복구: 덧붙은 항목 {len(extra)}개 제거, 원본 {orig.name}"


def main():
    for p in sorted(RESULTS.glob("*/*_final.pth")):
        print(p.relative_to(RESULTS), check_and_fix(p))


if __name__ == "__main__":
    main()
