"""P3 2단계 자동 선별: 1단계(2B 선별) 결과로 4B 확인 대상을 고르고 대기열을 그에 맞게 바꾼다 (대기열 task, 사람 개입 없음).

사전 규칙(계획 문서 "2차 개선 1"): 그룹 전체 효과가 양수인 변형 중 가장 큰 1개를 4B 2-fold로 확인. 둘 다 0 이하이면 생략.
- 선정된 변형의 fold 1 SFT 데이터를 만들고(fold 0은 1단계에서 이미 있음) 표식 reports/tables/p3_selected_<key>.ok를 쓴다.
- configs/run_queue.yaml에서 선정되지 않은 변형의 4B run 줄(`enabled: true   # p3_select:<key>`)을 enabled: false로 바꾼다.
  감독자는 대기열을 매 루프 다시 읽으므로 그대로 이어진다. 주석을 지키기 위해 YAML을 다시 쓰지 않고 그 줄만 바꾼다.
- 결과는 reports/tables/p3_selected.json에 남긴다. 다시 실행해도 같은 결과를 낸다.

    python -m src.report.p3_select
"""

import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
QUEUE = ROOT / "configs" / "run_queue.yaml"
TABLES = ROOT / "reports" / "tables"
KEYS = ("terms", "hints")
SFT_ARGS = {"terms": ["--det-terms", "--name", "p3_terms"],
            "hints": ["--b1-hints", "outputs/baselines/b1_f1_probs.jsonl", "--name", "p3_hints"]}


def set_enabled(text: str, key: str, enabled: bool) -> str:
    """`enabled: <값>   # p3_select:<key>` 표식 줄만 바꾼다."""
    pat = re.compile(rf"^(\s*enabled:\s*)(true|false)(\s*# p3_select:{key}\s*)$", re.M)
    return pat.sub(lambda m: f"{m.group(1)}{'true' if enabled else 'false'}{m.group(3)}", text)


def apply_queue(queue_path: Path, selected: str | None) -> dict:
    text = queue_path.read_text(encoding="utf-8")
    new = text
    for k in KEYS:
        new = set_enabled(new, k, k == selected)
    if new != text:
        tmp = queue_path.with_suffix(".yaml.tmp")
        tmp.write_text(new, encoding="utf-8")
        tmp.replace(queue_path)
    return {k: (k == selected) for k in KEYS}


def build_fold1(key: str) -> None:
    cmd = [sys.executable, "-m", "src.vlm.build_sft", "--fold", "1", "--crop-source", "salad", "--hard-neg", "150", *SFT_ARGS[key]]
    subprocess.run(cmd, cwd=ROOT, check=True)


def main():
    summary = json.loads((TABLES / "p3_summary.json").read_text(encoding="utf-8"))
    selected = (summary.get("screening") or {}).get("selected")
    if selected not in (None, *KEYS):
        raise ValueError(f"알 수 없는 선정 키: {selected}")
    if selected:
        meta = ROOT / "data" / "sft" / f"p3_{selected}_f1" / "meta.json"
        if not meta.exists():
            build_fold1(selected)
        (TABLES / f"p3_selected_{selected}.ok").write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
    flags = apply_queue(QUEUE, selected)
    out = {"selected": selected, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "queue_enabled": flags,
           "effects": {k: (summary.get("screening") or {}).get("variants", {}).get(k, {}).get("all", {}).get("diff")
                       for k in KEYS}}
    (TABLES / "p3_selected.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
