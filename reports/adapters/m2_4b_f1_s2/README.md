---
base_model: Qwen/Qwen3-VL-4B-Instruct
library_name: peft
license: cc-by-nc-sa-4.0
language:
- ko
pipeline_tag: image-text-to-text
tags:
- lora
- qwen3-vl
- anomaly-detection
- industrial-inspection
- mvtec-loco
- base_model:adapter:Qwen/Qwen3-VL-4B-Instruct
---

# LOCO VLM 이상 리포트 LoRA 어댑터: `m2_4b_f1_s2`

English summary: a rank-16 LoRA adapter for `Qwen/Qwen3-VL-4B-Instruct` that turns an industrial inspection image (plus a detector crop and a normal reference) into a fixed two-line Korean anomaly report on MVTec LOCO AD. Trained on the training side of fold 1; evaluate it only on the other fold. Non-commercial (CC BY-NC-SA 4.0) because the training data carries that license.

## 무엇을 하는 어댑터인가

- 입력: 이미지 3장(검사 이미지, 탐지기 의심 영역 crop에 윤곽선 표시, 최근접 정상 참조; 각 약 256 토큰)과
  제품·정상 규격·결함 그룹 목록·탐지기 백분위를 담은 한국어 프롬프트(저장소의 `src/vlm/prompts.py`).
- 출력: 정상이면 `정상` 한 단어, 이상이면 고정 2줄 `이상: [논리|구조] <결함 그룹>[ - <세부>][ (<위치>)]` / `조치: <조치>`.
- 학습 데이터: MVTec LOCO AD test 이미지의 2-fold 분할 중 fold 1 학습 측(학습 정상 150 / 이상 418장, inner-val 정상 43 / 이상 78장, 정상 하드 네거티브 150장, 목록 드롭아웃 0.12). fold 0 이미지는 학습에 쓰지 않았으므로 평가는 그 fold로만 한다.
- 탐지기: SALAD(ICCV 2025) 공식 가중치 재현 캐시. crop과 백분위는 이 캐시에서 나온다.

## 학습 설정

| 항목 | 값 |
|---|---|
| 기반 모델 | `Qwen/Qwen3-VL-4B-Instruct` (4비트 NF4(QLoRA, 이중 양자화, bf16 연산)) |
| LoRA | r=16, alpha=32, dropout=0.05, 언어 모델 q/k/v/o·gate/up/down, 비전 인코더 동결 |
| 최적화 | lr 0.0001, epoch 3.0, 배치 1 x 누적 8, warmup 5 step, seed 2 |
| 손실 | 정답 토큰(assistant 턴) logits만 계산 |
| 비용 | 학습 2.1h (RTX 3060 12GB), 최대 예약 VRAM 6.23GB, 생성 6.2초/장(배치 2) |

## 결과 (held-out fold 0, 원출력, n=785)

| 형식 준수 | 정상/이상 판정 | 결함 그룹 | 논리/구조 유형 | 세부 결함 | 위치 |
|---|---|---|---|---|---|
| 100.0 | 90.1 | 55.3 | 73.4 | 43.7 | 59.2 |

두 fold 합산 표1 기준(`Ours 4B`, seed 3개 평균): 그룹 55.7, 유형 71.4, 세부 44.5, 위치 적중 69.0, 정상 거부 86.8, 종단 그룹 F1 55.6 (π=1%: 10.6).

주의: 같은 입력을 쓰는 비-VLM 기준선(DINOv2 + SALAD 점수 -> 로지스틱 회귀)이 그룹 정확도 67.0으로 이 어댑터보다 높다. 자세한 비교와 한계는 저장소 README를 본다.

## 사용법

```python
import torch
from peft import PeftModel
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

base = Qwen3VLForConditionalGeneration.from_pretrained("Qwen/Qwen3-VL-4B-Instruct", dtype=torch.bfloat16,
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_skip_modules=["visual", "lm_head"]),
    attn_implementation="sdpa", device_map={"": 0})
model = PeftModel.from_pretrained(base, "<namespace>/loco-vlm-m2_4b_f1_s2").eval()
processor = AutoProcessor.from_pretrained("<namespace>/loco-vlm-m2_4b_f1_s2")
# 메시지: 이미지 3장 + 프롬프트(src/vlm/prompts.py의 user_text), greedy 생성 max_new_tokens=80
```

## 라이선스와 한계

- 어댑터 가중치: CC BY-NC-SA 4.0. MVTec LOCO AD(Bergmann et al., IJCV 2022, CC BY-NC-SA 4.0)로 학습했으므로 비상업·동일 조건을 따른다. 기반 모델 Qwen3-VL은 Apache-2.0.
- 조치 문장은 포트폴리오용 가상 SOP이며 실제 현장 절차가 아니다.
- 결함 그룹 36개 중 여럿이 10장 미만이고, 공개 데이터셋이라 기반 모델의 사전학습 노출 가능성을 배제하지 못한다.
- 대상 fold가 아닌 이미지(학습에 쓴 fold)에 대한 수치는 과대평가다.
