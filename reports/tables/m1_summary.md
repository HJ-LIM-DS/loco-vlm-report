# 선행 실험 결과 요약 (EfficientAD + 2B, 2개 카테고리)

## 탐지기 (EfficientAD-M, test 전체)

| 카테고리 | AUROC 전체 | 논리 | 구조 | 지연(ms/장) | 학습 step |
|---|---|---|---|---|---|
| breakfast_box | 77.0 | 82.1 | 72.3 | 44.62 | 70000 |
| screw_bag | 71.6 | 59.4 | 92.1 | 40.99 | 70000 |

## VLM 학습 (1회 학습 시간·VRAM 실측)

| run | 모델 | 양자화 | 학습 샘플 | step | 학습 시간(분) | 최대 VRAM 할당/예약(GB) |
|---|---|---|---|---|---|---|
| m1_vlm2b_f0 | Qwen3-VL-2B-Instruct | none | 261 | 99 | 24.6 | 4.99 / 6.5 |
| m1_vlm4b_bench | Qwen3-VL-4B-Instruct | nf4 | 261 | 30 | 15.8 | 5.43 / 6.15 |

## inner-val 생성 평가 (n=47, 원출력 / Guard 후)

| 지표 | 원출력 | Guard 후 |
|---|---|---|
| format_ok | 100.0 | 100.0 |
| normal_decision_acc | 72.3 | 72.3 |
| group_acc | 23.3 | 23.3 |
| type_acc | 50.0 | 50.0 |
| detail_exact | 0.0 | 3.3 |
| location_exact | 20.0 | 20.0 |

## 종단 (fold 1 held-out, 이상 196 / 정상 112)

| 구성 | 재현율 | FPR |
|---|---|---|
| 탐지기 단독 | 92.3 | 75.9 |
| 탐지기 + VLM 거부 | 81.6 | 51.8 |

- 이상 이미지의 종단 그룹 정확도: 24.5%
- 출력 형식 준수: 100.0%
- VLM 지연: 2.958초/장
