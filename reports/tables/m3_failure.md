# 실패 분석 (held-out 원출력, Ours 4B는 seed 3개 평균 정답률)

## 1. 결함 종류별 정확도

| 종류 | n | Ours 4B | B1 | 차이(Ours - B1, %p) |
|---|---|---|---|---|
| 개수 계열 | 169 | 64.3 | 64.5 | -0.2 |
| 기타 논리 | 392 | 60.3 | 80.6 | -20.3 |
| 구조 | 432 | 48.1 | 55.6 | -7.4 |

### 그룹별 (표본 10장 이상, B1 대비 격차가 큰 순 상위 10)

| 카테고리 | 그룹 | n | Ours 4B | B1 | 차이(%p) |
|---|---|---|---|---|---|
| pushpins | 칸막이 누락 | 13 | 2.6 | 100.0 | -97.4 |
| splicing_connectors | 케이블 길이 미달 | 12 | 47.2 | 100.0 | -52.8 |
| splicing_connectors | 커넥터·케이블 누락 | 26 | 56.4 | 100.0 | -43.6 |
| juice_bottle | 충전량 이상 | 48 | 68.1 | 97.9 | -29.9 |
| screw_bag | 나사 길이 규격 불일치 | 65 | 46.2 | 75.4 | -29.2 |
| juice_bottle | 라벨 부착 위치·각도 불량 | 18 | 50.0 | 77.8 | -27.8 |
| breakfast_box | 과일·토핑 손상 | 38 | 30.7 | 57.9 | -27.2 |
| screw_bag | 이물 혼입 | 25 | 21.3 | 44.0 | -22.7 |
| breakfast_box | 내용물 양·비율 이상 | 20 | 25.0 | 45.0 | -20.0 |
| pushpins | 이물 혼입 | 20 | 50.0 | 70.0 | -20.0 |

### 주요 혼동 (seed 0, 상위 12)

| 카테고리 | 정답 그룹 | 예측 | 장수 |
|---|---|---|---|
| screw_bag | 나사 길이 규격 불일치 | 부품 구성 수량 불일치 | 36 |
| screw_bag | 부품 구성 수량 불일치 | 나사 길이 규격 불일치 | 18 |
| screw_bag | 부품 구성 수량 불일치 | 정상 | 15 |
| breakfast_box | 과일·토핑 손상 | 정상 | 14 |
| pushpins | 푸시핀 휨·파손 | 정상 | 13 |
| pushpins | 칸막이 누락 | 칸별 푸시핀 수량 불일치 | 11 |
| pushpins | 푸시핀 휨·파손 | 칸별 푸시핀 수량 불일치 | 11 |
| splicing_connectors | 케이블 배선 불량 | 정상 | 10 |
| pushpins | 칸별 푸시핀 수량 불일치 | 푸시핀 휨·파손 | 10 |
| juice_bottle | 주스 종류·색상 불일치 | 충전량 이상 | 9 |
| juice_bottle | 충전량 이상 | 이물 혼입 | 9 |
| juice_bottle | 주스 종류·색상 불일치 | 과일 아이콘 누락·불일치 | 8 |

- 개수 계열 오답(seed 0)의 내용: 다른 그룹 36, 정상 17, 파싱 실패 1, 미분류 1

## 2. 해상도: 결함 성분 크기(검사 이미지 토큰 수 기준)

성분 크기 = GT 연결 성분 면적 중앙값 / (이미지 면적 / 256). 1 미만이면 부품 하나가 토큰 1개보다 작다.

| 구간 | 개수 계열 n / Ours / B1 | 기타 논리 n / Ours / B1 | 구조 n / Ours / B1 |
|---|---|---|---|
| <1 | 8 / 79.2 / 87.5 | 79 / 57.4 / 70.9 | 226 / 50.3 / 45.6 |
| 1-4 | 50 / 59.3 / 72.0 | 99 / 46.8 / 82.8 | 128 / 52.6 / 55.5 |
| 4-16 | 47 / 90.1 / 93.6 | 41 / 42.3 / 65.9 | 48 / 39.6 / 77.1 |
| >=16 | 64 / 47.4 / 34.4 | 173 / 73.6 / 87.3 | 30 / 26.7 / 96.7 |

## 3. crop: GT 마스크가 crop 상자에 들어간 비율

그 이미지를 예측한 모델 fold의 탐지 맵 기준으로 crop 상자를 다시 계산.

| 구간 | 개수 계열 n / Ours / B1 | 기타 논리 n / Ours / B1 | 구조 n / Ours / B1 |
|---|---|---|---|
| 탐지 영역 없음(전체 이미지) | 10 / 26.7 / 20.0 | 24 / 61.1 / 50.0 | 14 / 11.9 / 14.3 |
| <50% | 58 / 51.7 / 60.3 | 77 / 44.6 / 61.0 | 79 / 30.8 / 32.9 |
| 50-99% | 27 / 84.0 / 81.5 | 39 / 43.6 / 82.1 | 48 / 52.1 / 60.4 |
| 100% | 74 / 72.1 / 67.6 | 252 / 67.6 / 89.3 | 291 / 54.0 / 62.9 |

- 개수 계열 카테고리별 성분 크기 중앙값(토큰) / crop 포함률 중앙값 / Ours: breakfast_box 66.25 / 46% / 66.7; pushpins 2.04 / 53% / 81.6; screw_bag 123.88 / 100% / 44.9

## 4. 정상 참조

검사 이미지-참조 DINOv2 유사도의 카테고리 내 3분위별 Ours 정확도

| 유사도 | 개수 계열 | 기타 논리 | 구조 |
|---|---|---|---|
| 하 | 62.3 | 72.9 | 46.2 |
| 중 | 69.5 | 56.4 | 48.2 |
| 상 | 62.3 | 44.6 | 49.6 |

2B fold 0 모델 → fold 1 이미지, 결함 종류별 정확도

| 설정 | 개수 계열 | 기타 논리 | 구조 |
|---|---|---|---|
| 2B 기본 | 45.8 | 52.8 | 33.8 |
| 정상 참조 제거 | 47.0 | 50.8 | 35.2 |
| crop 제거 | 44.6 | 43.1 | 23.1 |
| 비전 인코더 LoRA | 47.0 | 56.9 | 37.5 |

(fold 1 개수 계열 83장)

## 5. 사례 후보 (PDF 그림4, held-out)

- 성공_논리: breakfast_box/test/logical_anomalies/002, juice_bottle/test/logical_anomalies/000, pushpins/test/logical_anomalies/001
- 성공_구조: breakfast_box/test/structural_anomalies/003, juice_bottle/test/structural_anomalies/000, pushpins/test/structural_anomalies/001
- 실패_개수_다른그룹: breakfast_box/test/logical_anomalies/055, pushpins/test/logical_anomalies/006, screw_bag/test/logical_anomalies/041
- 실패_작은결함: breakfast_box/test/logical_anomalies/046, juice_bottle/test/logical_anomalies/091, pushpins/test/logical_anomalies/015
- 실패_crop_놓침: breakfast_box/test/logical_anomalies/009, juice_bottle/test/structural_anomalies/005, pushpins/test/logical_anomalies/011
