# MVTec LOCO AD: 이상 탐지기 + LoRA 적응 VLM 2줄 이상 리포트

비지도 이상 탐지기(SALAD)가 1차 판정하고, 이상일 때만 QLoRA로 적응시킨 Qwen3-VL이
고정 템플릿 2줄을 출력한다. 정상이면 아무것도 출력하지 않는다.

```
이상: [논리] 부품 구성 수량 불일치 - 너트 1개 초과
조치: 구성표대로 부품 수량을 보정해 재포장하고, 부품별 투입 피더와 계량 센서를 점검
```

> 진행 중인 프로젝트다. 현재 Phase 0(데이터 인덱스, 택소노미, 분할)까지 완료.

## 출력 명세

```
이상: [<유형>] <그룹명>[ - <세부>[ + <세부>]][ (<위치>)]
조치: <조치>
```

- 유형 `논리 | 구조 | 미상`, 그룹명과 조치는 `taxonomy/action_kb.yaml` 문장을 그대로 쓴다.
- 위치는 세부 결함의 `loc_mode`로 정한다.
  - `point`: 3x3 칸 이름 (GT 마스크를 긴 변 2% 반경으로 closing한 뒤 면적 최대 칸)
  - `global`: `전체` (제품·구획 전체, 부품 사이 관계를 칠한 결함)
  - `instances`: 생략 (같은 종류 부품을 모두 칠해 어느 개체가 결함인지 GT로 알 수 없음)
- 각 줄 60자 이하. `tests/test_output_spec.py`가 모든 KB 조합에서 길이와 파서 왕복을 확인한다.

## 재현 절차

### Phase 0. 데이터 인덱스와 택소노미

```bash
python -m src.data.mask_stats        # 세부 결함별 마스크 통계 + 확인용 시트
python -m src.data.index_loco        # data/index.parquet (3,651장, 위치 라벨·정답 리포트 포함)
python -m src.data.splits            # 1차 분할 (근접 중복 목록이 없으면 그대로)
python -m src.data.near_dups         # DINOv2 임베딩 + 근접 중복 쌍
python -m src.data.splits            # 근접 중복을 같은 fold로 모아 다시 분할 + inner-val
python -m pytest
```

- 데이터: `MVTec-LOCO/`에 MVTec LOCO AD와 MMAD 부가자료(`QA.json` 등)를 둔다.
- 분할: test 1,568장을 (카테고리, 라벨, 결함 그룹) 층화 2-fold로 나누고, fold마다 학습 측 15%를
  inner-val(하이퍼파라미터·프롬프트·임계값 선택 전용)로 뗀다. fold에 2장 이하인 그룹은 inner-val에서 뺀다.
- `loc_mode` 확정 과정: 통계 기반 초안과 시트 확인 결과가 다른 20종은 `data/defect_mask_stats.csv`의
  `draft_loc_mode`와 `kb_loc_mode` 열에 남아 있다.
- 근접 중복: 처음 계획한 "DINOv2 CLS 코사인 0.98 이상"은 쓸 수 없었다. 카메라·배치가 고정된 장면이라
  서로 다른 이미지도 전역 특징이 거의 같다(pushpins는 test 쌍 유사도 중앙값이 0.9835라 3만 쌍 이상이 걸림).
  대신 같은 위치 패치 토큰의 평균 코사인을 쓰고, 카테고리마다 최근접 유사도의 중앙값 + 6 x MAD(정규 환산)를
  임계값으로 둔다. 결과는 핀이 모두 빠진 빈 케이스 5장(pushpins 논리 057~061) 한 묶음이며 같은 fold로 모았다.

## 데이터 사용 규칙

- MMAD QA: `annotation: false` 문항은 제외한다. `summary.json`은 test 전체를 요약한 것이라 쓰지 않는다(누수).
  MMAD는 학습에 쓰지 않고 학습 전후 객관식 정확도(전이·망각) 측정에만 쓴다.
- held-out fold는 최종 1회만 평가한다. 설정 선택은 inner-val로만 한다.
- 조치 KB(`taxonomy/action_kb.yaml`)는 포트폴리오용 **가상 SOP**이며 실제 현장 절차가 아니다.

## 외부 코드

`external/`에 복제해 쓰며 저장소에는 포함하지 않는다.

| 이름 | 복제 명령 | 커밋 |
|---|---|---|
| SALAD (ICCV 2025, MIT) | `git clone https://github.com/MaticFuc/SALAD.git external/SALAD` | `753ba942025d2644c494a18c1fb4b60ae15332d5` |
| CSAD (대체 탐지기 후보) | `git clone https://github.com/Tokichan/CSAD.git external/CSAD` | `d46cf856b99bf056088b5f510fd984e04813583a` |

## 라이선스

- MVTec LOCO AD: CC BY-NC-SA 4.0 (비상업). 데이터셋 이미지는 재배포하지 않는다.
- MMAD 부가자료: 라이선스 미표기라 재배포하지 않고 인용만 한다.
