# MVTec LOCO AD: 이상 탐지기 + LoRA 적응 VLM 2줄 이상 리포트

비지도 이상 탐지기(SALAD)가 1차 판정하고, 이상일 때만 QLoRA로 적응시킨 Qwen3-VL이
고정 템플릿 2줄을 출력한다. 정상이면 아무것도 출력하지 않는다.

```
이상: [논리] 부품 구성 수량 불일치 - 너트 1개 초과
조치: 구성표대로 부품 수량을 보정해 재포장하고, 부품별 투입 피더와 계량 센서를 점검
```

> 진행 중인 프로젝트다. Phase 0(데이터 인덱스, 택소노미, 분할) 완료, M1(MVP + 학습 운영)과 M2(본 실험) 진행 중.

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

### M1. MVP (EfficientAD + Qwen3-VL-2B LoRA, breakfast_box·screw_bag)

모든 학습은 대기열(`configs/run_queue.yaml`)에 넣고 감독자가 실행한다. Claude Code나 터미널의 자식 프로세스로
직접 띄우지 않는다.

```bash
python -m src.vlm.build_sft --fold 0 --categories breakfast_box screw_bag --crop-source gt --name m1_gt
python -m ops.smoke m1_vlm2b_f0
```

```powershell
.\ops\env.ps1 -Action setup
```

대기열 순서: EfficientAD 학습·탐지 캐시 내보내기(카테고리별) -> 탐지기 crop으로 SFT 데이터 재생성 ->
2B LoRA 학습(시간·VRAM 실측, inner-val 생성 평가) -> 4B QLoRA 30 step 실측 -> 종단 실행(`src.pipeline`).
결과 표는 `python -m src.report.tables m1`, 그림은 `python -m src.report.figures ...`로 다시 만든다.

### M2. 본 실험 (SALAD 캐시 동결, 5개 카테고리 2-fold)

SALAD는 `imgaug`(numpy 2 비호환) 때문에 별도 가상환경에서 공식 사전학습 가중치로 점수만 추출한다.

```bash
python -m venv E:\venvs\loco_salad
E:\venvs\loco_salad\Scripts\python.exe -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
E:\venvs\loco_salad\Scripts\python.exe -m pip install numpy==1.26.4 opencv-python==4.10.0.84 scikit-learn scipy timm "positional-encodings[pytorch]" imgaug "pandas<3" tqdm pillow gdown
```

- 배포 가중치 점검: `E:\venvs\loco_salad\Scripts\python.exe -m src.detector.fix_salad_weights`.
  공식 `salad_loco.zip`의 `breakfast_box/autoencoder_final.pth`에는 다른 가중치 파일 24개가 같은 zip 안에 덧붙어
  있어(1.2GB) `torch.load`가 실패한다. 모델 기록만 남겨 다시 묶으면 정상 로드된다(CRC 정상, 다른 카테고리와 같은 구조).
- `src/detector/salad_runner.py`: 공식 `test()`와 같은 프로세스·정규화 값으로 이미지별 점수 3항(국소 맵 최댓값,
  Mahalanobis, 합성 맵 최댓값)과 맵을 저장하고, 저장값으로 다시 계산한 AUROC 4종이 공식 출력과 같은지 확인한다.
- SFT: `build_sft --crop-source salad --hard-neg 150` (정상 하드 네거티브 = 학습 측 정상 중 탐지 점수 상위).
  입력 ablation은 `--drop-image ref|crop`.
- 학습: 4B QLoRA 2 fold x seed 3, 2B LoRA 2 fold, ablation(2B, fold 0), epoch 3 고정. 학습 후 held-out 예측을 한 번 저장.
- 기준선: B0 최빈값(`src.baselines.majority`), B1 비-VLM 진단기(`src.baselines.crop_classifier`),
  B3 프롬프트만 쓰는 VLM(`src.baselines.vlm_prompt`, zero/few-shot, 엄격·관대 파싱).
- MMAD 객관식(학습 전후, 전이·망각): `src.eval.mmad_eval`.
- 집계: `python -m src.report.m2_report` -> `reports/tables/m2_summary.md`.

## 학습 운영 (GPU 유휴 방지)

| 파일 | 역할 |
|---|---|
| `ops/supervisor.py` | 대기열 실행, 종료 코드·하트비트 감시, 오류 서명 기반 재시도(같은 오류 2회면 failed 후 다음 run), 재시작 시 실행 중 학습 이어받기, 완료 시 자동 원복 |
| `ops/train_hooks.py` | 학습 스크립트 공용: 진행률·하트비트, 원자적 체크포인트 + 완료 표식, NaN 감지, 손상 체크포인트를 건너뛰는 재개, 고장 주입 |
| `ops/monitor.py` | 모니터 창(진행 막대, GPU, 하트비트 경과, 최근 이벤트). 닫아도 학습은 계속되며 `ops\open_monitor.cmd`로 다시 연다. 실행 중인 run의 비고 칸 "직전 시도 오류"는 지금 오류가 아니라 이전 시도의 실패 이유다 |
| `ops/smoke.py` | 대기열에 넣을 때 20 step + 저장·재개 1회 스모크 관문 |
| `ops/envctl.py`, `ops/env.ps1` | 예약 작업 `\LOCO-Train\Supervisor` 등록·해제, 사전·사후 스냅샷, journal, 원복 검증 |
| `ops/fault_drill.py` | 고장 주입 시험 F1~F9 구동과 판정 (`reports/ops/fault_tests.md`) |

- 영구 변경은 예약 작업 1개뿐이다(로그온 시 + 10분마다, 대화형 로그온, 암호 저장 없음, 실행 시간 제한 해제).
- 모든 run이 done이고 결과 검증을 통과하면 자동 원복한다: 예약 작업 해제, done run의 중간 체크포인트 삭제,
  이벤트 로그 보관(`reports/ops/`), `ops/runtime/` 삭제, 사전 스냅샷과 비교(`reports/ops/teardown_report.md`).
- 수동: `.\ops\env.ps1 -Action teardown`(즉시 원복), `-Action verify`(잔여 변경 확인),
  `ops/runtime/PAUSE` 파일을 만들면 실행 중인 run을 멈추고 새 run을 시작하지 않는다.

환경 메모 (Windows, torch 2.5.1)
- torch를 먼저 불러온 뒤 `pyarrow.dataset`을 불러오면 DLL 충돌로 traceback 없이 접근 위반으로 죽는다.
  `src/__init__.py`에서 pyarrow를 먼저 불러온다.
- transformers는 torch 2.6 미만에서 체크포인트의 `torch.load`를 막는다(CVE-2025-32434). 재개 시 같은 run의
  출력 폴더 안 파일에 한해 해당 호출 동안만 허용한다(`src/vlm/train_lora.py`의 `trusted_checkpoint`).

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
