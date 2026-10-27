# 학습 환경 원복 보고 (2026-10-27 02:32:20, 자동)

## 1. 정지
- 감독자 정지: 실행 중 아님
- 실행 중 학습: 없음
## 2. 예약 작업
- \LOCO-Train\Supervisor 해제: 예
## 3. 모니터 창
- 종료: 예
## 4. 중간 체크포인트 삭제 (done run만, 최종 결과·로그는 보존)
- 삭제: ['m1_effad_breakfast_box (0.34 GB)', 'm1_effad_screw_bag (0.34 GB)', 'm1_vlm2b_f0 (0.39 GB)', 'm1_vlm4b_bench (0.74 GB)', 'm2_4b_f0_s0 (0.74 GB)', 'm2_4b_f1_s0 (0.74 GB)', 'm2_2b_f0_s0 (0.39 GB)', 'm2_2b_f1_s0 (0.39 GB)', 'm2_2b_f0_noref (0.39 GB)', 'm2_2b_f0_nocrop (0.39 GB)', 'm2_4b_f0_s1 (0.74 GB)', 'm2_4b_f1_s1 (0.74 GB)', 'm2_4b_f0_s2 (0.74 GB)', 'm2_4b_f1_s2 (0.74 GB)']
- 보존(done 아님): 없음
## 5. 이벤트 로그 보관과 운영 지표
- 이벤트 224건 -> reports/ops/events_20261027_023230.jsonl
- 유휴 6.1분, 시작 46회, 재시도 4회, 실패 1회, 하트비트 정지 0회
- journal: ['snapshot_pre', 'register_task']
## 6. ops/runtime 삭제
- 남음: False
## 7. 사전 스냅샷 비교
- 예약 작업: 사전 206개 / 사후 207개, 추가 ['\\SoftLanding\\S-1-5-21-2619809263-2611117730-3400742752-1001\\SoftLandingDeferralTask-{2d2a954c-71a4-44b0-bcfd-0a43781f4c4d}', '\\SoftLanding\\S-1-5-21-2619809263-2611117730-3400742752-1001\\SoftLandingRefreshForExpirationTask-{42198018-6d21-4f50-95c3-2a674440326e}', '\\SoftLanding\\S-1-5-21-2619809263-2611117730-3400742752-1001\\SoftLandingTriggerTask-128000000001615609-render-{013f2232-5068-4b8c-90b4-b55ec6b2e69a}'], 사라짐 ['\\SoftLanding\\S-1-5-21-2619809263-2611117730-3400742752-1001\\SoftLandingDeferralTask-{f044cbd0-b9da-48b4-adb7-1014b4ad946e}', '\\SoftLanding\\S-1-5-21-2619809263-2611117730-3400742752-1001\\SoftLandingTriggerTask-128000000001615609-render-{40b7fd12-d4ff-4a59-a526-5cab113be846}']
- pip 패키지 변화: 없음
- 사용자 환경 변수 변화(이름만): 없음
- 새 파일 10710개 (보존 대상 밖 0개: [])

**결과: 확인 필요 ['예약 작업 목록 변화']**

## 판정 보정 (2026-10-27)
- 예약 작업 차이 5건은 모두 Windows `\SoftLanding\` 알림 작업이 스스로 바뀐 것이고, `\LOCO-Train\` 관련 잔여는 없다.
- `ops/env.ps1 -Action verify` 잔여 변경 0건. 원복 비교 코드는 우리 폴더 밖 변화를 참고로만 적도록 수정했다.
- **보정 결과: 잔여 변경 없음**
