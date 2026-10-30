# 학습 환경 원복 보고 (2026-10-30 07:00:37, 자동)

## 1. 정지
- 감독자 정지: 실행 중 아님
- 실행 중 학습: 없음
## 2. 예약 작업
- \LOCO-Train\Supervisor 해제: 예
## 3. 모니터 창
- 종료: 예
## 4. 중간 체크포인트 삭제 (done run만, 최종 결과·로그는 보존)
- 삭제: ['m3_2b_f0_noneg (0.39 GB)', 'm3_2b_f0_nospec (0.39 GB)', 'm3_2b_f0_nodet (0.39 GB)', 'm3_2b_f0_nodrop (0.39 GB)', 'm3_2b_f0_vislora (0.53 GB)', 'm3_open_struct_4b_f0 (0.74 GB)', 'm3_open_struct_nodrop_4b_f0 (0.74 GB)', 'm3_open_logic_4b_f0 (0.74 GB)', 'm3_open_logic_nodrop_4b_f0 (0.74 GB)', 'm3_b2_4b_f0 (0.74 GB)', 'm3_b2_4b_f1 (0.74 GB)', 'm3_4b_f0_nocrop (0.74 GB)', 'm3_4b_f1_nocrop (0.74 GB)', 'm3_4b_f0_vislora (0.88 GB)', 'm3_4b_f1_vislora (0.88 GB)', 'p2_2b_f0_frac25 (0.20 GB)', 'p2_2b_f0_frac50 (0.39 GB)', 'p2_8b_f0 (0.98 GB)', 'p2_8b_f1 (0.98 GB)']
- 보존(done 아님): 없음
## 5. 이벤트 로그 보관과 운영 지표
- 이벤트 260건 -> reports/ops/events_20261030_070047.jsonl
- 유휴 140.4분, 시작 42회, 재시도 2회, 실패 2회, 하트비트 정지 0회
- journal: ['snapshot_pre', 'register_task']
## 6. ops/runtime 삭제
- 남음: False
## 7. 사전 스냅샷 비교
- 예약 작업: 사전 207개 / 사후 207개, \LOCO-Train\ 잔여 없음
- (참고) 시스템이 스스로 바꾼 예약 작업 6개: ['\\SoftLanding\\S-1-5-21-2619809263-2611117730-3400742752-1001\\SoftLandingDeferralTask-{3f88bf1d-af69-4702-a36f-0d6a5eda48d2}', '\\SoftLanding\\S-1-5-21-2619809263-2611117730-3400742752-1001\\SoftLandingRefreshForExpirationTask-{7c2761c3-c27a-4bbd-ae47-4a3b8642d486}', '\\SoftLanding\\S-1-5-21-2619809263-2611117730-3400742752-1001\\SoftLandingTriggerTask-128000000001615609-render-{d0510003-a3ae-48bc-a719-208dd0c4b565}', '\\SoftLanding\\S-1-5-21-2619809263-2611117730-3400742752-1001\\SoftLandingDeferralTask-{2d2a954c-71a4-44b0-bcfd-0a43781f4c4d}']
- pip 패키지 변화: 없음
- 사용자 환경 변수 변화(이름만): 없음
- 새 파일 344개 (보존 대상 밖 1개: ['.claude/launch.json'])

**결과: 확인 필요 ['보존 대상 밖 새 파일']**
