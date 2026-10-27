"""학습 운영 환경 설정·원복·검증 (ops/env.ps1이 호출).

영구 변경은 작업 스케줄러 폴더 \\LOCO-Train\\의 작업 1개(Supervisor)뿐이다. 절전 방지·환경 변수·
CUDA 설정은 모두 프로세스 범위라 종료 시 사라진다. 모든 변경은 journal에 남기고, 원복은 journal
역순으로 되돌린 뒤 사전 스냅샷과 비교해 검증한다. 삭제는 프로젝트 하위 경로로만 제한한다.

    setup     사전 스냅샷 -> 예약 작업 등록 (등록 직전에 사용자 확인을 받는다)
    teardown  감독자·모니터 정지 -> 예약 작업 해제 -> done run의 중간 체크포인트 삭제
              -> 이벤트 로그 보관 -> ops/runtime 삭제 -> 사후 스냅샷 비교 -> reports/ops/teardown_report.md
    verify    잔여 변경(예약 작업, 런타임 파일, 중간 체크포인트) 확인

사용법:
    python -m ops.envctl setup | teardown [--auto] [--force] | verify
"""

import argparse
import base64
import hashlib
import json
import shutil
import subprocess
import sys
import time
import winreg
from pathlib import Path

import psutil

from ops.paths import (
    EVENTS_FILE, JOURNAL_FILE, MONITOR_EXIT, MONITOR_PID, ROOT, RUNS, RUNTIME, STATE_FILE, SUPERVISOR_PID, append_event,
    read_events, read_json, write_json_atomic,
)

TASK_FOLDER = "\\LOCO-Train\\"
TASK_NAME = "Supervisor"
TASK_FULL = TASK_FOLDER + TASK_NAME
SNAP_PRE = RUNTIME / "snapshot_pre"
REPORT_DIR = ROOT / "reports" / "ops"
# 파일 목록 스냅샷에서 제외 (데이터셋, 외부 코드, git 내부, 캐시)
SNAP_EXCLUDE = {"MVTec-LOCO", "external", ".git", "__pycache__", ".pytest_cache"}
CREATE_NO_WINDOW = 0x08000000


def ps(script: str, check: bool = True) -> str:
    """PowerShell 5.1 스크립트 실행 (UTF-16 base64로 넘겨 따옴표·한글 문제를 피함)."""
    full = ("[Console]::OutputEncoding = [Text.Encoding]::UTF8\n$ErrorActionPreference = 'Stop'\n"
            "$ProgressPreference = 'SilentlyContinue'\n" + script)
    enc = base64.b64encode(full.encode("utf-16-le")).decode("ascii")
    r = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-EncodedCommand", enc],
                       capture_output=True, creationflags=CREATE_NO_WINDOW)
    out = r.stdout.decode("utf-8", errors="replace")
    if check and r.returncode != 0:
        raise RuntimeError(f"PowerShell 실패({r.returncode}): {r.stderr.decode('utf-8', errors='replace')[-800:]}")
    return out


def journal(action: str, **detail) -> None:
    JOURNAL_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(JOURNAL_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": time.time(), "action": action, **detail}, ensure_ascii=False) + "\n")


def read_journal() -> list:
    try:
        return [json.loads(x) for x in JOURNAL_FILE.read_text(encoding="utf-8").splitlines() if x.strip()]
    except FileNotFoundError:
        return []


# --- 스냅샷 ---

def task_list() -> list:
    out = ps("Get-ScheduledTask | ForEach-Object { $_.TaskPath + $_.TaskName }")
    return sorted(x.strip() for x in out.splitlines() if x.strip())


def user_env_hashes() -> dict:
    """사용자 환경 변수: 값에 비밀이 있을 수 있어 이름과 값의 해시만 남긴다."""
    out = {}
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
        i = 0
        while True:
            try:
                name, val, _ = winreg.EnumValue(k, i)
            except OSError:
                break
            out[name] = hashlib.sha256(str(val).encode("utf-8")).hexdigest()[:16]
            i += 1
    return out


def project_files() -> dict:
    out = {}
    for p in ROOT.rglob("*"):
        rel = p.relative_to(ROOT)
        if rel.parts and rel.parts[0] in SNAP_EXCLUDE or "__pycache__" in rel.parts:
            continue
        if rel.parts[:2] == ("ops", "runtime"):
            continue
        if p.is_file():
            out[rel.as_posix()] = p.stat().st_size
    return out


def snapshot() -> dict:
    # pip freeze가 이 환경에서 내부 오류를 내므로 설치 배포판 목록을 직접 읽는다
    from importlib.metadata import distributions
    pip = [f"{d.metadata['Name']}=={d.version}" for d in distributions()]
    try:
        power = subprocess.run(["powercfg", "/requests"], capture_output=True, creationflags=CREATE_NO_WINDOW)
        power_txt = power.stdout.decode("cp949", errors="replace")
    except Exception as e:  # noqa: BLE001 - 참고용
        power_txt = f"확인 실패: {e}"
    return {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "tasks": task_list(), "pip": sorted(pip),
            "user_env": user_env_hashes(), "files": project_files(), "powercfg_requests": power_txt}


# --- 예약 작업 ---

def task_info() -> dict | None:
    out = ps(f"""
try {{ $t = Get-ScheduledTask -TaskPath '{TASK_FOLDER}' -TaskName '{TASK_NAME}' -ErrorAction Stop }}
catch {{ $t = $null }}
if ($t) {{
  [pscustomobject]@{{ State = "$($t.State)"; Limit = $t.Settings.ExecutionTimeLimit;
    Multi = "$($t.Settings.MultipleInstances)"; Triggers = $t.Triggers.Count;
    Restart = $t.Settings.RestartCount; Exec = $t.Actions[0].Execute; Args = $t.Actions[0].Arguments;
    Dir = $t.Actions[0].WorkingDirectory; Logon = "$($t.Principal.LogonType)"; Run = "$($t.Principal.RunLevel)" }} |
    ConvertTo-Json -Compress
}}""")
    out = out.strip()
    return json.loads(out) if out else None


def register_task() -> None:
    pyw = Path(sys.executable).with_name("pythonw.exe")
    if not pyw.exists():
        raise RuntimeError(f"pythonw.exe 없음: {pyw}")
    ps(f"""
$user = "$env:USERDOMAIN\\$env:USERNAME"
$a = New-ScheduledTaskAction -Execute '{pyw}' -Argument '-m ops.supervisor' -WorkingDirectory '{ROOT}'
$rep = (New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Minutes 10)).Repetition
$t1 = New-ScheduledTaskTrigger -AtLogOn -User $user
$t1.Repetition = $rep
$t2 = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) -RepetitionInterval (New-TimeSpan -Minutes 10)
$s = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -ExecutionTimeLimit ([TimeSpan]::Zero) `
  -StartWhenAvailable -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1) `
  -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
$p = New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskPath '{TASK_FOLDER}' -TaskName '{TASK_NAME}' -Action $a -Trigger @($t1, $t2) `
  -Settings $s -Principal $p -Description 'LOCO VLM training supervisor (auto-removed by ops/env.ps1 teardown)' | Out-Null
""")


def unregister_task() -> bool:
    if task_info() is None:
        return False
    ps(f"""
Stop-ScheduledTask -TaskPath '{TASK_FOLDER}' -TaskName '{TASK_NAME}' -ErrorAction SilentlyContinue
Unregister-ScheduledTask -TaskPath '{TASK_FOLDER}' -TaskName '{TASK_NAME}' -Confirm:$false
$svc = New-Object -ComObject Schedule.Service; $svc.Connect()
try {{ $svc.GetFolder('\\').DeleteFolder('{TASK_FOLDER.strip(chr(92))}', 0) }} catch {{ }}
""")
    return True


# --- 프로세스 정지 ---

def _alive(rec: dict | None) -> psutil.Process | None:
    if not rec:
        return None
    try:
        p = psutil.Process(rec["pid"])
        return p if abs(p.create_time() - rec.get("create_time", 0)) < 1.0 else None
    except (psutil.NoSuchProcess, KeyError, ValueError):
        return None


def stop_supervisor() -> bool:
    p = _alive(read_json(SUPERVISOR_PID))
    if p is None:
        return False
    p.kill()
    p.wait(timeout=30)
    return True


def stop_monitor() -> bool:
    p = _alive(read_json(MONITOR_PID))
    if p is None:
        return False
    MONITOR_EXIT.parent.mkdir(parents=True, exist_ok=True)
    MONITOR_EXIT.write_text("teardown", encoding="utf-8")
    try:
        p.wait(timeout=15)  # 모니터가 표식을 보고 정상 종료하면 창도 닫힌다
    except psutil.TimeoutExpired:
        p.kill()
    return True


def running_training() -> list:
    state = read_json(STATE_FILE, {"runs": {}})
    out = []
    for name, st in state.get("runs", {}).items():
        if st.get("status") == "running" and _alive({"pid": st.get("pid"), "create_time": st.get("create_time")}):
            out.append(name)
    return out


# --- 명령 ---

def setup() -> None:
    if task_info() is not None:
        print("이미 등록되어 있음:", TASK_FULL)
        return
    RUNTIME.mkdir(parents=True, exist_ok=True)
    if not (SNAP_PRE / "snapshot.json").exists():
        write_json_atomic(SNAP_PRE / "snapshot.json", snapshot())
        journal("snapshot_pre", path=str(SNAP_PRE / "snapshot.json"))
    register_task()
    journal("register_task", task=TASK_FULL)
    append_event("env_setup", None, task=TASK_FULL)
    info = task_info()
    print(json.dumps(info, ensure_ascii=False, indent=1))
    if info is None or info.get("Limit") not in ("PT0S", "P0D", "00:00:00"):
        raise RuntimeError(f"등록 확인 실패 또는 실행 시간 제한이 해제되지 않음: {info}")


def idle_metrics(events: list) -> dict:
    """대기열에 할 일이 있는데 학습이 돌지 않은 시간(유휴) 합계."""
    idle, gap_start = 0.0, None
    for ev in events:
        k = ev["kind"]
        if k in ("run_done", "run_failed", "run_retry", "heartbeat_timeout", "run_paused"):
            gap_start = ev["ts"]
        elif k == "run_start" and gap_start is not None:
            idle += ev["ts"] - gap_start
            gap_start = None
        elif k == "queue_finished":
            gap_start = None
    starts = [e for e in events if e["kind"] == "run_start"]
    return {"idle_minutes": round(idle / 60, 1), "run_starts": len(starts),
            "retries": sum(e["kind"] == "run_retry" for e in events),
            "failures": sum(e["kind"] == "run_failed" for e in events),
            "heartbeat_timeouts": sum(e["kind"] == "heartbeat_timeout" for e in events)}


def teardown(auto: bool, force: bool) -> int:
    lines = [f"# 학습 환경 원복 보고 ({time.strftime('%Y-%m-%d %H:%M:%S')}, {'자동' if auto else '수동'})", ""]
    if auto:
        time.sleep(5)  # 자동 원복을 부른 감독자가 종료할 시간을 준다
    busy = running_training()
    if busy and not force:
        print("실행 중인 학습이 있어 원복을 보류함:", busy, "(--force로 강제)")
        return 2
    state = read_json(STATE_FILE, {"runs": {}})
    pre = read_json(SNAP_PRE / "snapshot.json")
    events = read_events()
    jr = read_journal()

    lines.append("## 1. 정지")
    lines.append(f"- 감독자 정지: {'예' if stop_supervisor() else '실행 중 아님'}")
    lines.append(f"- 실행 중 학습: {busy or '없음'}")

    lines.append("## 2. 예약 작업")
    lines.append(f"- {TASK_FULL} 해제: {'예' if unregister_task() else '등록되어 있지 않음'}")

    lines.append("## 3. 모니터 창")
    lines.append(f"- 종료: {'예' if stop_monitor() else '열려 있지 않음'}")

    lines.append("## 4. 중간 체크포인트 삭제 (done run만, 최종 결과·로그는 보존)")
    removed, kept = [], []
    for name, st in state.get("runs", {}).items():
        ck = (RUNS / name / "checkpoints").resolve()
        if not ck.is_dir():
            continue
        if RUNS.resolve() not in ck.parents:
            raise RuntimeError(f"프로젝트 밖 경로 삭제 시도: {ck}")
        if st.get("status") == "done":
            size = sum(f.stat().st_size for f in ck.rglob("*") if f.is_file())
            shutil.rmtree(ck)
            removed.append(f"{name} ({size / 1024**3:.2f} GB)")
        else:
            kept.append(f"{name} ({st.get('status')})")
    lines.append(f"- 삭제: {removed or '없음'}")
    lines.append(f"- 보존(done 아님): {kept or '없음'}")

    lines.append("## 5. 이벤트 로그 보관과 운영 지표")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    if EVENTS_FILE.exists():
        shutil.copy2(EVENTS_FILE, REPORT_DIR / f"events_{stamp}.jsonl")
    write_json_atomic(REPORT_DIR / f"queue_state_{stamp}.json", state)
    m = idle_metrics(events)
    lines.append(f"- 이벤트 {len(events)}건 -> reports/ops/events_{stamp}.jsonl")
    lines.append(f"- 유휴 {m['idle_minutes']}분, 시작 {m['run_starts']}회, 재시도 {m['retries']}회, "
                 f"실패 {m['failures']}회, 하트비트 정지 {m['heartbeat_timeouts']}회")
    lines.append(f"- journal: {[j['action'] for j in jr]}")

    lines.append("## 6. ops/runtime 삭제")
    if RUNTIME.exists():
        if ROOT.resolve() not in RUNTIME.resolve().parents:
            raise RuntimeError(f"프로젝트 밖 런타임 경로: {RUNTIME}")
        shutil.rmtree(RUNTIME, ignore_errors=True)
    lines.append(f"- 남음: {RUNTIME.exists()}")

    lines.append("## 7. 사전 스냅샷 비교")
    problems = compare_with(pre, lines)
    lines.append("")
    lines.append(f"**결과: {'잔여 변경 없음' if not problems else '확인 필요 ' + str(problems)}**")
    (REPORT_DIR / "teardown_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0 if not problems else 1


def compare_with(pre: dict | None, lines: list) -> list:
    problems = []
    if pre is None:
        lines.append("- 사전 스냅샷 없음: 비교 생략")
        return ["사전 스냅샷 없음"]
    post = snapshot()
    added = sorted(set(post["tasks"]) - set(pre["tasks"]))
    missing = sorted(set(pre["tasks"]) - set(post["tasks"]))
    # 우리가 바꾸는 것은 \LOCO-Train\ 폴더뿐이다. 그 밖의 변화(예: Windows SoftLanding 알림 작업의 자체 갱신)는
    # 원복 대상이 아니므로 참고로만 적는다
    ours = [t for t in added + missing if t.startswith(TASK_FOLDER)]
    other = [t for t in added + missing if not t.startswith(TASK_FOLDER)]
    lines.append(f"- 예약 작업: 사전 {len(pre['tasks'])}개 / 사후 {len(post['tasks'])}개, "
                 f"{TASK_FOLDER} 잔여 {ours or '없음'}")
    if other:
        lines.append(f"- (참고) 시스템이 스스로 바꾼 예약 작업 {len(other)}개: {other[:4]}")
    if ours:
        problems.append("LOCO 예약 작업 잔여")
    pip_diff = sorted(set(pre["pip"]) ^ set(post["pip"]))
    lines.append(f"- pip 패키지 변화: {pip_diff or '없음'}")
    if pip_diff:
        problems.append("pip 변화")
    env_diff = sorted(k for k in set(pre["user_env"]) | set(post["user_env"])
                      if pre["user_env"].get(k) != post["user_env"].get(k))
    lines.append(f"- 사용자 환경 변수 변화(이름만): {env_diff or '없음'}")
    if env_diff:
        problems.append("환경 변수 변화")
    new_files = sorted(set(post["files"]) - set(pre["files"]))
    # 보존 대상: 학습 결과(runs/), 탐지 캐시·예측(outputs/), 보고서(reports/), 실험 기록(mlruns/),
    # 소스 코드 변경은 git이 추적
    preserved = ("runs/", "outputs/", "reports/", "src/", "ops/", "configs/", "tests/", "data/", "app/", "mlruns/",
                 "README.md", "pyproject.toml", ".gitignore")
    unexpected = [f for f in new_files if not f.startswith(preserved)]
    lines.append(f"- 새 파일 {len(new_files)}개 (보존 대상 밖 {len(unexpected)}개: {unexpected[:20]})")
    if unexpected:
        problems.append("보존 대상 밖 새 파일")
    return problems


def verify() -> int:
    issues = []
    info = task_info()
    if info is not None:
        issues.append(f"예약 작업 남음: {TASK_FULL} (상태 {info.get('State')}, 실행 제한 {info.get('Limit')})")
    if RUNTIME.exists():
        issues.append(f"런타임 폴더 남음: {RUNTIME}")
    state_runs = {}
    for d in RUNS.glob("*/checkpoints"):
        if any(d.iterdir()):
            state_runs[d.parent.name] = sum(1 for _ in d.iterdir())
    if state_runs:
        issues.append(f"중간 체크포인트 남음: {state_runs}")
    if _alive(read_json(SUPERVISOR_PID)):
        issues.append("감독자 실행 중")
    print("잔여 변경 0건" if not issues else "잔여 변경:\n- " + "\n- ".join(issues))
    if info is not None:
        print(json.dumps(info, ensure_ascii=False, indent=1))
    return 0 if not issues else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["setup", "teardown", "verify", "snapshot"])
    ap.add_argument("--auto", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.action == "setup":
        setup()
    elif args.action == "teardown":
        sys.exit(teardown(args.auto, args.force))
    elif args.action == "verify":
        sys.exit(verify())
    else:
        print(json.dumps({k: (len(v) if isinstance(v, (list, dict)) else v) for k, v in snapshot().items()
                          if k != "powercfg_requests"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
