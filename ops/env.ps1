# LOCO 학습 운영 환경 설정·원복·검증. 실제 동작은 ops/envctl.py에 있다.
#   .\ops\env.ps1 -Action setup      사전 스냅샷 + 예약 작업 \LOCO-Train\Supervisor 등록
#   .\ops\env.ps1 -Action teardown   즉시 원복 (실행 중 학습이 있으면 -Force 필요)
#   .\ops\env.ps1 -Action verify     잔여 변경 확인
param(
    [Parameter(Mandatory = $true)][ValidateSet('setup', 'teardown', 'verify')][string]$Action,
    [switch]$Force
)
$root = Split-Path -Parent $PSScriptRoot
$env:PYTHONIOENCODING = 'utf-8'
[Console]::OutputEncoding = [Text.Encoding]::UTF8
Push-Location $root
try {
    $a = @('-m', 'ops.envctl', $Action)
    if ($Force) { $a += '--force' }
    & python @a
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
