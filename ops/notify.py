"""Windows 토스트 알림 (PowerShell 내장 WinRT API, 추가 모듈 없음).

한글이 깨지지 않도록 스크립트를 UTF-16LE base64(-EncodedCommand)로 넘긴다.

사용법:
    python -m ops.notify "제목" "본문"
"""

import base64
import os
import subprocess
import sys

# PowerShell의 AppUserModelID: 별도 앱 등록 없이 토스트를 띄울 수 있다
APP_ID = r"{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\WindowsPowerShell\v1.0\powershell.exe"
CREATE_NO_WINDOW = 0x08000000


def _ps_quote(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


def toast(title: str, body: str, wait: bool = False) -> None:
    if os.environ.get("LOCO_NO_TOAST") == "1":  # 운영 장치 시험 중에는 알림을 띄우지 않는다
        return
    script = f"""
[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] > $null
$x = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent([Windows.UI.Notifications.ToastTemplateType]::ToastText02)
$t = $x.GetElementsByTagName('text')
$t.Item(0).AppendChild($x.CreateTextNode({_ps_quote(title)})) > $null
$t.Item(1).AppendChild($x.CreateTextNode({_ps_quote(body)})) > $null
$n = [Windows.UI.Notifications.ToastNotification]::new($x)
[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier({_ps_quote(APP_ID)}).Show($n)
"""
    enc = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    cmd = ["powershell.exe", "-NoProfile", "-NonInteractive", "-WindowStyle", "Hidden", "-EncodedCommand", enc]
    try:
        p = subprocess.Popen(cmd, creationflags=CREATE_NO_WINDOW,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if wait:
            p.wait(timeout=30)
    except Exception:  # noqa: BLE001 - 알림 실패가 학습 운영을 멈추면 안 된다
        pass


if __name__ == "__main__":
    toast(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "", wait=True)
