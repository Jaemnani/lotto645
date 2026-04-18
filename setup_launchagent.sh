#!/bin/bash
# setup_launchagent.sh — 파이프라인 LaunchAgent 자동 설정 스크립트
# 아이맥에서 실행: bash setup_launchagent.sh

set -e

LABEL="com.lotto645.pipeline"
PLIST="$HOME/Library/LaunchAgents/${LABEL}.plist"
PYTHON="$HOME/workspace/venv_common/bin/python"
ROOT="$HOME/workspace/lotto645"
LOG_DIR="$ROOT/logs"

# ── 사전 검증 ────────────────────────────────────────────────────────────────────
if [ ! -f "$PYTHON" ]; then
    echo "[ERROR] Python not found: $PYTHON"
    exit 1
fi

if [ ! -f "$ROOT/pipeline.py" ]; then
    echo "[ERROR] pipeline.py not found: $ROOT/pipeline.py"
    exit 1
fi

# ── 기존 등록 해제 ───────────────────────────────────────────────────────────────
if launchctl list 2>/dev/null | grep -q "$LABEL"; then
    echo "[INFO] 기존 LaunchAgent 해제 중..."
    launchctl unload "$PLIST" 2>/dev/null || true
fi

# ── logs 디렉토리 생성 ───────────────────────────────────────────────────────────
mkdir -p "$LOG_DIR"

# ── plist 파일 생성 ──────────────────────────────────────────────────────────────
cat > "$PLIST" << PLISTEOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>${LABEL}</string>

    <key>ProgramArguments</key>
    <array>
        <string>${PYTHON}</string>
        <string>${ROOT}/pipeline.py</string>
    </array>

    <key>WorkingDirectory</key>
    <string>${ROOT}</string>

    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin</string>
    </dict>

    <!-- 매주 금요일 10:00 KST -->
    <key>StartCalendarInterval</key>
    <dict>
        <key>Weekday</key>
        <integer>5</integer>
        <key>Hour</key>
        <integer>10</integer>
        <key>Minute</key>
        <integer>0</integer>
    </dict>

    <key>StandardOutPath</key>
    <string>${LOG_DIR}/pipeline_stdout.log</string>
    <key>StandardErrorPath</key>
    <string>${LOG_DIR}/pipeline_stderr.log</string>
</dict>
</plist>
PLISTEOF

echo "[OK] plist 생성: $PLIST"

# ── 등록 ─────────────────────────────────────────────────────────────────────────
launchctl load "$PLIST"
echo "[OK] LaunchAgent 등록 완료"

# ── 등록 확인 ────────────────────────────────────────────────────────────────────
echo ""
echo "=== 등록 상태 ==="
launchctl list | grep "$LABEL" || echo "[WARN] 등록 확인 실패"

echo ""
echo "=== 설정 요약 ==="
echo "  plist:  $PLIST"
echo "  python: $PYTHON"
echo "  script: $ROOT/pipeline.py"
echo "  logs:   $LOG_DIR/"
echo "  실행:   매주 금요일 10:00"
echo ""
echo "수동 테스트: launchctl start $LABEL"
echo "로그 확인:   tail -f $LOG_DIR/pipeline_stdout.log"
echo "해제:        launchctl unload $PLIST"
