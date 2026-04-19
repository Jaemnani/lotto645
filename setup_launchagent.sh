#!/bin/bash
# setup_launchagent.sh — iMac LaunchAgent 자동 설정
#
# 2개의 LaunchAgent 등록:
#   1) com.lotto645.daily-crawl  : 매일 11:00 크롤링 (cron.sh)
#   2) com.lotto645.friday-buy   : 매주 금요일 10:00 파이프라인 (구매 포함)
#
# 실행: bash setup_launchagent.sh

set -e

PYTHON="$HOME/workspace/venv_common/bin/python"
ROOT="$HOME/workspace/lotto645"
LOG_DIR="$ROOT/logs"
LA_DIR="$HOME/Library/LaunchAgents"

LABEL_CRAWL="com.lotto645.daily-crawl"
LABEL_BUY="com.lotto645.friday-buy"
PLIST_CRAWL="$LA_DIR/${LABEL_CRAWL}.plist"
PLIST_BUY="$LA_DIR/${LABEL_BUY}.plist"

# ── 사전 검증 ────────────────────────────────────────────────────────────────────
if [ ! -f "$PYTHON" ]; then
    echo "[ERROR] Python not found: $PYTHON"
    exit 1
fi
if [ ! -f "$ROOT/cron.sh" ]; then
    echo "[ERROR] cron.sh not found: $ROOT/cron.sh"
    exit 1
fi
if [ ! -f "$ROOT/pipeline.py" ]; then
    echo "[ERROR] pipeline.py not found: $ROOT/pipeline.py"
    exit 1
fi

mkdir -p "$LOG_DIR" "$LA_DIR"

# ── 1) 매일 크롤링 ──────────────────────────────────────────────────────────────
launchctl unload "$PLIST_CRAWL" 2>/dev/null || true
cat > "$PLIST_CRAWL" << PLISTEOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>${LABEL_CRAWL}</string>

    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>${ROOT}/cron.sh</string>
    </array>

    <key>WorkingDirectory</key>
    <string>${ROOT}</string>

    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin</string>
    </dict>

    <!-- 매일 11:00 KST -->
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>11</integer>
        <key>Minute</key>
        <integer>0</integer>
    </dict>

    <key>StandardOutPath</key>
    <string>${LOG_DIR}/daily_crawl_stdout.log</string>
    <key>StandardErrorPath</key>
    <string>${LOG_DIR}/daily_crawl_stderr.log</string>
</dict>
</plist>
PLISTEOF
launchctl load "$PLIST_CRAWL"
echo "[OK] 등록: $LABEL_CRAWL  (매일 11:00 크롤링)"

# ── 2) 매주 금요일 구매 ─────────────────────────────────────────────────────────
launchctl unload "$PLIST_BUY" 2>/dev/null || true
cat > "$PLIST_BUY" << PLISTEOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>${LABEL_BUY}</string>

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
    <string>${LOG_DIR}/friday_buy_stdout.log</string>
    <key>StandardErrorPath</key>
    <string>${LOG_DIR}/friday_buy_stderr.log</string>
</dict>
</plist>
PLISTEOF
launchctl load "$PLIST_BUY"
echo "[OK] 등록: $LABEL_BUY  (매주 금요일 10:00 파이프라인)"

echo ""
echo "=== 등록 상태 ==="
launchctl list | grep lotto645 || echo "[WARN] 등록 확인 실패"

echo ""
echo "=== 관리 명령어 ==="
echo "수동 실행: launchctl start $LABEL_CRAWL"
echo "해제:      launchctl unload $PLIST_CRAWL"
echo "로그:      tail -f $LOG_DIR/daily_crawl_stdout.log"
