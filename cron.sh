#!/bin/bash
# 매주 금요일 11시 실행
# crontab: 0 11 * * 5 /Users/ohyeahdani_m1/workspace/lotto645/cron.sh

PYTHON="/Users/ohyeahdani_m1/workspace/venv_common/bin/python"
ROOT="/Users/ohyeahdani_m1/workspace/lotto645"
LOG="$ROOT/logs/crawl_$(date +\%Y\%m\%d).log"

mkdir -p "$ROOT/logs"

# Discord 알림 (DISCORD_WEBHOOK_URL 없으면 자동 no-op)
NOTIFY_ROOT="$ROOT" NOTIFY_PYTHON="$PYTHON" source "$ROOT/scripts/notify.sh"
notify_start "매일 크롤링" "$LOG"   # 시작 알림 + EXIT trap(종료 요약)

echo "===== $(date '+%Y-%m-%d %H:%M:%S') 크롤링 시작 =====" >> "$LOG"

cd "$ROOT"
"$PYTHON" crawling/01_dh_caffe_crawling_with_auto_login.py >> "$LOG" 2>&1

# m04 (Mitra-v2) 다음 회차 확률표 → Supabase model_predictions
#   전용 venv(model_m04_mitra/requirements.txt)가 없으면 건너뜀. 실패해도 크롤링 결과에는 영향 없음.
M04_PYTHON="${M04_PYTHON:-/Users/ohyeahdani_m1/workspace/venv_m04/bin/python}"
if [ -x "$M04_PYTHON" ]; then
    echo "----- $(date '+%Y-%m-%d %H:%M:%S') m04 예측 시작 -----" >> "$LOG"
    if ! "$M04_PYTHON" model_m04_mitra/m04_predict.py --source supabase >> "$LOG" 2>&1; then
        echo "[m04] 예측 실패 — 서버는 m03 로 폴백" >> "$LOG"
        discord_send "⚠️ m04 예측 실패 (서버는 m03 폴백) · 로그: $LOG"
    fi
else
    echo "[m04] $M04_PYTHON 없음 — m04 예측 건너뜀" >> "$LOG"
fi

echo "===== $(date '+%Y-%m-%d %H:%M:%S') 완료 =====" >> "$LOG"
