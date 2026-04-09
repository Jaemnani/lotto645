#!/bin/bash
# 매주 금요일 11시 실행
# crontab: 0 11 * * 5 /Users/ohyeahdani_m1/workspace/lotto645/cron.sh

PYTHON="/Users/ohyeahdani_m1/workspace/venv_common/bin/python"
ROOT="/Users/ohyeahdani_m1/workspace/lotto645"
LOG="$ROOT/logs/crawl_$(date +\%Y\%m\%d).log"

mkdir -p "$ROOT/logs"

echo "===== $(date '+%Y-%m-%d %H:%M:%S') 크롤링 시작 =====" >> "$LOG"

cd "$ROOT"
"$PYTHON" crawling/01_dh_caffe_crawling_with_auto_login.py >> "$LOG" 2>&1

echo "===== $(date '+%Y-%m-%d %H:%M:%S') 완료 =====" >> "$LOG"
