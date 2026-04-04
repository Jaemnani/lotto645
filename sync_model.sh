#!/bin/bash
# Oracle Cloud에서 실행 - 맥미니가 학습한 최신 모델/데이터 동기화
# crontab: 0 2 * * 1 /path/to/lotto645/sync_model.sh >> /var/log/lotto_sync.log 2>&1
# (매주 월요일 02:00에 실행 - 금요일 파이프라인 완료 후 반영)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 모델 동기화 시작"
git pull origin master
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 동기화 완료 (최신 model + CSV 반영)"
