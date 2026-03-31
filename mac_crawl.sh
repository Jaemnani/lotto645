#!/bin/bash
# mac_crawl.sh — Mac Mini 전용 크롤링 스크립트
#
# 역할: 데이터 크롤링 후 변경이 있을 때만 GitHub에 push
#       GitHub Actions가 push를 감지하여 학습+구매 파이프라인 시작
#
# crontab 등록 (매주 목요일 오전 9시 KST):
#   0 0 * * 4 bash /Users/jeremyye/workspace/lotto645/mac_crawl.sh >> /Users/jeremyye/workspace/lotto645/crawl_log.txt 2>&1
#
# crontab 등록 방법:
#   crontab -e  →  위 줄 추가 후 저장

set -e

PYTHON=/Users/jeremyye/workspace/venv/venv_default/bin/python
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_FILE="$ROOT_DIR/data/history_from_cafe.csv"

echo "========================================"
echo " Mac Mini 크롤링 시작"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"

cd "$ROOT_DIR"

# ── 크롤링 (갱신 필요시에만 실행됨) ──────────────────────────────────────────
cd "$ROOT_DIR/crawling"
$PYTHON 01_dh_caffe_crawling_with_auto_login.py
cd "$ROOT_DIR"

# ── 변경 여부 확인 후 push ────────────────────────────────────────────────────
if git diff --quiet "$DATA_FILE"; then
    echo ""
    echo "데이터 변경 없음 → push 생략 (GitHub Actions 미실행)"
else
    echo ""
    echo "새 데이터 감지 → GitHub에 push..."

    LATEST_ROUND=$(tail -1 "$DATA_FILE" | cut -d',' -f2)
    git add "$DATA_FILE"
    git commit -m "data: 제${LATEST_ROUND}회 추첨 데이터 추가 ($(date '+%Y-%m-%d'))"
    git push origin master

    echo "push 완료 → GitHub Actions 자동 실행 예정"
fi

echo ""
echo "========================================"
echo " 크롤링 완료: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
