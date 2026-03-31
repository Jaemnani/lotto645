#!/bin/bash
# process.sh — 크롤링 → 학습 → 예측 파이프라인 (구매 제외)
#
# 사용법:
#   bash process.sh                  # 전체 실행 (크롤링+학습+예측, 공세트 1~5)
#   bash process.sh --skip-crawl     # 크롤링 건너뜀
#   bash process.sh --skip-train     # 학습 건너뜀
#   bash process.sh --skip-crawl --skip-train          # 예측만
#   bash process.sh --ball-set 3                       # 공세트 3만 예측
#   bash process.sh --skip-crawl --skip-train --ball-set 3  # 예측만, 공세트 3

set -e

PYTHON=/Users/jeremyye/workspace/venv/venv_default/bin/python
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR="$ROOT_DIR/model_m02_claude"
CRAWL_SCRIPT="$ROOT_DIR/crawling/01_dh_caffe_crawling_with_auto_login.py"

# ── 인자 파싱 ─────────────────────────────────────────────────────────────────
SKIP_CRAWL=0
SKIP_TRAIN=0
BALL_SET=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-crawl)  SKIP_CRAWL=1 ;;
        --skip-train)  SKIP_TRAIN=1 ;;
        --ball-set)    BALL_SET="$2"; shift ;;
        --ball-set=*)  BALL_SET="${1#*=}" ;;
        *) echo "알 수 없는 옵션: $1"; exit 1 ;;
    esac
    shift
done

# ── 실행 요약 ─────────────────────────────────────────────────────────────────
echo "========================================"
echo " 로또 파이프라인 시작 (구매 제외)"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo " 크롤링: $([ $SKIP_CRAWL -eq 1 ] && echo '건너뜀' || echo '실행')"
echo " 학습:   $([ $SKIP_TRAIN -eq 1 ] && echo '건너뜀' || echo '실행')"
echo " 예측:   공세트 ${BALL_SET:-1~5 전체}"
echo "========================================"

# ── Step 1: 크롤링 ───────────────────────────────────────────────────────────
if [ $SKIP_CRAWL -eq 0 ]; then
    echo ""
    echo "[Step 1] 데이터 크롤링..."
    cd "$ROOT_DIR/crawling"
    $PYTHON "$CRAWL_SCRIPT"
    cd "$ROOT_DIR"
    echo "[Step 1] 완료"
else
    echo ""
    echo "[Step 1] 크롤링 건너뜀"
fi

# ── Step 2: 학습 ─────────────────────────────────────────────────────────────
if [ $SKIP_TRAIN -eq 0 ]; then
    echo ""
    echo "[Step 2] 모델 학습..."
    cd "$MODEL_DIR"
    $PYTHON m02_train.py
    cd "$ROOT_DIR"
    echo "[Step 2] 완료"
else
    echo ""
    echo "[Step 2] 학습 건너뜀"
fi

# ── Step 3: 예측 ─────────────────────────────────────────────────────────────
echo ""
echo "[Step 3] 번호 예측..."
cd "$MODEL_DIR"

if [ -n "$BALL_SET" ]; then
    $PYTHON m02_run_all.py --ball_set "$BALL_SET"
else
    $PYTHON m02_run_all.py
fi

cd "$ROOT_DIR"
echo "[Step 3] 완료"

echo ""
echo "========================================"
echo " 파이프라인 완료"
echo " 구매까지 실행: bash process.sh && python pipeline.py"
echo "========================================"
