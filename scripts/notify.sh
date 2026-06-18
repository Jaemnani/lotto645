#!/bin/bash
# scripts/notify.sh — Discord Webhook 공용 알림 라이브러리
#
# 설계 (docs/05-ops-notifications.md):
#   - 미설정 = 무동작. DISCORD_WEBHOOK_URL 없으면 조용히 no-op → 기존 동작 불변.
#   - 로그를 데이터 소스로. 이미 찍는 요약 로그를 파싱해 1건 요약.
#   - 종료 경로 단일화(EXIT trap)로 성공/실패/크래시 모두 알림 1건.
#   - 실패 격리(타임아웃 + || true), 길이 truncate → 웹훅 장애가 본 작업을 안 죽임.
#
# 사용 (run 스크립트 상단):
#   source "$(dirname "$0")/scripts/notify.sh"   # 경로는 호출 스크립트 위치에 맞게
#   notify_start "크롤링" "$LOG"                  # 시작 알림 + EXIT trap 설치
#   ...본 작업...                                  # 종료 시 trap이 자동으로 digest 발송
#
# 테스트:
#   DISCORD_WEBHOOK_URL='<url>' bash scripts/notify.sh test

# ── .env 자동 로드 (cron 무수정 원칙: run 스크립트가 .env를 self-load) ──────────────
_notify_load_env() {
    local root="${NOTIFY_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
    if [ -z "$DISCORD_WEBHOOK_URL" ] && [ -f "$root/.env" ]; then
        # .env 의 DISCORD_WEBHOOK_URL 만 안전하게 추출 (전체 source 회피)
        local line
        line=$(grep -E '^[[:space:]]*DISCORD_WEBHOOK_URL=' "$root/.env" | tail -1)
        if [ -n "$line" ]; then
            DISCORD_WEBHOOK_URL="${line#*=}"
            DISCORD_WEBHOOK_URL="${DISCORD_WEBHOOK_URL%\"}"
            DISCORD_WEBHOOK_URL="${DISCORD_WEBHOOK_URL#\"}"
            export DISCORD_WEBHOOK_URL
        fi
    fi
}
_notify_load_env

# JSON 인코딩에 쓸 인터프리터 (venv python 우선, 없으면 system python3)
_NOTIFY_PY="${NOTIFY_PYTHON:-${PYTHON:-python3}}"

# ── discord_send <content> ───────────────────────────────────────────────────────
# DISCORD_WEBHOOK_URL 있으면 POST, 없으면 조용히 no-op. 실패해도 본 작업 안 죽임.
discord_send() {
    local content="$1"
    [ -z "$DISCORD_WEBHOOK_URL" ] && return 0

    # Discord content 2000자 제한 → 안전하게 truncate
    if [ "${#content}" -gt 1900 ]; then
        content="${content:0:1900}…(생략)"
    fi

    # 따옴표/줄바꿈/유니코드 안전을 위해 런타임 인터프리터로 JSON 직렬화 (jq 의존 없음)
    local payload
    payload=$(CONTENT="$content" "$_NOTIFY_PY" -c \
        'import json,os; print(json.dumps({"content": os.environ["CONTENT"]}))' 2>/dev/null)
    [ -z "$payload" ] && return 0

    curl -sS --max-time 10 \
        -H "Content-Type: application/json" \
        -d "$payload" \
        "$DISCORD_WEBHOOK_URL" >/dev/null 2>&1 || true
}

# ── discord_digest <log> <rc> <elapsed_sec> <label> ───────────────────────────────
# 런 로그 파일을 grep 해 상태/소요시간/단계결과/경고 를 한 건으로 요약 발송.
discord_digest() {
    local log="$1" rc="$2" elapsed="$3" label="$4"

    # 상태 판정: 종료코드 75 = IP 차단(docs/01), 0 = 정상, 그 외 = 실패
    local status
    case "$rc" in
        0)  status="✅ 성공" ;;
        75) status="⚠️ 차단(IP block, rc=75)" ;;
        *)  status="❌ 실패 (rc=$rc)" ;;
    esac

    local host; host="$(hostname -s 2>/dev/null || echo host)"
    local when; when="$(date '+%Y-%m-%d %H:%M:%S')"
    local elapsed_h="${elapsed}s"

    local msg="**[$label]** $status · ${elapsed_h} · \`$host\` · $when"

    if [ -f "$log" ]; then
        # 단계별 완료 라인 / 경고·스킵 라인 추출 (있으면)
        local steps warns errors
        steps=$(grep -aE '\[done\]|완료|\[Step [0-9]' "$log" 2>/dev/null | tail -8)
        warns=$(grep -aiE 'warn|경고|skip|건너뜀|차단' "$log" 2>/dev/null | tail -5)
        errors=$(grep -aiE 'error|traceback|exception|실패' "$log" 2>/dev/null | tail -5)

        [ -n "$steps" ]  && msg="$msg"$'\n'"— 단계 —"$'\n'"$steps"
        [ -n "$warns" ]  && msg="$msg"$'\n'"— 경고/스킵 —"$'\n'"$warns"
        [ "$rc" != "0" ] && [ -n "$errors" ] && msg="$msg"$'\n'"— 에러 —"$'\n'"$errors"
    fi

    discord_send "$msg"
}

# ── notify_start <label> <log> ────────────────────────────────────────────────────
# 시작 알림 1건 + 종료 시 digest 를 보내는 EXIT trap 설치.
# 정상 종료(exit 0)·조기종료·크래시·차단(75) 모든 경로에서 알림 1건 보장.
notify_start() {
    NOTIFY_LABEL="$1"
    NOTIFY_LOG="$2"
    NOTIFY_START_TS=$(date +%s)

    discord_send "**[$NOTIFY_LABEL]** ▶️ 시작 · \`$(hostname -s 2>/dev/null || echo host)\` · $(date '+%Y-%m-%d %H:%M:%S')"

    trap '_notify_on_exit' EXIT
}

_notify_on_exit() {
    local rc=$?
    local elapsed=$(( $(date +%s) - ${NOTIFY_START_TS:-$(date +%s)} ))
    discord_digest "$NOTIFY_LOG" "$rc" "$elapsed" "$NOTIFY_LABEL"
    return $rc
}

# ── CLI: 테스트 ───────────────────────────────────────────────────────────────────
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    case "${1:-}" in
        test)
            if [ -z "$DISCORD_WEBHOOK_URL" ]; then
                echo "[notify] DISCORD_WEBHOOK_URL 미설정 — no-op 동작 확인 (메시지 안 보냄)"
                exit 0
            fi
            discord_send "✅ lotto645 notify.sh 테스트 · \`$(hostname -s 2>/dev/null || echo host)\` · $(date '+%Y-%m-%d %H:%M:%S')"
            echo "[notify] 테스트 메시지 발송 시도 완료"
            ;;
        *)
            echo "사용법: bash scripts/notify.sh test"
            ;;
    esac
fi
