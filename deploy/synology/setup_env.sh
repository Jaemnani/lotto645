#!/bin/bash
# setup_env.sh — self-host(모드 B) .env 시크릿 자동 세팅
# ====================================================
# 하는 일 (deploy/synology 디렉토리 기준):
#   1) .env 없으면 .env.example 에서 생성
#   2) POSTGRES_PASSWORD / AUTHENTICATOR_PASSWORD / JWT_SECRET / ADMIN_KEY 생성
#      (이미 값이 있으면 보존 — 재실행해도 키가 안 바뀜. 강제 교체는 --force)
#   3) SUPABASE_URL=http://proxy, DEPLOY_ENV=synology 설정
#   4) JWT_SECRET 으로 anon/service 키 발급 → SUPABASE_KEY / SUPABASE_SERVICE_KEY 기록
#   5) (옵션) --webhook <URL> 로 DISCORD_WEBHOOK_URL 설정
#
# 사용:
#   bash setup_env.sh                          # 비어있는 시크릿만 생성(안전, 멱등)
#   bash setup_env.sh --webhook https://...    # 디스코드 웹훅까지
#   bash setup_env.sh --force                  # 모든 시크릿 재생성 (⚠️ 아래 경고)
#
# ⚠️ --force 로 JWT_SECRET/비밀번호를 바꾸면, 이미 초기화된 DB(volumes/db/data)와
#    어긋나 접속이 깨진다. 처음부터 다시 할 때만 쓰고, volumes/db/data 도 비워야 함.

set -e
cd "$(dirname "$0")"

ENV_FILE=".env"
FORCE=0
WEBHOOK=""

while [ $# -gt 0 ]; do
    case "$1" in
        --force)   FORCE=1 ;;
        --webhook) WEBHOOK="$2"; shift ;;
        --webhook=*) WEBHOOK="${1#*=}" ;;
        -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "[ERROR] 알 수 없는 옵션: $1"; exit 1 ;;
    esac
    shift
done

# ── 의존성 확인 ──────────────────────────────────────────────────────────────────
command -v openssl >/dev/null 2>&1 || { echo "[ERROR] openssl 필요"; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "[ERROR] python3 필요"; exit 1; }

# ── .env 준비 ────────────────────────────────────────────────────────────────────
if [ ! -f "$ENV_FILE" ]; then
    cp .env.example "$ENV_FILE"
    echo "[+] .env 생성 (.env.example 복사)"
fi

# ── KEY=VALUE 읽기/쓰기 헬퍼 (인라인 주석/특수문자 안전) ───────────────────────────
get_kv() {
    grep -E "^$1=" "$ENV_FILE" 2>/dev/null | tail -1 \
        | sed -E "s/^$1=//" | sed -E 's/[[:space:]]*#.*$//' | sed -E 's/[[:space:]]+$//'
}
set_kv() {
    # 값은 ENVIRON 으로 awk 에 전달 → / + = . 등 이스케이프 불필요. 인라인 주석은 제거됨.
    KV_KEY="$1" KV_VAL="$2" awk '
        BEGIN { k = ENVIRON["KV_KEY"]; v = ENVIRON["KV_VAL"]; done = 0 }
        $0 ~ "^" k "=" { print k "=" v; done = 1; next }
        { print }
        END { if (!done) print k "=" v }
    ' "$ENV_FILE" > "$ENV_FILE.tmp" && mv "$ENV_FILE.tmp" "$ENV_FILE"
}

# 비어있을 때만(또는 --force) 생성, 아니면 보존
gen_secret() {  # gen_secret <KEY> <생성명령>
    local key="$1" cmd="$2" cur
    cur="$(get_kv "$key")"
    if [ -n "$cur" ] && [ "$FORCE" -eq 0 ]; then
        echo "[=] $key 보존 (기존 값 유지)"
        return
    fi
    set_kv "$key" "$(eval "$cmd")"
    echo "[+] $key $([ -n "$cur" ] && echo 재생성 || echo 생성)"
}

# ── 1) 시크릿 ────────────────────────────────────────────────────────────────────
gen_secret POSTGRES_PASSWORD      "openssl rand -base64 24"
gen_secret AUTHENTICATOR_PASSWORD "openssl rand -base64 24"
gen_secret JWT_SECRET             "openssl rand -base64 48"
gen_secret ADMIN_KEY              "openssl rand -hex 32"

# ── 2) 고정 값 (모드 B) ──────────────────────────────────────────────────────────
set_kv SUPABASE_URL "http://proxy"
echo "[+] SUPABASE_URL=http://proxy"
[ -z "$(get_kv DEPLOY_ENV)" ] && set_kv DEPLOY_ENV "synology" && echo "[+] DEPLOY_ENV=synology"

# ── 3) 디스코드 웹훅 (옵션) ──────────────────────────────────────────────────────
if [ -n "$WEBHOOK" ]; then
    set_kv DISCORD_WEBHOOK_URL "$WEBHOOK"
    echo "[+] DISCORD_WEBHOOK_URL 설정"
fi

# ── 4) anon / service 키 발급 (현재 JWT_SECRET 기준) ─────────────────────────────
KEYS_OUT="$(python3 gen_keys.py)"
ANON_KEY="$(echo "$KEYS_OUT"    | grep -E '^ANON_KEY='    | sed -E 's/^ANON_KEY=//')"
SERVICE_KEY="$(echo "$KEYS_OUT" | grep -E '^SERVICE_KEY=' | sed -E 's/^SERVICE_KEY=//')"
if [ -z "$ANON_KEY" ]; then
    echo "[ERROR] anon 키 발급 실패 (gen_keys.py 출력 확인)"; echo "$KEYS_OUT"; exit 1
fi
set_kv SUPABASE_KEY         "$ANON_KEY"
set_kv SUPABASE_SERVICE_KEY "$SERVICE_KEY"
echo "[+] SUPABASE_KEY / SUPABASE_SERVICE_KEY 발급 (JWT_SECRET 기준)"

# ── 요약 (시크릿은 마스킹) ───────────────────────────────────────────────────────
mask() { local v; v="$(get_kv "$1")"; [ -z "$v" ] && { echo "(빈값)"; return; }; echo "${v:0:6}…(len=${#v})"; }
echo ""
echo "===== .env 세팅 완료 ====="
echo "  SUPABASE_URL        = $(get_kv SUPABASE_URL)"
echo "  SUPABASE_KEY        = $(mask SUPABASE_KEY)"
echo "  POSTGRES_PASSWORD   = $(mask POSTGRES_PASSWORD)"
echo "  AUTHENTICATOR_PASS  = $(mask AUTHENTICATOR_PASSWORD)"
echo "  JWT_SECRET          = $(mask JWT_SECRET)"
echo "  ADMIN_KEY           = $(mask ADMIN_KEY)"
echo "  DEPLOY_ENV          = $(get_kv DEPLOY_ENV)"
echo "  WEB_PORT/PROXY_PORT = $(get_kv WEB_PORT) / $(get_kv PROXY_PORT)"
echo "  DISCORD_WEBHOOK_URL = $([ -n "$(get_kv DISCORD_WEBHOOK_URL)" ] && echo '설정됨' || echo '(빈값 → 알림 off)')"
echo ""
echo "다음 단계:"
echo "  sudo mkdir -p volumes/db/data volumes/caddy/data volumes/caddy/config"
echo "  sudo docker compose --profile selfhost up -d --build"
