# Synology 배포 — lotto645

오라클 프리티어에서 돌던 FastAPI 로또 서비스를 **Synology NAS(Container Manager / Docker Compose)** 에서도
돌리기 위한 배포 묶음. 근거 문서: [docs/02-backend.md](../../docs/02-backend.md),
[docs/06-infra-deployment.md](../../docs/06-infra-deployment.md), [docs/05-ops-notifications.md](../../docs/05-ops-notifications.md).

## 두 가지 모드

| | (A) managed Supabase | (B) self-host |
|---|---|---|
| 띄우는 컨테이너 | `web` | `web` + `db`(Postgres) + `rest`(PostgREST) + `proxy`(Caddy) |
| DB | 클라우드 Supabase (오라클과 공유) | NAS 안 Postgres (완전 독립) |
| `SUPABASE_URL` | `https://xxx.supabase.co` | `http://proxy` |
| 외부 노출 | DSM 역프록시 443 → `<NAS_IP>:8000` | DSM 역프록시 443 → `<NAS_IP>:8080`(Caddy) |
| 실행 | `docker compose up -d --build` | `docker compose --profile selfhost up -d --build` |
| 추천 | **오라클과 같은 DB 그대로 쓰며 web 만 NAS 이중화** | **오라클/Supabase 의존 끊고 NAS 단독 운영** |

> 이 서비스는 이미지 저장이 없어 MinIO(오브젝트 스토리지)는 쓰지 않는다 (docs/02 의 MinIO 절차는 생략).

---

## 사전 준비 (공통)

```bash
# NAS 에서 (SSH)
cd /volume1/docker
git clone https://github.com/Jaemnani/lotto645.git
cd lotto645/deploy/synology
cp .env.example .env
```

`.env` 공통 항목 채우기:
```dotenv
ADMIN_KEY=<openssl rand -hex 32>
DEPLOY_ENV=synology
DISCORD_WEBHOOK_URL=<없으면 비워둠 → 알림 자동 off>
```

---

## 모드 (A) managed Supabase — 가장 단순

`.env`:
```dotenv
SUPABASE_URL=https://YOUR_PROJECT.supabase.co
SUPABASE_KEY=YOUR_ANON_KEY
```
```bash
sudo docker compose up -d --build
sudo docker compose ps
curl -s http://localhost:8000/health     # {"status":"ok"}
```
끝. 오라클과 **같은 Supabase DB** 를 바라보므로 데이터/스케줄러가 그대로 동작한다.
(두 서버가 동시에 토요일 잡을 돌리면 중복 알림이 올 수 있다 — 아래 "이중화 주의" 참고.)

---

## 모드 (B) self-host — DB 까지 NAS 단독

### 1) 시크릿 생성 → `.env`
```bash
echo "POSTGRES_PASSWORD=$(openssl rand -base64 24)"
echo "AUTHENTICATOR_PASSWORD=$(openssl rand -base64 24)"
echo "JWT_SECRET=$(openssl rand -base64 48)"
```
세 값을 `.env` 에 붙여넣는다. 이어서 anon/service 키 발급:
```bash
python3 gen_keys.py          # .env 의 JWT_SECRET 으로 ANON_KEY / SERVICE_KEY 출력
```
`.env` 마무리:
```dotenv
SUPABASE_URL=http://proxy
SUPABASE_KEY=<위 ANON_KEY>
# SUPABASE_SERVICE_KEY=<위 SERVICE_KEY>   # 크롤러 적재용(웹 서빙엔 불필요)
```

### 2) bind-mount 디렉토리 생성 + 기동
```bash
mkdir -p volumes/db/data volumes/caddy/data volumes/caddy/config
sudo docker compose --profile selfhost up -d --build
sudo docker compose ps                       # db/rest/proxy/web 정상
sudo docker compose logs -f db | grep -m1 "database system is ready"
```
최초 기동 시 `db/00_roles.sh` → 마이그레이션(`10_*`) → `db/90_rls.sql` 가 1회 자동 실행
(역할 4종 + 테이블 + RLS public read).

### 3) 검증
```bash
docker exec -it lotto645-db psql -U postgres -c "\du"     # anon/authenticated/service_role/authenticator
docker exec -it lotto645-db psql -U postgres -c "\dt"     # draw_results 등 3테이블
curl -s http://localhost:8080/health                      # Caddy 200
curl -s -H "Authorization: Bearer $SUPABASE_KEY" \
  "http://localhost:8080/rest/v1/draw_results?select=round&limit=1"   # [] 또는 데이터(정상)
```
> 401 = ANON_KEY ↔ JWT_SECRET 불일치 · 404 = 마이그레이션 미적용. (docs/02 트러블슈팅 표)

### 4) 데이터 이관 (오라클/Supabase → NAS, 선택)
```bash
# 기존 Supabase 에서 덤프 → NAS Postgres 로 복원 (3 테이블만)
pg_dump "<기존_DATABASE_URL>" -t draw_results -t user_extractions -t weekly_announcements --data-only \
  | docker exec -i lotto645-db psql -U postgres
docker exec -it lotto645-db psql -U postgres -c "NOTIFY pgrst,'reload schema'"
```

---

## 외부 노출 (DSM)

1. **인증서**: 제어판 → 보안 → 인증서 → Let's Encrypt (`<DOMAIN>` + 이메일).
2. **역방향 프록시**: 제어판 → 로그인 포털 → 고급 → 역방향 프록시 → 생성
   - 소스: HTTPS / `<DOMAIN>` / 443
   - 대상: HTTP / `localhost` / **8000**(모드 A) 또는 **8080**(모드 B, Caddy)
3. **포트포워딩**: 라우터 443 → `<NAS_IP>:443`. (또는 Cloudflare Tunnel 로 포트포워딩 생략)

> 내부 포트(8000/8080)는 LAN 전용. 공개는 역프록시(443) 경로로만.

---

## Discord 알림 (cron / 크롤링 / 스케줄러)

웹훅 1개로 끝 (docs/05). `DISCORD_WEBHOOK_URL` 만 설정하면 자동 on, 없으면 no-op.

- **컨테이너(스케줄러)**: `.env` 의 `DISCORD_WEBHOOK_URL` → 토요일 통계 시작/완료/실패, 신규 회차
  감지→재학습, 재학습 완료/실패 알림 (`web/notify.py`).
- **크롤러 머신(아이맥)**: 루트 `.env` 에 같은 키 → `cron.sh`/`process.sh`/`mac_crawl.sh` 가
  `scripts/notify.sh` 로 시작 + 종료 요약(성공/차단/실패) 알림.

테스트:
```bash
DISCORD_WEBHOOK_URL='<url>' bash ../../scripts/notify.sh test   # 크롤러 머신
docker exec lotto645-web python -c "from web.notify import discord_send; discord_send('✅ synology web 테스트')"
```

---

## 운영

```bash
sudo docker compose [--profile selfhost] up -d --build   # 코드 갱신 후 재배포
sudo docker compose logs -f web                          # 로그
sudo docker compose --profile selfhost up -d --force-recreate proxy   # Caddyfile 변경 반영
```

### 이중화 주의 (오라클 + NAS 동시 운영 시)
- 같은 Supabase 를 보는 모드(A)에서 두 서버가 다 켜져 있으면 **토요일 통계/재학습 잡이 양쪽에서**
  돈다 → 작업 자체는 멱등(upsert)이라 데이터는 안전하지만 **알림이 2배**. 한쪽만 스케줄러를 켜거나
  `DEPLOY_ENV` 로 구분해 한쪽 알림만 받는 걸 권장.
- NAS 를 주(主)로 전환하면 오라클 systemd 서비스(`lotto645`)를 stop.

### 백업 (self-host)
```bash
docker exec lotto645-db pg_dump -U postgres -Fc postgres > backups/db_$(date +%F).dump
# DSM Hyper Backup 대상에 deploy/synology/volumes/ 포함
```
