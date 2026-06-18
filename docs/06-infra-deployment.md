# 06 · 인프라 / 배포

API 연동은 아니지만, 위 연동들이 **외부에 도달 가능하도록** 묶는 네트워킹·호스팅 계층. self-host
백엔드를 공개 인터넷에 안전하게 노출하고, 프론트는 별도 PaaS에 둔다.

이 문서 하나로 self-host 백엔드 노출 + 프론트 배포를 재현할 수 있도록 구체 절차를 포함한다.
placeholder: `<NAS_IP>` · `<DOMAIN>`(예 your.synology.me) · `<COMPOSE>`(compose 프로젝트명).

## 구성 한눈에

```
[브라우저/PaaS]
      │ https (443)
      ▼
[Cloudflare Tunnel] ──→ [역방향 프록시(DSM 등)] ──→ [Caddy :80] ──┬─ /rest/v1/* → PostgREST
   (외부 진입, TLS)        (도메인 Let's Encrypt)    (경로 라우팅)   └─ /storage/* → MinIO :9000
```

## 사전 조건
DSM 7.2(또는 동등 호스트) · Container Manager/Docker · SSH 활성화 · Cloudflare 연결 도메인.
repo를 호스트(`/volume1/docker/<proj>` 등)에 두고 `deploy/<host>/`를 작업 디렉토리로.

## 구성요소 + 설정 절차

### 1) 컨테이너 기동 (Docker Compose)
서비스: `db`(Postgres/PostGIS) · `rest`(PostgREST) · `storage`(MinIO, `9000`/`9001`) ·
`proxy`(Caddy, `8080:80`) · (옵션) `pgadmin`.
```bash
# bind-mount 디렉토리 먼저 생성 (compose가 자동 생성 안 함, git 제외됨)
mkdir -p volumes/db/data volumes/storage volumes/caddy/data volumes/caddy/config
sudo docker compose up -d
sudo docker compose ps            # db/rest/storage/proxy healthy/running
sudo docker compose logs -f db    # "database system is ready" 확인
```
- 내부 통신은 docker 네트워크(서비스명 해석). 외부엔 역프록시 경로만 노출.
- S3 포트(9000)·프록시 포트(8080)는 **LAN 전용** — 외부는 Tunnel(443)로만.
- (DB 역할/시크릿/마이그레이션/키 발급은 [02-backend.md](02-backend.md) 참고.)

### 2) Caddy — 경로 기반 라우팅 (API 게이트웨이 대체)
DDNS가 서브도메인(와일드카드)을 못 만들 때, **단일 호스트 + 경로**로 분기:
```caddy
:80 {
    handle_path /rest/v1/* { reverse_proxy rest:3000 }     # supabase-js가 ${URL}/rest/v1/* 호출
    handle_path /storage/* { reverse_proxy storage:9000 }  # 공개 이미지 GET → {bucket}/{key}
    respond /health 200
    handle { respond "use /rest/v1/* or /storage/*" 404 }
}
```
- 매니지드 Supabase의 Kong 게이트웨이를 self-host에서 대체.
- 설정은 컨테이너에 **bind-mount** → 변경 후 **recreate/restart** 해야 반영(중요):
  ```bash
  cd deploy/<host> && git pull
  sudo docker compose up -d --force-recreate proxy
  curl -s https://<DOMAIN>/health    # 200 = 새 설정 반영
  ```

### 3) 외부 노출 — 역방향 프록시 + 인증서 + 포트포워딩
**인증서**: DSM → 제어판 → 보안 → 인증서 → 추가 → Let's Encrypt (`<DOMAIN>` + 이메일, 자동 갱신).
**역방향 프록시**: DSM → 제어판 → 로그인 포털 → 고급 → 역방향 프록시 → 생성
- 소스: **HTTPS / `<DOMAIN>` / 443**
- 대상: **HTTP / `localhost` / 8080** (Caddy)
- 저장 후 보안 → 인증서 → 설정에서 이 서비스에 위 인증서 지정.

**포트포워딩**: 라우터 외부 `443` → `<NAS_IP>:443`.
> 외부 진입점은 `https://<DOMAIN>` 하나. REST=`/rest/v1/*`, 이미지=`/storage/{bucket}/{key}`.
> (Cloudflare Tunnel을 쓰면 포트포워딩 없이 `cloudflared`가 아웃바운드로 대체 가능.)

### 4) 프론트 호스팅 (Vercel 등 PaaS)
웹은 PaaS에 배포, 백엔드는 공개 도메인으로 호출. **env는 플랫폼 대시보드에 설정 + 재배포**.
```
SUPABASE_URL=https://<DOMAIN>
SUPABASE_KEY=<ANON_KEY>
STORAGE_PUBLIC_URL=https://<DOMAIN>/storage
DATA_GO_KR_API_KEY=<서버 전용 OpenAPI 키>
```
> ISR/캐시가 있으면 빌드시 박히는 값(이미지 URL 등)은 **재배포**로 갱신.

### 5) 크롤러 머신 env (LAN 내부 IP 직접)
```dotenv
SUPABASE_URL=http://<NAS_IP>:8080
SUPABASE_KEY=<ANON_KEY>
SUPABASE_SERVICE_KEY=<SERVICE_KEY>
MINIO_ENDPOINT=http://<NAS_IP>:9000
MINIO_ACCESS_KEY=<MINIO_ROOT_USER>
MINIO_SECRET_KEY=<MINIO_ROOT_PASSWORD>
STORAGE_PUBLIC_URL=https://<DOMAIN>/storage
# + 보강 API 키(03), DISCORD_WEBHOOK_URL(05)
```
> 크롤러는 LAN에서 `:8080`/`:9000` 직결(빠름), 공개 URL 생성만 `STORAGE_PUBLIC_URL` 사용.

## ⚠️ 운영 함정 (실제 겪음)
- **프록시 설정 변경이 런타임에 안 붙음** — bind-mount면 컨테이너 recreate 필요. `/health`로 반영 확인.
- **공개 URL은 포트 없이.** 외부는 역프록시(443) 경로로만. 내부 포트(8080/9000) 직결은 연결거부가 정상.
- **PaaS env 누락** — 프론트가 백엔드/스토리지 URL 못 찾아 404. 대시보드 설정 + 재배포 확인.
- **명령은 올바른 호스트에서.** 컨테이너 조작은 NAS 쉘에서, 크롤러 조작은 크롤러 머신에서.

## 진단 빠른 길
```bash
curl -s -o /dev/null -w "%{http_code}\n" https://<DOMAIN>/health        # 프록시/Caddy 살아있나(200)
curl -s -o /dev/null -w "%{http_code}\n" https://<DOMAIN>/rest/v1/      # DB 라우트
curl -s -o /dev/null -w "%{http_code} %{content_type}\n" \
  https://<DOMAIN>/storage/<bucket>/<key>                              # 스토리지(200 image/*)
```

## 백업
```bash
docker exec <db_container> pg_dump -U postgres -Fc postgres > /backups/db_$(date +%F).dump
# 스토리지: MinIO volumes/storage/ 를 백업 솔루션(Hyper Backup 등) 대상에 포함
```

## 재사용 체크리스트
- [ ] bind-mount 디렉토리 생성 → `compose up -d` → 헬스 확인.
- [ ] 게이트웨이(Caddy) 경로 라우팅: REST/Storage/health.
- [ ] 인증서(Let's Encrypt) + 역방향 프록시 + 443 진입(포트포워딩 또는 Tunnel).
- [ ] 내부 포트 LAN 전용, 공개는 프록시 경로만.
- [ ] PaaS env 설정 + 재배포, `/health`로 반영 확인.
- [ ] pg_dump + 스토리지 볼륨 백업.
