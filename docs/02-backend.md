# 02 · 백엔드 (DB + 오브젝트 스토리지)

영속 데이터(관계형 DB)와 바이너리/이미지(오브젝트 스토리지)를 다루는 연동. 매니지드(Supabase
클라우드)와 self-host(Postgres + PostgREST + MinIO)를 **같은 코드로** 쓰도록 추상화한다.

이 문서 하나로 self-host 백엔드를 처음부터 구축할 수 있도록 구체 절차를 포함한다.
placeholder: `<NAS_IP>`(예 192.168.50.201) · `<DOMAIN>`(예 your.synology.me) · `<COMPOSE>`(compose 프로젝트명).

---

## 2-1. DB — PostgREST / Supabase

PostgreSQL을 REST로 노출하는 PostgREST(Supabase의 핵심 구성요소) 위에서 동작. 매니지드 Supabase든
self-host PostgREST든 클라이언트 SDK가 동일하게 붙는다.

### 환경변수
| 키 | 용도 | 노출 |
|---|---|---|
| `SUPABASE_URL` | API 베이스. 내부=`http://<NAS_IP>:8080`, 외부=`https://<DOMAIN>` | 공개 가능 |
| `SUPABASE_KEY` (anon) | 공개 읽기용 (RLS로 보호) | 공개 가능 |
| `SUPABASE_SERVICE_KEY` | 적재/관리(쓰기) — RLS 우회 | **서버 전용** |

### 연동 패턴
- **클라이언트:** `supabase-py`(Python 적재), `@supabase/supabase-js`(웹 읽기).
- **읽기:** anon 키 + **RLS public read 정책**으로 공개 테이블만 SELECT 허용.
- **쓰기:** service key(또는 DB 직접)로 크롤러가 upsert. 클라이언트 번들에 절대 포함 금지.
- **호출 경로:** SDK는 `${SUPABASE_URL}/rest/v1/<table>` 로 요청. self-host는 역프록시가
  `/rest/v1/*` → PostgREST로 라우팅 ([06-infra-deployment.md](06-infra-deployment.md) 참고).
- **페이지네이션:** PostgREST 기본 `max-rows`(예 1000) 존재 → 대량 조회는 `Range` 헤더/`range()`로 분할.

### self-host 구축 절차

**(1) 시크릿 생성** (`.env`):
```bash
echo "POSTGRES_PASSWORD=$(openssl rand -base64 24)"
echo "AUTHENTICATOR_PASSWORD=$(openssl rand -base64 24)"
echo "JWT_SECRET=$(openssl rand -base64 48)"   # anon/service 키 서명 + PostgREST 검증 공용
```
- `JWT_SECRET`은 **anon/service 키 발급과 PostgREST 검증이 같은 값**을 써야 한다 (불일치 시 401).

**(2) DB 역할(roles)** — bootstrap SQL이 DB 최초 생성 시 자동 실행. 필요한 역할:
`anon`(공개 읽기) · `authenticated` · `service_role`(쓰기) · `authenticator`(PostgREST 접속용,
`AUTHENTICATOR_PASSWORD`와 동일하게 동기화). 확인:
```bash
docker exec -i <db_container> psql -U postgres -c "\du"   # 4개 역할 보여야 정상
```

**(3) 마이그레이션 적용** — 확장(postgis/pg_trgm) + 스키마 + **RLS public read 정책**.
self-host는 `storage.*`(Supabase 전용) 마이그레이션을 SKIP(스토리지는 MinIO). 적용 후:
```bash
docker exec -i <db_container> psql -U postgres -c "\dt"          # 테이블 확인
docker exec -i <db_container> psql -U postgres -c "NOTIFY pgrst,'reload schema'"
```

**(4) anon / service 키 발급** — `JWT_SECRET`으로 서명한 HS256 JWT.
payload: `{"role":"anon", "iss":"<issuer>"}` / `{"role":"service_role", ...}`.
```bash
JWT=$(grep '^JWT_SECRET=' .env | cut -d= -f2-)
node gen-anon-key.mjs "$JWT"      # ANON_KEY / SERVICE_KEY 출력
# node 없으면: docker run --rm -v "$PWD":/w -w /w node:20 node gen-anon-key.mjs "$JWT"
```
→ `ANON_KEY`는 web/crawler의 `SUPABASE_KEY`, `SERVICE_KEY`는 크롤러 적재용.

**(5) 검증**:
```bash
ANON="<ANON_KEY>"
curl -H "Authorization: Bearer $ANON" "http://<NAS_IP>:8080/rest/v1/<table>?select=*&limit=3"   # 내부
curl -H "Authorization: Bearer $ANON" "https://<DOMAIN>/rest/v1/<table>?select=*&limit=3"        # 외부
# 빈 [] = 정상(데이터 없음) · 401 = ANON↔JWT_SECRET 불일치 · 404 = 마이그레이션 미적용
```

코드 위치(예): `web/src/lib/supabase.ts`, `crawler/src/*/store.py`

---

## 2-2. 오브젝트 스토리지 — MinIO (S3 호환)

이미지/바이너리 저장. **S3 호환**이라 boto3로 붙고, 매니지드 Supabase Storage로도 폴백 가능하게
백엔드를 추상화한다.

### 환경변수
| 키 | 용도 | 노출 |
|---|---|---|
| `MINIO_ENDPOINT` | S3 API 주소 (`http://<NAS_IP>:9000`). 있으면 MinIO, 없으면 Supabase Storage 폴백 | 서버 전용 |
| `MINIO_ACCESS_KEY` / `MINIO_SECRET_KEY` | S3 자격증명 (`MINIO_ROOT_USER`/`PASSWORD`) | **서버 전용** |
| `STORAGE_PUBLIC_URL` | 공개 GET URL 베이스 → `{STORAGE_PUBLIC_URL}/{bucket}/{key}` (예 `https://<DOMAIN>/storage`) | 공개 가능 |

### 연동 패턴 (추상화)
- **백엔드 선택:** `MINIO_ENDPOINT` 있으면 `boto3`(S3 `put_object`), 없으면 Supabase Storage SDK.
  단일 `StorageBackend` 인터페이스(`upload`, `public_url`)로 호출 측은 차이를 모름.
- **업로드:** 크롤러가 원본 + 썸네일(예: Pillow 320×240)을 같은 버킷에 PUT. `upsert=true`로 멱등.
- **공개 읽기:** 버킷을 **public(anonymous download)**으로 설정. 웹은 `STORAGE_PUBLIC_URL` 기반으로
  `<base>/<bucket>/<path>` URL을 만들어 `<img src>`로 직접 로드.

### 버킷 생성 + 공개 정책
콘솔 `http://<NAS_IP>:9001`(로그인 ROOT_USER/PASSWORD)에서 버킷 생성 → Access Policy **public(download)**,
또는 `mc` CLI:
```bash
MINIO_ROOT_USER=$(grep '^MINIO_ROOT_USER=' .env | cut -d= -f2-)
MINIO_ROOT_PASSWORD=$(grep '^MINIO_ROOT_PASSWORD=' .env | cut -d= -f2-)
# 네트워크명은 'docker network ls | grep <COMPOSE>' 로 확인 (예: <COMPOSE>_internal)
docker run --rm --network <COMPOSE>_internal --entrypoint /bin/sh \
  -e MC_HOST_local="http://${MINIO_ROOT_USER}:${MINIO_ROOT_PASSWORD}@<minio_container>:9000" \
  minio/mc -c "
    mc mb -p local/<bucket-a> local/<bucket-b> &&
    mc anonymous set download local/<bucket-a> &&
    mc anonymous set download local/<bucket-b>
  "
```
> `minio/mc`는 entrypoint가 `mc`라 `sh -c`를 직접 못 받음 → `--entrypoint /bin/sh` 로 덮어쓰고 `-c`.
> 크롤러 쪽 의존성: MinIO(S3) 업로드에 `boto3` 필요 (`pip install boto3`).

### ⚠️ 자주 터지는 함정 (실제 겪음)
- **`STORAGE_PUBLIC_URL` 누락** → 웹이 매니지드 Supabase 경로(`/storage/v1/object/public/...`)로
  폴백 → self-host엔 그 경로 없음 → 이미지 전부 404. **반드시 설정 + 재배포.**
- **역프록시 `/storage` 라우트 미반영** → 공개 URL이 엣지에서 404. 프록시 설정 변경 후 **재기동** 필수
  (bind-mount 설정은 컨테이너 recreate 해야 반영).
- **버킷 public 미설정** → 이미지 **403** (`mc anonymous set download` 확인).
- **내부 포트 노출 오인** — S3 API 포트(9000)는 LAN 전용, 공개는 역프록시(443)의 `/storage/*`로만.
  외부에서 `:9000`/`:8080` 직결은 연결 거부가 정상.
- **mixed-content** — https 페이지에서 http 이미지 URL은 브라우저가 차단.

코드 위치(예): `crawler/src/storage_backend.py`, `web/src/lib/supabase.ts`

---

## 트러블슈팅 (DB/스토리지)
| 증상 | 확인 |
|---|---|
| PostgREST 401 | ANON_KEY가 JWT_SECRET으로 서명됐는지 (`.env` JWT_SECRET ↔ 키 발급 입력 일치) |
| PostgREST 404 (테이블) | 마이그레이션 적용됐는지 `\dt`. `NOTIFY pgrst,'reload schema'` |
| anon read 빈값/권한오류 | RLS public read 정책 + 역할 grant 적용됐는지 |
| 이미지 403 | MinIO 버킷 `anonymous set download` 됐는지 |
| 크롤러 write 실패 | `SUPABASE_SERVICE_KEY`(또는 postgres) 권한 + `MINIO_*` 키 |

## 백업
```bash
docker exec <db_container> pg_dump -U postgres -Fc postgres > backups/db_$(date +%F).dump   # DB
# 스토리지: MinIO 데이터 볼륨(volumes/storage/)을 백업 대상에 포함
```

## 재사용 체크리스트
- [ ] 시크릿 생성(`POSTGRES_PASSWORD`/`AUTHENTICATOR_PASSWORD`/`JWT_SECRET`).
- [ ] 역할 4종 + `authenticator` 비번 동기화.
- [ ] 마이그레이션 + RLS public read + schema reload.
- [ ] `JWT_SECRET`으로 anon/service 키 발급.
- [ ] 버킷 생성 + **public download** + `STORAGE_PUBLIC_URL`(웹 재배포).
- [ ] 내부/외부 `curl`로 rest·storage 200 확인.
- [ ] pg_dump + 스토리지 볼륨 백업 루틴.
