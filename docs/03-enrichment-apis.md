# 03 · 보강 API (Enrichment)

수집한 데이터를 외부 API로 **보강**하는 연동. 모두 **API 키 필요**, 모두 **서버 전용**(키 노출 금지),
무료 할당량/요금이 있으니 호출량 관리가 중요하다. 세 가지 유형: 지오코딩 / 정부 OpenAPI / LLM.

공통 원칙:
- 키는 환경변수로만. 브라우저로 내려보내지 않음 → 호출은 **서버 라우트/백엔드**에서.
- 타임아웃 + 실패 격리(`try/except`, 빈 결과 반환). 보강 실패가 본 파이프라인을 죽이지 않게.
- 결과 캐싱/재호출 방지(이미 채워진 row는 skip)로 할당량 절약.

---

## 3-1. 지오코딩 — Kakao Local API

좌표 ↔ 주소 변환. 예: 좌표만 있고 주소가 빈 레코드를 **역지오코딩**으로 채움.

| 항목 | 값 |
|---|---|
| 엔드포인트 | `GET https://dapi.kakao.com/v2/local/geo/coord2address.json?x={lng}&y={lat}&input_coord=WGS84` |
| 인증 | 헤더 `Authorization: KakaoAK ${KAKAO_REST_API_KEY}` |
| env | `KAKAO_REST_API_KEY` (서버 전용) |
| 응답 | `documents[].road_address.address_name`(도로명) / `.address.address_name`(지번) |

패턴: 대상 row를 페이지 분할로 모아 `asyncio` + `Semaphore(동시 5~8)`로 병렬 호출, 결과를 비파괴
update(값 있을 때만). 코드 위치(예): `crawler/scripts/ingest.py` `reverse-geocode`.

> 무료 할당량 넉넉(로컬 API 일 10만 건 수준). 대량 백필도 분 단위로 끝남.

---

## 3-2. 정부 공공데이터 OpenAPI (data.go.kr)

공공 OpenAPI. 유형별로 **엔드포인트가 갈린다**(예: 실거래가가 아파트/연립/단독/오피스텔/토지/상업/
공장 등 9종). XML 응답이 흔함.

| 항목 | 값 |
|---|---|
| 엔드포인트 | `https://apis.data.go.kr/<svc-id>/<op>/<op>` (유형별 N개) |
| 인증 | 쿼리파라미터 `serviceKey=<KEY>` |
| env | `DATA_GO_KR_API_KEY` (서버 전용) |
| 파라미터(예) | `LAWD_CD`(지역코드), `DEAL_YMD`(YYYYMM), `numOfRows` |

### ⚠️ 함정
- **ServiceKey는 이미 URL-encoded 형태**로 발급됨 → URL에 **raw concat**해야 함.
  `encodeURIComponent`로 재인코딩하면 `%2F`가 `%252F`가 돼 깨짐. (다른 파라미터만 인코딩.)
- 응답이 XML → 파서 필요. 유형↔엔드포인트 매핑 테이블을 코드에 둔다.
- 서버 라우트로 **프록시**해서 키를 숨기고 브라우저엔 결과만 전달.

코드 위치(예): `web/src/app/api/molit-deals/route.ts` (Next.js 서버 라우트).

---

## 3-3. LLM 분류 — Google Gemini

룰로 분류 안 되는 항목을 LLM으로 보강 분류. 저비용 모델 + 호출 최소화가 핵심.

| 항목 | 값 |
|---|---|
| 라이브러리 | `google-genai` (`from google import genai`) |
| 모델 | `gemini-2.5-flash-lite` (저비용 티어) |
| 인증 | `GEMINI_API_KEY` (서버 전용) |
| 비용(참고) | 입력 ≈ $0.10 / 1M tok, 출력 ≈ $0.40 / 1M tok → 건당 ~$0.00004 |

패턴:
- **룰 엔진 먼저, LLM은 폴백.** 규칙으로 분류되는 건 LLM 호출 안 함 → 비용 최소화.
- 구조화 출력(JSON) 받아 파싱. 실패 시 빈 결과로 degrade.
- 누적 토큰으로 `cost_estimate()` 산출, 실행 totals에 기록.

> 최신 모델/요금/가용성은 가정하지 말고 공급자 문서로 확인할 것. 모델 ID는 설정값(`GeminiConfig.model`)으로
> 빼두면 교체 쉬움.

코드 위치(예): `crawler/src/llm/gemini.py`.

---

## 재사용 체크리스트
- [ ] 키 발급 + 무료 할당량/요금 확인.
- [ ] 서버 전용 env, 브라우저 비노출(서버 라우트 프록시).
- [ ] 타임아웃 + 실패 시 빈 결과 degrade.
- [ ] 이미 보강된 레코드 skip(재호출 방지).
- [ ] 호출량/비용 로깅.
