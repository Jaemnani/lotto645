# 01 · 데이터 수집원 (크롤/스크래핑)

외부 웹사이트·내부 API에서 데이터를 **주기적으로 수집**하는 연동. 공식 OpenAPI가 없거나 부족할 때
HTML/내부 JSON 엔드포인트를 직접 호출한다. 인증이 없는 대신, 서버가 봇을 차단(IP block)하거나
세션/헤더를 검사하는 경우가 많아 **위장·재시도·차단 감지**가 핵심이다.

## 패턴 (이 프로젝트 구현)

- **라이브러리:** `httpx.AsyncClient` (async). HTTP/2 off, 커스텀 헤더, 타임아웃.
- **세션/헤더 위장:** `User-Agent`, `Origin`, `Referer`를 실제 브라우저처럼 설정.
- **설정 객체:** `@dataclass` Config — `base_url`, `timeout_s`, 재시도 `backoff_base_s`/`backoff_max_s`,
  `save_dir`(raw 응답 보존), `dead_letter_path`(영구 실패 jsonl).
- **차단 처리:** 네트워크 재시도와 **별도 카운터 + 지수 backoff**. 차단이 확정되면 전용 예외를
  던지고(예: `IpBlocked`) 세션을 조기 종료 → 호출 측이 종료 코드로 감지해 후속 단계 skip.
- **멱등 적재:** 페이지 단위로 upsert → 중간에 끊겨도 다음 실행에서 재개.
- **로컬 throttle:** 요청 간 최소 간격(`_throttle`)으로 과부하·차단 위험 완화.

코드 위치(예):
- `crawler/src/<source>/client.py` — `ClientConfig` + `AsyncClient` 래퍼, `async def search/detail/list_*`
- `crawler/scripts/*ingest.py` — 페이지 순회 → store upsert, 차단 시 `sys.exit(75)`

## 등록된 수집원

| 수집원 | base_url | 방식 | 인증 |
|---|---|---|---|
| 소스 A | `https://www.courtauction.go.kr` | 내부 JSON 엔드포인트(POST), 세션 헤더 | 없음 |
| 소스 B | `https://www.bit.courts.go.jp` | HTML form POST → 파싱 | 없음 |

> 두 소스 모두 **API 키 불필요**. 대신 IP 차단·세션 검사가 있어 backoff/throttle/헤더 위장이 필수.

## 차단(IP block) 대응 흐름

1. 정상 요청 → 응답이 차단 신호(특정 상태/본문/리다이렉트)인지 검사.
2. 차단이면 지수 backoff로 N회 재시도 (`backoff_base_s × 2^k`, `backoff_max_s` 상한).
3. 재시도 소진 → 전용 예외 → 프로세스 종료 코드 `75`.
4. 오케스트레이터(run 스크립트)가 `75`를 보고 **같은 IP라 막힌 후속 step 전부 skip**, 다음 회차 재개.

## 운영 가이드

- **동시성 낮게.** detail 백필 등은 `--concurrency 2~5` 수준. 높이면 차단 위험 ↑.
- **raw 보존.** `save_dir`로 원본 응답을 남겨두면 스키마 변화·디버깅에 유리.
- **dead-letter.** 영구 실패 요청을 jsonl로 적재해 나중에 재처리.
- **스케줄.** cron으로 1일 1회 등. 전체 예산(`TIME_BUDGET`)을 두고 초과 시 남은 step 건너뜀.

## 재사용 체크리스트

- [ ] 대상 사이트의 실제 요청(엔드포인트·헤더·payload)을 DevTools/recon으로 캡처.
- [ ] `base_url`·헤더·throttle 간격 설정.
- [ ] 차단 신호 식별 → 전용 예외 + 종료코드 규약.
- [ ] 멱등 upsert 키 정의 (자연키).
- [ ] raw 저장 + dead-letter 경로.
- [ ] **법적/약관 검토** — 스크래핑 허용 범위, robots, 상업적 이용 제한 확인.
