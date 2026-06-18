# 05 · 운영 알림 (Discord Webhook)

배치/크롤/배포 같은 **자동 작업의 시작·결과·에러**를 채팅 채널로 받는 연동. 가장 가벼운 방식은
**Incoming Webhook**(Discord/Slack 공통 개념) — 봇 토큰·OAuth 없이 URL 하나로 POST.

## Discord Webhook

| 항목 | 값 |
|---|---|
| 엔드포인트 | `POST https://discord.com/api/webhooks/<id>/<token>` |
| 인증 | URL 자체가 시크릿 (별도 키 없음) |
| 요청 | `Content-Type: application/json`, body `{"content": "..."}` (마크다운 지원, 2000자 제한) |
| env | `DISCORD_WEBHOOK_URL` (시크릿; gitignore) |

### 발급
채널 → **설정 → 연동(Integrations) → 웹후크 → 새 웹후크 → URL 복사**.

### 연동 패턴 (이 프로젝트)
shell 오케스트레이터(run 스크립트)에 webhook을 거는 방식. 공용 `notify.sh`를 두 스크립트가 source.

- `discord_send <content>` — `DISCORD_WEBHOOK_URL` 있으면 `curl` POST, **없으면 조용히 no-op**.
  - JSON 인코딩은 런타임에 있는 인터프리터(여기선 venv `python -c json.dumps`)로 → 따옴표/줄바꿈/유니코드
    안전, `jq` 의존 없음.
  - `curl -sS --max-time 10 ... || true` + 길이 truncate → 웹훅 실패·과길이가 본 작업을 안 죽임.
- `discord_digest <log> <rc> <elapsed> <label>` — 그 런의 **로그 파일을 grep**해 요약 1건 구성:
  상태(✅/⚠️경고/⚠️차단/❌실패) + 소요시간 + 단계별 결과(`[done] ...` 라인) + 경고/스킵.
- **EXIT trap**으로 종료 경로를 단일화 → 정상·조기종료(exit 0)·크래시·예산소진 **모든 경우**에 요약 1건.
  시작 시 1건 + 종료 시 1건 = 런당 2건(에러면 종료 메시지가 경고 스타일로 escalate).

### 설계 원칙
- **미설정 = 무동작.** env 없으면 알림 off, 기존 동작 불변.
- **로그를 데이터 소스로.** 이미 찍는 요약 로그를 파싱 → 별도 수집 로직 불필요. bash 레벨 이벤트
  (차단·예산소진·크래시)까지 한 번에 잡힘.
- **cron 무수정.** run 스크립트가 `.env`를 자체 로드하므로 키만 추가하면 적용.

코드 위치(예): `crawler/lib/notify.sh`, `crawler/run_daily.sh`, `crawler/run_jp_daily.sh`.

### 테스트
```bash
DISCORD_WEBHOOK_URL='<url>' bash crawler/lib/notify.sh test
```

## Slack로 바꾸려면
거의 동일. Incoming Webhook URL 발급 후 body만 `{"text": "..."}`로. 나머지 패턴(no-op·digest·trap)
그대로 재사용.

## 재사용 체크리스트
- [ ] 채널에 Incoming Webhook 발급 → URL을 시크릿 env로.
- [ ] `send`(no-op 가드) + `digest`(로그 파싱) 함수.
- [ ] 종료 경로 단일화(EXIT trap)로 성공/실패/크래시 모두 1건.
- [ ] 실패 격리(타임아웃·`|| true`), 길이 truncate.
- [ ] 레이트리밋 고려(Discord 웹훅 ~30/min) — 런당 메시지 수 최소화.
