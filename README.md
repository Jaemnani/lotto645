# lotto645

LSTM 모델 기반 로또 6/45 번호 추출 서비스

---

## 구성

```
lotto645/
├── crawling/          # 네이버 카페 크롤링 (Selenium)
├── data/              # 추첨 히스토리 CSV
├── docs/              # 서버 세팅 문서
├── frontend/          # React 프론트엔드 (Vite + Tailwind)
├── model_m02_claude/  # LSTM 모델 및 학습 코드
├── scripts/           # Supabase 동기화 스크립트
├── supabase/          # DB 마이그레이션 SQL
├── web/               # FastAPI 백엔드
├── cron.sh            # 아이맥 매일 크롤링 스크립트 (LaunchAgent 에서 호출)
└── run_web.py         # 서버 실행 엔트리포인트
```

---

## 기술 스택

| 구분 | 기술 |
|------|------|
| 백엔드 | FastAPI, Python 3.12 |
| 프론트엔드 | React, TypeScript, Vite, Tailwind CSS |
| DB | Supabase (PostgreSQL) |
| 모델 | PyTorch LSTM |
| 서버 | Oracle Cloud Free Tier (Ubuntu 24.04) |
| 스케줄러 | APScheduler (서버, 매시간 재학습 + 토요일 통계), launchd LaunchAgent (아이맥, 매일 크롤링 + 금요일 구매) |

---

## DB 테이블

### draw_results
추첨 결과 저장. 회차당 2행 (모의추첨 / 실제 당첨번호).

| 컬럼 | 설명 |
|------|------|
| round | 회차 |
| draw_date | 추첨일 |
| is_winning | false=모의추첨, true=실제 당첨번호 |
| ball_set | 볼셋 번호 (카페 크롤링 전 NULL) |
| n1~n6 | 번호 |
| bonus | 보너스 번호 |
| prize_1~5 | 등수별 당첨금 (공홈 크롤링 후 갱신) |

### user_extractions
번호 추출 히스토리.

| 컬럼 | 설명 |
|------|------|
| session_id | 익명 세션 ID |
| user_name | 계정 이름 (미계정 시 NULL) |
| phone_last4 | 전화번호 뒷 4자리 (미계정 시 NULL) |
| target_round | 응모 회차 |
| ball_set | 볼셋 |
| strategy | 추출 방식 (1~4) |
| numbers | 추출 번호 배열 |
| rank | 등수 (추첨 후 업데이트) |

### weekly_announcements
회차별 통계 공지. 토요일 추첨 후 자동 생성.

---

## 자동화 흐름

### 스케줄 요약

| 시점 | 동작 | 실행 주체 |
|------|------|-----------|
| 매일 11:00 KST | 네이버 카페 크롤링 → Supabase 동기화 | 아이맥 LaunchAgent (`com.lotto645.daily-crawl`) |
| 매시간 정각 | DB 신규 회차 감지 시 m03 재학습 + 메모리 리로드 | 서버 APScheduler (`hourly_retrain_check`) |
| 매주 토요일 21:05 KST | 당첨번호 fetch + 사용자 등수 계산 + 주간 공지 생성 | 서버 APScheduler (`saturday_job`) |
| 매주 금요일 10:00 KST | 파이프라인 (크롤링→학습→예측→구매) | 아이맥 LaunchAgent (`com.lotto645.friday-buy`) |
| FastAPI 기동 시 | 30초 후 모델 자동 로드 (없거나 오래됐으면 재학습) | 서버 APScheduler |

### 아이맥 LaunchAgent 등록

```bash
bash ~/workspace/lotto645/setup_launchagent.sh
launchctl list | grep lotto645   # 등록 확인
```

### 서버 재학습 흐름

```
아이맥: cron.sh → 카페 크롤링 → scripts/sync_cafe_history.py → Supabase draw_results
    ↓
서버 매시간 정각: Supabase.max(round) > 현재 모델.round_range[1] ?
    └─ YES → retrain() 호출
              └─ Supabase fetch → BayesianFrequencyModel.fit() → .npz 원자적 교체
              └─ number_gen.reload_model() → 메모리 싱글턴 교체
              └─ 사용자가 즉시 최신 모델 기반 번호 수령
```

### 관리자 API

```bash
# 현재 모델 정보 (학습 회차, 로드 시각, 마지막 재학습 결과)
curl http://YOUR_SERVER_IP/api/model/info

# 수동 재학습 트리거
curl -X POST http://YOUR_SERVER_IP/api/admin/retrain \
  -H "X-Admin-Key: $ADMIN_KEY"
```

---

## API

| 메서드 | 경로 | 설명 |
|--------|------|------|
| POST | /api/extract | 번호 추출 (save=true 시 DB 저장) |
| GET | /api/draw/latest | 최신 추첨 결과 |
| GET | /api/announcement/latest | 최신 주간 공지 |
| GET | /api/announcements | 공지 목록 |
| POST | /api/admin/fetch-and-calc | 추첨 결과 수동 fetch + 통계 계산 (관리자) |

관리자 API는 `X-Admin-Key` 헤더 필요.

---

## 번호 추출 전략

| 전략 | 설명 |
|------|------|
| 1 | 확률 상위 — 모델이 가장 높게 예측한 조합 |
| 2 | 가중 랜덤 — 확률 기반 다양성 확보 |
| 3 | 구간 균형 — 낮은/중간/높은 번호 골고루 |
| 4 | Cold 번호 포함 — 오래 안 나온 번호 포함 |

---

## 환경변수

`.env` 파일을 프로젝트 루트에 생성:

```
SUPABASE_URL=https://YOUR_PROJECT_ID.supabase.co
SUPABASE_KEY=YOUR_SUPABASE_ANON_KEY
ADMIN_KEY=YOUR_ADMIN_KEY
HOST=0.0.0.0
PORT=8000
NAVER_ID=YOUR_NAVER_ID
NAVER_PW=YOUR_NAVER_PW
GEMINI_API_KEY=YOUR_GEMINI_API_KEY
```

---

## 서버 배포

서버 세팅 상세 내용은 [docs/server_setup.md](docs/server_setup.md) 참고.

### 코드 배포

```bash
# 로컬에서
git push origin master

# 서버에서
cd ~/workspace/lotto645
git pull origin master
sudo systemctl restart lotto645
```

### 프론트엔드 빌드

```bash
cd frontend
npm run build   # web/static/ 에 빌드됨
```

### 서버 로그 확인

```bash
journalctl -u lotto645 -f
```

---

## DB 마이그레이션

```bash
supabase link
supabase db push
```

새 마이그레이션 추가:
```bash
supabase migration new 마이그레이션_이름
# supabase/migrations/에 생성된 파일에 SQL 작성 후
supabase db push
```

---

## 로컬 개발 실행

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 백엔드
python run_web.py

# 프론트엔드 (별도 터미널)
cd frontend
npm install
npm run dev
```
