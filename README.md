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
├── cron.sh            # 주간 크롤링 실행 스크립트
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
| 스케줄러 | APScheduler (서버), crontab (로컬) |

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

### 매주 금요일 11:00 — 카페 크롤링 (로컬)

```
cron.sh 실행
  → crawling/01_dh_caffe_crawling_with_auto_login.py
    → 네이버 카페에서 추첨 데이터 크롤링
    → data/history_from_cafe.csv 저장
    → scripts/sync_cafe_history.py 자동 호출
      → Supabase draw_results 동기화
```

로컬 crontab 등록:
```
0 11 * * 5 /path/to/lotto645/cron.sh
```

### 매주 토요일 21:05 — 추첨 결과 처리 (서버)

```
APScheduler 자동 실행
  → 동행복권 공홈에서 최신 회차 결과 fetch
  → draw_results 저장 (is_winning=True)
  → user_extractions 등수 일괄 계산
  → weekly_announcements 생성/업데이트
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
