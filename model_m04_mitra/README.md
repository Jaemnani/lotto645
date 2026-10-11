# m04 — Mitra-v2 번호 확률 모델 + 인기 회피 (v2)

설계 배경: [`docs/07-mitra-v2-integration.md`](../docs/07-mitra-v2-integration.md)

m04 는 두 가지로 유리함을 노린다.

| 축 | 무엇을 | 현재 근거 (2026-10, 733~1230회) |
|---|---|---|
| **당첨 시 받는 돈** ✅ | 남들이 덜 고르는 조합 → 1등 공동당첨자 감소 | 5등 당첨자 수로 번호별 인기 추정 (학습 안 한 최근 100회 상관 0.89), 1등 효과 z≈5. 인기 하위 1% 조합 ≈ 공동당첨 0.82배 |
| **당첨 확률** ❌ (아직) | 공세트 편향(본게임 + 직전 리허설) + 기타 패턴을 Mitra-v2 가 판단 | 공세트 빈도 순위의 상위 6개 적중: 901~1000 −0.04, 1001~1100 −0.09, 1101~1200 +0.19, 1201~1230 +0.17 → 330회 합산 ≈ +0.03, **재현 안 됨**. α 는 dev 앞 구간에서 고르면 1000(≈균등). Mitra 는 맥북에서 판정 필요 |

그래서 **지금 권장하는 구매 방식은 "확률은 균등으로 두고 인기 회피로 고르기"** 다 (`m04_pick.py --backend uniform`).
Mitra 판정에서 균등을 확실히 이기면 그때 Mitra 확률을 함께 쓴다.

| 파일 | 역할 |
|------|------|
| `m04_data.py` | CSV / Supabase → 회차 순 배열(`History`). 리허설·등수별 당첨자 수·판매액 포함 |
| `m04_features.py` | (회차, 번호, 공세트) 한 행의 feature. 그 회차 **이전** 데이터만 사용. `--use-rehearsal` 로 v2 칼럼(r_post, sr_post, agree) |
| `m04_model.py` | 백엔드: `mitra`(Mitra-v2) / `logreg` / `m03` / `m03r`(리허설 합산) / `uniform` |
| `m04_tune.py` | 공세트 posterior 평활 강도 α 튜닝 (dev 구간) |
| `m04_backtest.py` | rolling-origin 백테스트 + paired bootstrap CI |
| `m04_popularity.py` | 번호별 인기도 + 1등 공동당첨 배수 모델 |
| `m04_predict.py` | 다음 회차 확률표(공세트 5 × 번호 45) → `prediction.json` (+ Supabase) |
| `m04_pick.py` | **최종 티켓 선택**: 확률 × 인기 회피 = 1등 기대가치 최대 N장 |
| `test_m04.py` | 누수·m03 동등성·v2 칼럼·인기도·티켓 제약 테스트 |

---

## 맥북에서 직접 돌리기 (Mitra 는 여기서만 가능)

### 0. 준비 (최초 1회)
```bash
cd ~/workspace/lotto645
git checkout master && git pull origin master

python3 -m venv ~/workspace/venv_m04
~/workspace/venv_m04/bin/pip install -r model_m04_mitra/requirements.txt
~/workspace/venv_m04/bin/python -m pytest model_m04_mitra -q        # 14 passed 확인
```
첫 Mitra 실행 때 `autogluon/mitra-classifier-2` 가중치를 Hugging Face 에서 자동으로 받는다 (수백 MB).

### 1. 데이터 최신화
```bash
python crawling/01_dh_caffe_crawling_with_auto_login.py     # 카페 크롤링 → data/history_from_cafe.csv
```

### 2. Mitra 판정 (최초 1회, 설정 바꿀 때마다)
```bash
cd model_m04_mitra
PY=~/workspace/venv_m04/bin/python

# 2-1. Mitra 가 실제로 로드·동작하는지 (5회차, 수 분)
$PY m04_backtest.py --backend uniform mitra --use-rehearsal --alpha 1000 --n-rounds 5

# 2-2. 본 판정 (100회차) — m03r 와 비교
$PY m04_backtest.py --backend uniform m03 m03r mitra --use-rehearsal --alpha 1000
```
- MPS(애플 GPU) 에러가 나면 `--device cpu` 를 붙인다.
- 결과 표에서 `mitra_vs_uniform` / `mitra_vs_m03` 의 ✅ 를 본다. **설정을 확정한 뒤에만** `--final` 을 한 번 돌린다(최근 30회 holdout).
- 출력 전체를 공유해 주면 판정을 같이 한다.

### 3. 매주 번호 뽑기
```bash
cd model_m04_mitra
PY=~/workspace/venv_m04/bin/python

# 권장 (현재): 확률은 균등, 인기 회피로 5장
$PY m04_pick.py --backend uniform --n 5

# Mitra 가 판정(2-2)에서 균등을 확실히 이긴 뒤에만: Mitra 확률 × 인기 회피
$PY m04_predict.py --source csv --backend mitra --use-rehearsal --alpha 1000 --dry-run
$PY m04_pick.py --n 5
```
`prediction.json` 이 지난 회차용이면 `m04_pick.py` 가 거부한다 (`--allow-stale` 로 무시 가능).

`m04_pick.py` 출력 열:
- **1등 기대가치**: 평균 조합 대비 (당첨확률 배수 ÷ 공동당첨 배수)
- **당첨확률(모델)**: 모델이 본 이 조합의 확률, 균등 대비 (번호 6개가 서로 다르다는 제약을 반영한 조건부 포아송 분포). `--backend uniform` 이면 1.00
- **공동당첨**: 1등이 됐을 때 같이 당첨될 사람 수, 평균 조합 대비
- 옵션: `--weight 0` 이면 인기 무시(확률만), `--ball-set 4` 처럼 공세트 고정, `--max-overlap` 티켓 간 겹침 수

### 4. (선택) α 다시 고르기 — 데이터가 많이 쌓였을 때
```bash
$PY m04_tune.py           # 백테스트 dev 구간 **앞** 100회에서 log loss 가 가장 낮은 α (dev 와 안 겹치게)
```

---

## 운영 (자동)

`cron.sh` 가 크롤링 직후 `m04_predict.py --source supabase` 를 실행한다 (v2 칼럼은 Mitra 판정 후에 켠다)
(`M04_PYTHON` venv 가 있을 때만). `model_predictions` 는 RLS 로 anon 쓰기를 막아 두었으므로 `.env` 에
**`SUPABASE_SERVICE_KEY`** 가 있어야 업서트된다. 서버는 매시간 다시 읽고, 해당 회차 확률표가 없으면 m03 로 폴백한다.
인기 회피(`m04_pick.py`)는 아직 서버 화면에는 붙어 있지 않다 — 직접 구매용.

## 백테스트 옵션

| 옵션 | 의미 |
|------|------|
| `--use-rehearsal` | 직전 회차까지의 리허설 기반 칼럼 (r_n, r_post, sr_post, agree) |
| `--alpha A` | r_post/sr_post 평활 강도 (m04_tune.py 로 결정, 현재 1000 ≈ 균등) |
| `--context-rounds N` | 컨텍스트 회차 수 (×45행, 기본 110 ≈ 4,950행 / 한도 5,120) |
| `--context same_set` | 예측할 공세트와 같은 세트 회차만 컨텍스트로 |
| `--n-estimators K` | Mitra 앙상블 수 |
| `--fine-tune` | Mitra fine-tune (GPU 권장) |
| `--final` | 최근 30회 holdout 평가 — 설정 확정 후 1회만 |
