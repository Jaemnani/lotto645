# m04 — Mitra-v2 번호 확률 모델 (실험)

설계 배경: [`docs/07-mitra-v2-integration.md`](../docs/07-mitra-v2-integration.md)

| 파일 | 역할 |
|------|------|
| `m04_data.py` | CSV / Supabase → 회차 순 배열(`History`). 리허설 포함 |
| `m04_features.py` | (회차, 번호, 공세트) 한 행의 feature. 그 회차 **이전** 데이터만 사용 |
| `m04_model.py` | 백엔드: `mitra`(Mitra-v2) / `logreg`(기준선) / `m03` / `uniform` |
| `m04_backtest.py` | rolling-origin 백테스트 + paired bootstrap CI |
| `m04_predict.py` | 다음 회차 확률표(공세트 5 × 번호 45) → Supabase `model_predictions` |
| `test_m04.py` | 누수·m03 동등성·정규화 테스트 |

## 설치 (아이맥, 전용 venv)

```bash
python3 -m venv ~/workspace/venv_m04
~/workspace/venv_m04/bin/pip install -r model_m04_mitra/requirements.txt
~/workspace/venv_m04/bin/python -m pytest model_m04_mitra -q
```

첫 실행 때 `autogluon/mitra-classifier-2` 가중치를 Hugging Face 에서 받는다.

## 1. 먼저 백테스트 (도입 판정)

```bash
cd model_m04_mitra
# 기준선만 (수 초)
python m04_backtest.py --backend uniform m03 logreg
# Mitra-v2 ICL (CPU/MPS, 회차당 수 초~수십 초)
python m04_backtest.py --backend uniform m03 mitra
```

`mitra_vs_uniform` 의 logloss CI 가 0 을 넘을 때만 "패턴이 있다"고 본다.
설정을 바꿔가며 여러 번 돌린 뒤, 확정한 설정으로 `--final` 을 **한 번만** 돌려 최근 30회차로 최종 확인한다.

입력 형태를 바꿔보는 옵션:

| 옵션 | 의미 |
|------|------|
| `--context-rounds N` | 컨텍스트 회차 수 (×45행, 기본 110 ≈ 4,950행 / 한도 5,120) |
| `--context same_set` | 예측할 공세트와 같은 세트 회차만 컨텍스트로 |
| `--use-rehearsal` | 이전 회차 리허설 빈도 feature 추가 (같은 회차 리허설은 쓰지 않음) |
| `--n-estimators K` | Mitra 앙상블 수 |
| `--fine-tune` | Mitra fine-tune (GPU 권장) |

## 2. 주간 예측 (운영)

`cron.sh` 가 크롤링 직후 자동 실행한다 (`M04_PYTHON` venv 가 있을 때만).

```bash
python model_m04_mitra/m04_predict.py --source supabase            # 예측 + 업서트
python model_m04_mitra/m04_predict.py --source csv --backend logreg --dry-run   # 점검용
```

서버는 매시간 `model_predictions` 를 다시 읽는다. 해당 회차 m04 확률표가 없으면 m03 로 폴백한다.
