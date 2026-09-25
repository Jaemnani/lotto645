# Mitra-v2 연동 방안 (m04)

기존 번호 추출(공 세트 × 전략 1~4)의 **확률 공급원**으로 Amazon Mitra-v2를 붙이는 설계안.
m03(베이지안 빈도)는 그대로 두고, 같은 인터페이스(`prob: np.ndarray(45,)`)를 내는 m04를 추가한다.

---

## 0. 먼저 짚고 갈 것

- 로또 6/45는 **각 회차가 독립인 균등 추첨**이 되도록 설계되어 있다. m03의 "공 질량 편향" 가설이든 Mitra든,
  기대값은 "균등분포(p = 6/45 ≈ 0.1333)와 통계적으로 구분이 안 됨"이다.
- 그래서 이 설계의 핵심은 모델 자체보다 **§5 백테스트 게이트**다. 균등분포/m03 대비 개선이 신뢰구간 밖으로 나오지 않으면
  m04는 "실험 모델"로만 노출하고 m03를 대체하지 않는다.
- 실패해도 비용이 작도록 **오프라인 배치 → 확률 테이블만 서버로** 가는 구조로 설계한다 (§4).

---

## 1. Mitra-v2 요약

| 항목 | 내용 |
|------|------|
| 종류 | **테이블형(tabular) 파운데이션 모델** — 시계열 모델 아님 |
| 방식 | In-context learning (학습 데이터를 컨텍스트로 넣고 query 행을 예측). 선택적으로 fine-tune |
| 구조 | 12-layer 2D Transformer(행·열 attention), 약 76.7M 파라미터 |
| v1 → v2 | 컨텍스트 10배, feature 수 3배, 옵티마이저 개선 |
| 체크포인트 | `autogluon/mitra-classifier-2`, `autogluon/mitra-regressor-2` (HF) |
| 라이선스 | Apache-2.0, 상업적 사용 가능 |
| 실행 | `autogluon.tabular[mitra]` (현재 1.6.3). Python ≥3.10, torch ≥2.10 (macOS는 <2.11) |
| 하드웨어 | GPU 강력 권장 — CPU에서 ICL 추론은 12~63배 느림, CPU fine-tune은 비현실적 |

AutoGluon 1.6.3 소스 기준 확인 사항:

- 기본 체크포인트는 여전히 **v1**(`autogluon/mitra-classifier`). v2를 쓰려면 `hf_model`(또는 `hf_cls_model`)에 repo id를 **명시해야** 한다.
- TabularPredictor 경유 시 기본 제한은 `max_rows=10000`, `max_features=500`, `max_classes=10`(v1 기준값).
  → 45-class 다중분류는 불가하고, 행 수도 초과하므로 §2처럼 문제를 정의하고 **sklearn 인터페이스를 직접** 쓴다.

```python
from autogluon.tabular.models.mitra.sklearn_interface import MitraClassifier

clf = MitraClassifier(
    hf_model="autogluon/mitra-classifier-2",
    fine_tune=False,          # ICL only (CPU/MPS 현실적인 설정)
    n_estimators=1,
    device="mps",             # 아이맥 Apple Silicon. CUDA면 "cuda"
    metric="log_loss",
    seed=0,
)
clf.fit(X_ctx, y_ctx)
p = clf.predict_proba(X_query)[:, 1]
```

> ⚠️ 미검증: 이 컨테이너에서는 HF 다운로드가 막혀 있어 **AutoGluon 1.6.3의 `Tab2D`가 v2 체크포인트를 그대로 로드하는지**는 확인하지 못했다.
> PR1의 첫 작업이 이 확인이다 (안 되면 v2를 지원하는 AutoGluon 버전으로 올림).

---

## 2. 문제 정의 — "(회차, 번호) 이진분류"

45-class 분류는 `max_classes` 제한에 걸리고, 한 회차에 6개가 동시에 나오는 구조와도 맞지 않는다.
대신 **한 행 = (회차 t, 번호 i)** 로 펼친다.

- 타깃 `y = 1` ⇔ 번호 i가 회차 t 당첨번호 6개에 포함 (보너스는 옵션, 기본 제외 — m03와 동일)
- 기저율 6/45 = 13.3% (불균형 크지 않음)
- 규모: 공 세트가 있는 실제 추첨 ≈ 396회차(835~1230) × 45 = **약 17.8k행**
  (733~834회차는 ball_set이 없어 `ball_set=0`(unknown)으로 넣거나 제외 — 백테스트로 결정)

예측 시(다음 회차 T+1)에는 공 세트를 알 수 없으므로, 현재 UX와 똑같이 **공 세트 1~5를 가정한 query 225행**(5 × 45)을 만든다.
출력 `p[bs, i]`는 세트별로 합이 1이 되게 정규화해서 m03 `posterior()`와 같은 스케일로 맞춘다.

```
prob_bs = p[bs] / p[bs].sum()      # (45,) — generate_numbers()에 그대로 투입
```

---

## 3. Feature 설계 (point-in-time)

**모든 feature는 회차 t 이전(< t) 데이터만으로 계산**한다. 이게 깨지면 백테스트가 전부 무의미해진다.

| 그룹 | feature | 비고 |
|------|---------|------|
| 식별 | `number`(1~45), `ball_set`(0~5) | categorical 로 처리 |
| 번호 속성 | 홀짝, 구간(1–15/16–30/31–45), 끝자리 | 정적 feature |
| 전체 빈도 | 누적 출현율, 최근 10/30/100회 출현 횟수 | |
| 세트별 빈도 | **m03 사후확률(해당 세트, t 시점 expanding)**, 세트별 최근 10/30회 출현 | m03를 feature로 흡수 → m04 ⊇ m03 |
| 간격 | 마지막 출현 이후 회차 수(gap), 같은 세트 기준 gap | 전략 4 "Cold"의 정량화 |
| 직전 회차 | 직전 회차 당첨 여부, 직전 회차 보너스 여부 | |

제외(기본값): **모의추첨(리허설) 번호**. 같은 회차 리허설은 구매 마감(토 20:00) 전에 알 수 없으므로 누수다.
직전 회차 이하의 리허설만 쓰는 버전은 실험 플래그(`--use-rehearsal`)로 백테스트에서만 비교한다.

---

## 4. 시스템 연동 구조

### 4.1 실행 위치: 서버가 아니라 아이맥 배치

| 선택지 | 판단 |
|--------|------|
| 오라클 서버(Free Tier, CPU)에서 요청 시 추론 | ✗ — torch+autogluon 의존성이 무겁고, CPU ICL은 느림 |
| 오라클 서버에서 매시간 재학습 루프에 편입 | ✗ — 같은 이유. 새 데이터는 주 1회뿐이라 의미도 없음 |
| **아이맥에서 주 1회 배치 추론 → 확률 테이블만 DB에 업서트** | ✅ 권장 — 이미 매일 크롤링이 돌고 있고 MPS 사용 가능 |
| GitHub Actions (CPU, `fine_tune=False`) | 차선 — 아이맥이 꺼져 있을 때의 백업 |

서버는 **autogluon/torch를 설치하지 않는다.** m04의 서버 측 산출물은 `(5, 45)` 확률 테이블 하나뿐이다.

### 4.2 데이터 흐름

```
토 21:05  서버 saturday_job → 당첨번호 fetch
일 11:00  아이맥 cron.sh → 카페 크롤링(ball_set 확보) → Supabase draw_results
          └─ (추가) model_m04_mitra/m04_predict.py
               ├─ draw_results fetch → feature 생성 (< T+1)
               ├─ MitraClassifier(v2) ICL → query 225행 predict_proba
               └─ Supabase model_predictions 업서트 (model='m04', target_round=T+1)
매시간     서버 hourly_retrain_check → m03 재학습 + m04 테이블 리로드
요청 시    /api/extract { model: 'm04' } → 테이블 조회 → 기존 전략 1~4 그대로 적용
```

m04 예측이 해당 회차에 아직 없으면 **m03로 폴백**하고 응답에 `model_used`로 알린다.

### 4.3 변경 포인트

| 위치 | 변경 |
|------|------|
| `model_m04_mitra/` (신규) | `m04_features.py`(point-in-time feature), `m04_predict.py`(주간 배치), `m04_backtest.py`(§5), `requirements.txt`(별도 venv) |
| `supabase/migrations/005_model_predictions.sql` | `model_predictions(model, target_round, ball_set, probs jsonb, trained_through_round, created_at)`, PK `(model, target_round, ball_set)` |
| 〃 | `user_extractions.model text not null default 'm03'` — 모델별 등수 비교용 |
| `deploy/synology/db/` | 위 테이블에 RLS 정책 추가 (anon read, service write) |
| `web/number_gen.py` | `_get_prob(ball_set)` → `_get_prob(ball_set, model)` 로 분리. 전략 1~4 로직은 **변경 없음** |
| `web/scheduler.py` | hourly 체크에서 m04 최신 `target_round` 리로드 |
| `web/app.py` | `ExtractRequest.model: Literal['m03','m04'] = 'm03'`, 응답에 `model_used`; `/api/model/info`에 m04 메타 |
| `frontend` | ExtractPanel에 모델 선택(m03 / m04 "실험") 추가, 결과 뱃지에 모델 표시 |
| 주간 공지 | 모델별 평균 일치 개수/등수 분포 추가 |
| `cron.sh` / `setup_launchagent.sh` | 크롤링 성공 후 m04 predict 호출 (실패해도 크롤링은 성공 처리 + Discord 알림) |

---

## 5. 검증 — 도입 여부를 가르는 게이트

### 5.1 Rolling-origin 백테스트

최근 N=100회차 각각에 대해: `< t` 데이터로 컨텍스트 구성 → 회차 t의 **실제 ball_set** 기준 45행 예측 → 채점.
(ICL only라 회차당 "학습"은 컨텍스트 교체일 뿐. 아이맥에서 하룻밤 분량, GPU면 수십 분)

| 지표 | 균등분포 기대값 | 의미 |
|------|----------------|------|
| 번호별 log loss | −[(6/45)ln(6/45) + (39/45)ln(39/45)] ≈ 0.3927 | 확률 보정까지 포함한 주 지표 |
| Brier score | (6/45)(39/45) ≈ 0.1156 | 보조 |
| top-15 풀 적중 수 | 15 × 6/45 = **2.0** | 현재 서비스의 번호 풀 품질 |
| 전략 1 티켓 일치 수 | 6 × 6/45 = **0.8** | 사용자가 체감하는 값 |

비교 대상: **균등분포 / m03 / m04(Mitra-v2)**, 추가로 m04 ablation(m03 feature 제거, 리허설 포함 버전).

### 5.2 판정 규칙

- 회차별 (m04 − 기준) 차이에 대해 paired bootstrap 95% CI(또는 Wilcoxon)를 낸다.
- **CI가 0을 넘어서 개선**일 때만 m04를 기본 모델 후보로 올린다.
- 그 외(대부분 이 경우일 것)에는: m04는 "실험" 라벨로 선택 가능하게만 두고, 운영 데이터(`user_extractions.model`)로 계속 추적.
- 다중 비교 주의: feature/하이퍼파라미터를 여러 번 바꿔가며 백테스트하면 우연히 좋은 설정이 나온다.
  **최종 비교용으로 최근 30회차는 끝까지 떼어두고**, 설정 확정 후 한 번만 평가한다.

---

## 6. 단계별 작업 (PR 단위)

1. **PR1 — 오프라인 검증** (서버 변경 없음)
   - v2 체크포인트 로드 확인 (`hf_model="autogluon/mitra-classifier-2"`), 안 되면 AutoGluon 버전 조정
   - `m04_features.py` + 누수 방지 단위 테스트(회차 t feature가 t 이후 데이터에 불변인지)
   - `m04_backtest.py` → 결과 리포트 → §5.2 판정
2. **PR2 — 배치 파이프라인**: migration 005, `m04_predict.py`, 아이맥 LaunchAgent 연동, Discord 알림
3. **PR3 — 서비스 노출**: `number_gen`/API/프론트 모델 선택, `user_extractions.model`, 주간 공지 모델별 통계

PR1 결과가 나쁘면 PR2/3은 "실험 탭" 수준으로 축소하거나 보류한다.

---

## 7. 리스크

| 리스크 | 대응 |
|--------|------|
| AutoGluon 1.6.3이 v2 체크포인트 미지원 | PR1 첫 단계에서 확인, 버전 올림 |
| 의존성 충돌 (루트 `requirement.txt`는 torch 무버전·numpy 1.26.4 고정, AG는 torch ≥2.10) | m04 전용 venv 분리. 서버 requirements에는 추가하지 않음 |
| macOS torch <2.11 제약, MPS 경로 불안정 | 실패 시 `device="cpu"` + `fine_tune=False`로 폴백 (주 1회 225행이라 CPU도 감당 가능) |
| feature 누수 | point-in-time 단위 테스트 필수, 리허설 기본 제외 |
| 아이맥 미가동으로 m04 예측 누락 | 서버 m03 폴백 + GitHub Actions 백업 워크플로 |
| 사용자에게 "AI 모델이라 더 잘 맞는다"는 인상 | UI에 "실험" 라벨, 공지에 모델별 실제 성적을 그대로 노출 |

---

참고:
- [Mitra: Mixed synthetic priors for enhancing tabular foundation models (Amazon Science)](https://www.amazon.science/blog/mitra-mixed-synthetic-priors-for-enhancing-tabular-foundation-models)
- [autogluon/mitra-classifier-2 (Hugging Face)](https://huggingface.co/autogluon/mitra-classifier-2)
- [AutoGluon Tabular — Foundational Models](https://auto.gluon.ai/stable/tutorials/tabular/tabular-foundational-models.html)
