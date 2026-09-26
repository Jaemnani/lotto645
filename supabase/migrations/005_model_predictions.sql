-- ============================================================
-- 005_model_predictions.sql
-- m04(Mitra-v2) 등 오프라인 배치 모델의 회차별 번호 확률표 저장
--   · 아이맥 배치(model_m04_mitra/m04_predict.py)가 service 키로 업서트
--   · 서버(web/number_gen.py)는 읽기만 — autogluon/torch 불필요
-- user_extractions.model : 어떤 모델로 추출했는지 (모델별 등수 비교용)
-- ============================================================

create table if not exists model_predictions (
  model                  text         not null,              -- 'm04'
  target_round           int          not null,              -- 예측 대상 회차
  ball_set               int          not null check (ball_set between 1 and 5),
  probs                  jsonb        not null,              -- 번호 1~45 확률 배열 (합 1, m03 posterior 와 같은 스케일)
  trained_through_round  int          not null,              -- 컨텍스트에 쓴 마지막 회차
  backend                text,                               -- 'mitra' | 'logreg' ...
  config                 jsonb,
  created_at             timestamptz  default now(),
  primary key (model, target_round, ball_set)
);

comment on table  model_predictions       is '오프라인 배치 모델의 회차별 번호 확률표 (공세트 1~5 × 번호 45).';
comment on column model_predictions.probs is '길이 45 배열. index 0 = 1번. 합 1.';

-- 이 표가 곧 사용자에게 나가는 번호를 결정하므로 anon 키로는 읽기만 허용한다.
-- 쓰기 정책은 두지 않음 → 업서트는 RLS 를 우회하는 service_role 키(아이맥 배치)만 가능.
alter table model_predictions enable row level security;
drop policy if exists "public read model_predictions" on model_predictions;
create policy "public read model_predictions" on model_predictions
  for select to anon, authenticated using (true);

alter table user_extractions add column if not exists model text not null default 'm03';
comment on column user_extractions.model is '추출에 실제 사용된 모델 (m03 | m04). m04 예측이 없으면 m03 로 폴백된 값.';
