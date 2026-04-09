-- ============================================================
-- 002_add_rehearsal_to_draw_results.sql
-- draw_results에 모의/실제 구분 컬럼 추가
-- round 단독 UNIQUE → (round, is_winning) 복합 UNIQUE로 변경
-- ============================================================

alter table draw_results
  add column if not exists is_winning boolean not null default false;

comment on column draw_results.is_winning is
  'false = 모의추첨(리허설), true = 실제 당첨번호. CSV 회차당 첫 번째 행 = false, 두 번째 행 = true.';

-- 기존 round 단독 unique 제약 제거
alter table draw_results
  drop constraint if exists draw_results_round_key;

-- (round, is_winning) 복합 unique
create unique index if not exists uq_draw_results_round_winning
  on draw_results (round, is_winning);
