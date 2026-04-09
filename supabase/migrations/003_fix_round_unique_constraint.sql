-- ============================================================
-- 003_fix_round_unique_constraint.sql
-- round 단독 unique 제약(실제 이름 ix_draw_results_round) 제거 후
-- (round, is_winning) 복합 unique로 교체
-- ============================================================

-- 실제 존재하는 제약 제거
drop index if exists ix_draw_results_round;
alter table draw_results drop constraint if exists draw_results_round_key;
alter table draw_results drop constraint if exists ix_draw_results_round;

-- (round, is_winning) 복합 unique
create unique index if not exists uq_draw_results_round_winning
  on draw_results (round, is_winning);
