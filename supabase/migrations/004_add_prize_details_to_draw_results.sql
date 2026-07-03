-- ============================================================
-- 004_add_prize_details_to_draw_results.sql
-- draw_results에 등수별 당첨 게임 수 / 등위별 총 당첨금 / 총 판매금액 컬럼 추가
-- 모두 is_winning=true 행에만 채워짐 (모의추첨/리허설 행은 NULL)
-- ============================================================

alter table draw_results add column if not exists winners_1     int;
alter table draw_results add column if not exists winners_2     int;
alter table draw_results add column if not exists winners_3     int;
alter table draw_results add column if not exists winners_4     int;
alter table draw_results add column if not exists winners_5     int;
alter table draw_results add column if not exists total_prize_1 bigint;
alter table draw_results add column if not exists total_prize_2 bigint;
alter table draw_results add column if not exists total_prize_3 bigint;
alter table draw_results add column if not exists total_prize_4 bigint;
alter table draw_results add column if not exists total_prize_5 bigint;
alter table draw_results add column if not exists total_sales   bigint;

comment on column draw_results.winners_1     is '1등 당첨 게임 수. 공홈 API에서 갱신, 모의추첨 행은 NULL.';
comment on column draw_results.total_prize_1 is '1등 등위별 총 당첨금 (원). 공홈 API에서 갱신.';
comment on column draw_results.total_sales   is '해당 회차 총 판매금액 (원). 공홈 API에서 갱신.';
