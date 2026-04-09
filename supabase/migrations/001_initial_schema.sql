-- ============================================================
-- 001_initial_schema.sql
-- Lotto645 초기 테이블 생성 (idempotent)
-- ============================================================

-- ──────────────────────────────────────────────────────────
-- 1. draw_results
-- ──────────────────────────────────────────────────────────
create table if not exists draw_results (
  id          bigint generated always as identity primary key,
  round       int          not null unique,
  draw_date   date         not null,
  ball_set    int,
  n1          int          not null,
  n2          int          not null,
  n3          int          not null,
  n4          int          not null,
  n5          int          not null,
  n6          int          not null,
  bonus       int          not null,
  prize_1     bigint,
  prize_2     bigint,
  prize_3     bigint,
  prize_4     bigint,
  prize_5     bigint,
  fetched_at  timestamptz  default now()
);

-- 테이블이 이미 존재했을 경우 누락 컬럼 추가
alter table draw_results add column if not exists ball_set   int;
alter table draw_results add column if not exists prize_1    bigint;
alter table draw_results add column if not exists prize_2    bigint;
alter table draw_results add column if not exists prize_3    bigint;
alter table draw_results add column if not exists prize_4    bigint;
alter table draw_results add column if not exists prize_5    bigint;
alter table draw_results add column if not exists fetched_at timestamptz default now();

comment on column draw_results.ball_set is '추첨 기계 볼셋 번호. 카페 크롤링 전까지 NULL.';
comment on column draw_results.prize_1  is '1등 당첨금 (원). 공홈 API에서 갱신.';


-- ──────────────────────────────────────────────────────────
-- 2. user_extractions
-- ──────────────────────────────────────────────────────────
create table if not exists user_extractions (
  id            bigint generated always as identity primary key,
  session_id    text         not null,
  user_name     text,
  phone_last4   char(4),
  target_round  int          not null,
  ball_set      int          not null,
  strategy      int          not null,
  numbers       jsonb        not null,
  extracted_at  timestamptz  default now(),
  rank          int,
  match_count   int,
  bonus_match   boolean
);

alter table user_extractions add column if not exists user_name    text;
alter table user_extractions add column if not exists phone_last4  char(4);

comment on column user_extractions.session_id  is '익명 세션 ID (항상 존재)';
comment on column user_extractions.user_name   is '계정 이름 (미계정 시 NULL)';
comment on column user_extractions.phone_last4 is '전화번호 뒷 4자리 (미계정 시 NULL)';
comment on column user_extractions.strategy    is '추출 방식: 0=랜덤, 1=통계기반, …';
comment on column user_extractions.rank        is '1~5 당첨, 6 낙첨. 추첨 전 NULL.';


-- ──────────────────────────────────────────────────────────
-- 3. weekly_announcements
-- ──────────────────────────────────────────────────────────
create table if not exists weekly_announcements (
  id                bigint generated always as identity primary key,
  round             int     not null unique,
  draw_date         date    not null,
  winning_numbers   jsonb   not null,
  stats             jsonb   not null,
  total_extractions int     not null default 0,
  published_at      timestamptz default now()
);

comment on column weekly_announcements.stats is
  '{"1등":0,"2등":0,"3등":0,"4등":0,"5등":0,"낙첨":0,"total":0}';


-- ──────────────────────────────────────────────────────────
-- 인덱스
-- ──────────────────────────────────────────────────────────
create index if not exists idx_user_extractions_target_round
  on user_extractions (target_round);

create index if not exists idx_user_extractions_session_id
  on user_extractions (session_id);

create index if not exists idx_user_extractions_rank
  on user_extractions (rank)
  where rank is not null;
