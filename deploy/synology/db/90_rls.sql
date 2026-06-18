-- RLS + 권한 (docs/02-backend.md). 마이그레이션(10_*)으로 테이블 생성된 뒤 실행.
-- anon = 공개 읽기만, service_role = 전체 쓰기(적재). 멱등(재실행 안전).

-- ── 공개 읽기 대상: draw_results, weekly_announcements ──────────────────────────
alter table draw_results          enable row level security;
alter table weekly_announcements  enable row level security;
alter table user_extractions      enable row level security;

-- draw_results : 누구나 SELECT
drop policy if exists "public read draw_results" on draw_results;
create policy "public read draw_results" on draw_results
  for select to anon, authenticated using (true);

-- weekly_announcements : 누구나 SELECT
drop policy if exists "public read announcements" on weekly_announcements;
create policy "public read announcements" on weekly_announcements
  for select to anon, authenticated using (true);

-- user_extractions : 익명 INSERT 허용(번호 저장) + SELECT 허용.
--   (개인정보 컬럼 user_name/phone_last4 는 웹앱이 select 하지 않음. 민감하면 뷰로 분리 권장.)
drop policy if exists "public read extractions"   on user_extractions;
drop policy if exists "public insert extractions" on user_extractions;
create policy "public read extractions" on user_extractions
  for select to anon, authenticated using (true);
create policy "public insert extractions" on user_extractions
  for insert to anon, authenticated with check (true);

-- ── 테이블/시퀀스 권한 ──────────────────────────────────────────────────────────
grant select on draw_results, weekly_announcements, user_extractions to anon, authenticated;
grant insert on user_extractions to anon, authenticated;

-- service_role : 전체 (적재/관리). bypassrls 라 정책 무관하게 동작.
grant all on all tables    in schema public to service_role;
grant all on all sequences in schema public to service_role;

-- PostgREST 스키마 캐시 리로드
notify pgrst, 'reload schema';
