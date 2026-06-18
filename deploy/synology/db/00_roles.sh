#!/bin/bash
# DB 최초 init 시 1회 실행 (docs/02-backend.md 2-1).
# PostgREST 가 쓰는 역할 4종 생성 + authenticator 비밀번호 동기화.
#   anon          : 공개 읽기 (RLS public read)
#   authenticated : 로그인 사용자(현재 미사용, 호환용)
#   service_role  : 적재/관리 (RLS 우회)
#   authenticator : PostgREST 접속 전용 (위 역할들로 SET ROLE)
set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
	do \$\$
	begin
	  if not exists (select from pg_roles where rolname = 'anon') then
	    create role anon nologin noinherit;
	  end if;
	  if not exists (select from pg_roles where rolname = 'authenticated') then
	    create role authenticated nologin noinherit;
	  end if;
	  if not exists (select from pg_roles where rolname = 'service_role') then
	    create role service_role nologin noinherit bypassrls;
	  end if;
	  if not exists (select from pg_roles where rolname = 'authenticator') then
	    create role authenticator login noinherit;
	  end if;
	end
	\$\$;

	-- PostgREST 의 PGRST_DB_URI 비밀번호와 반드시 동일해야 함 (불일치 시 접속 실패)
	alter role authenticator with password '${AUTHENTICATOR_PASSWORD}';

	-- authenticator 가 각 역할로 전환할 수 있도록
	grant anon, authenticated, service_role to authenticator;

	-- 스키마 사용 권한
	grant usage on schema public to anon, authenticated, service_role;
EOSQL

echo "[00_roles] anon / authenticated / service_role / authenticator 생성 완료"
