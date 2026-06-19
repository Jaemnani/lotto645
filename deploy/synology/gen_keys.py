#!/usr/bin/env python3
"""
anon / service_role JWT 키 발급 (docs/02-backend.md 2-1 (4))
============================================================
self-host PostgREST 의 PGRST_JWT_SECRET 과 동일한 JWT_SECRET 으로 HS256 서명한
JWT 를 만든다. 표준 라이브러리만 사용 (PyJWT 불필요).

사용:
    python3 gen_keys.py                 # .env 의 JWT_SECRET 사용
    python3 gen_keys.py <JWT_SECRET>    # 직접 전달

출력된 ANON_KEY → web 의 SUPABASE_KEY, SERVICE_KEY → 적재용 SUPABASE_SERVICE_KEY.
"""

from __future__ import annotations   # NAS 의 구버전 Python(3.8/3.9)에서도 `str | None` 동작

import base64
import hashlib
import hmac
import json
import sys
from pathlib import Path

# 만료 없는 장기 키 (Supabase anon/service 키와 동일 컨셉). 발급 시각만 박는다.
# iat 는 고정값으로 두어 재실행 시 동일 키가 나오도록(멱등) — 2024-01-01 UTC.
IAT = 1704067200


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def make_jwt(role: str, secret: str) -> str:
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {"role": role, "iss": "lotto645-selfhost", "iat": IAT}
    seg = _b64url(json.dumps(header, separators=(",", ":")).encode()) + "." + \
          _b64url(json.dumps(payload, separators=(",", ":")).encode())
    sig = hmac.new(secret.encode(), seg.encode(), hashlib.sha256).digest()
    return seg + "." + _b64url(sig)


def _load_secret_from_env() -> str | None:
    env = Path(__file__).resolve().parent / ".env"
    if not env.exists():
        return None
    for line in env.read_text().splitlines():
        line = line.strip()
        if line.startswith("JWT_SECRET="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def main() -> int:
    secret = sys.argv[1] if len(sys.argv) > 1 else _load_secret_from_env()
    if not secret:
        print("JWT_SECRET 을 찾을 수 없습니다. .env 에 설정하거나 인자로 전달하세요.",
              file=sys.stderr)
        print("  예: openssl rand -base64 48  → .env 의 JWT_SECRET 에 입력", file=sys.stderr)
        return 1

    print("# --- 아래 값을 .env 에 채워넣으세요 ---")
    print(f"# SUPABASE_KEY (anon)  : web/크롤러 읽기용")
    print(f"ANON_KEY={make_jwt('anon', secret)}")
    print()
    print(f"# SUPABASE_SERVICE_KEY : 크롤러 적재용 (서버 전용)")
    print(f"SERVICE_KEY={make_jwt('service_role', secret)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
