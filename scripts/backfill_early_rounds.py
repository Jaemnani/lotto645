"""
카페 게시글이 없던 시절(리허설/볼세트 정보 없음)의 회차를 동행복권 공홈 API로
가져와 DB(draw_results)와 data/history_from_cafe.csv 양쪽에 적재.

기본 대상은 733~834회차 — 현재 CSV/DB의 가장 오래된 회차가 835이고, 카페 크롤링
데이터가 그보다 이전엔 없어서 공홈 API로만 채운다. 이 구간은 회차당 1행만 저장하고
ball_set은 빈 값(NULL)으로 둔다 (다른 구간처럼 리허설 행을 만들지 않음).

DB 저장은 web.fetcher.save_draw_result()를 그대로 재사용 — is_winning=True 행만
다루고 ball_set 없이 INSERT하는 기존 동작 그대로 맞아떨어진다.

CSV는 새 회차들을 오름차순으로 만들어 기존 파일 맨 앞에 이어붙인다 (원본은 .bak 백업).

사용법:
  python scripts/backfill_early_rounds.py                   # 733~834
  python scripts/backfill_early_rounds.py --start 700 --end 834
  python scripts/backfill_early_rounds.py --dry-run          # DB/CSV 변경 없이 조회 결과만 출력
"""

import argparse
import logging
import shutil
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env")
sys.path.insert(0, str(PROJECT_ROOT))

from web.database import get_supabase_admin
from web.fetcher import fetch_draws_around, open_session, save_draw_result

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

DATA_PATH = PROJECT_ROOT / "data/history_from_cafe.csv"

# CSV 뒤쪽 16개 상세 컬럼 — scripts/sync_cafe_history.py의 DETAIL_FIELDS와 순서 동일해야 함
CSV_DETAIL_FIELDS = [
    "winners_1", "winners_2", "winners_3", "winners_4", "winners_5",
    "total_prize_1", "total_prize_2", "total_prize_3", "total_prize_4", "total_prize_5",
    "total_sales",
    "prize_1", "prize_2", "prize_3", "prize_4", "prize_5",
]

REQUEST_DELAY = 0.5
RETRY_BACKOFF = 3.0
MAX_RETRIES = 20   # 최초 접속이 확률적으로 실패하는 구간 대비 (backfill_prize_details.py와 동일 사유)


def _open_with_retry():
    wait = RETRY_BACKOFF
    for attempt in range(MAX_RETRIES):
        try:
            return open_session()
        except Exception as e:
            logger.warning(f"세션 연결 실패 ({attempt + 1}/{MAX_RETRIES}), {wait:.0f}s 후 재시도: {e}")
            time.sleep(wait)
            wait = min(wait * 1.5, 30.0)
    raise RuntimeError("공홈 세션 연결 반복 실패")


def _csv_row(data: dict) -> str:
    nums = data["numbers"]
    cols = ["", str(data["round"]), data["draw_date"]] + [str(n) for n in nums] + [str(data["bonus"])]
    cols += ["" if data.get(f) is None else str(data[f]) for f in CSV_DETAIL_FIELDS]
    return ",".join(cols)


def backfill(start: int, end: int, dry_run: bool = False):
    rounds = list(range(start, end + 1))
    pending = set(rounds)
    fetched: dict[int, dict] = {}

    session = _open_with_retry()

    while pending:
        cursor = min(pending)
        batch = fetch_draws_around(cursor, session=session)
        if not batch:
            logger.warning(f"[백필] {cursor}회차 조회 실패 — 세션 재연결")
            session = _open_with_retry()
            batch = fetch_draws_around(cursor, session=session)

        if not batch:
            logger.warning(f"[백필] {cursor}회차 조회 포기 — 건너뜀")
            pending.discard(cursor)
            continue

        newly = 0
        for item in batch:
            r = item["round"]
            if r in pending:
                fetched[r] = item
                pending.discard(r)
                newly += 1

        logger.info(f"[백필] {cursor}회차 기준 조회 → 신규 {newly}건 확보, 남은 대상 {len(pending)}회차")
        if newly == 0:
            logger.warning(f"[백필] {cursor}회차가 응답에 없음 — 건너뜀")
            pending.discard(cursor)

        if pending:
            time.sleep(REQUEST_DELAY)

    missing = sorted(set(rounds) - set(fetched))
    logger.info(f"공홈 조회 완료: {len(fetched)}/{len(rounds)}회차")
    if missing:
        logger.warning(f"[백필] 끝내 조회 못한 회차: {missing}")

    if dry_run:
        for r in sorted(fetched):
            logger.info(f"  [dry-run] {r}회차 → {fetched[r]}")
        return

    # ── DB 적재 (is_winning=True, ball_set 없음) ──────────────────────────
    db = get_supabase_admin()
    for r in sorted(fetched):
        save_draw_result(db, fetched[r])
        logger.info(f"  {r}회차 DB 저장 완료")

    # ── CSV 적재 — 기존 파일 맨 앞에 오름차순으로 이어붙임 ─────────────────
    new_lines = [_csv_row(fetched[r]) for r in sorted(fetched)]

    backup = DATA_PATH.with_suffix(".csv.bak")
    shutil.copy(DATA_PATH, backup)
    logger.info(f"백업: {backup}")

    with open(DATA_PATH, "r") as f:
        existing_lines = [line.rstrip("\n") for line in f if line.strip()]

    with open(DATA_PATH, "w") as f:
        f.write("\n".join(new_lines + existing_lines) + "\n")

    logger.info(f"CSV 반영 완료: {len(new_lines)}행 추가 (총 {len(new_lines) + len(existing_lines)}행)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="리허설 없는 구버전 회차 공홈 백필 (DB+CSV)")
    parser.add_argument("--start", type=int, default=733, help="시작 회차 (기본 733)")
    parser.add_argument("--end", type=int, default=834, help="끝 회차 (기본 834)")
    parser.add_argument("--dry-run", action="store_true", help="DB/CSV 변경 없이 조회 결과만 출력")
    args = parser.parse_args()
    backfill(args.start, args.end, args.dry_run)
