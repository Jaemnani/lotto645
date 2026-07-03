"""
draw_results 등수별 상세(winners_1~5/total_prize_1~5/total_sales/prize_1~5) 백필

기존 회차(is_winning=true 행)는 이 컬럼들이 없던 시절에 적재되어 NULL 상태다.
동행복권 공홈 API(selectPstLt645InfoNew.do, srchDir=center)는 한 번 호출에 최대
10회차(오래된5+선택1+최신4)를 반환하므로, 매번 남은 대상 중 가장 작은 회차를 기준으로
조회해 한 번에 여러 회차를 함께 채운다 (396회차라도 API 호출은 수십 건 수준).

사용법:
  python scripts/backfill_prize_details.py            # 전체 백필
  python scripts/backfill_prize_details.py --dry-run  # DB 변경 없이 대상/조회 결과만 출력
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env")
sys.path.insert(0, str(PROJECT_ROOT))

from web.database import get_supabase_admin
from web.fetcher import fetch_draws_around, open_session

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

DETAIL_FIELDS = [
    "winners_1", "winners_2", "winners_3", "winners_4", "winners_5",
    "total_prize_1", "total_prize_2", "total_prize_3", "total_prize_4", "total_prize_5",
    "total_sales",
    "prize_1", "prize_2", "prize_3", "prize_4", "prize_5",
]

REQUEST_DELAY = 0.5     # 회차 조회 사이 대기 (예의상 최소 간격)
RETRY_BACKOFF = 2.0     # 조회 실패 시 재시도 전 대기
MAX_RETRIES = 2


class _SessionBox:
    """세션을 하나 열어 계속 재사용 — 브라우저에서 회차 이동 버튼을 연달아 눌렀을 때
    빠른 것과 같은 이유(연결 재사용)로, 매번 새 세션을 여는 것보다 훨씬 빠르고 덜 막힌다.
    최초 연결 자체가 확률적으로 실패할 수 있어(관찰상 절반 정도) 재시도를 둔다.
    재시도로도 계속 실패하면 세션이 죽은 것으로 보고 한 번 다시 연다."""
    def __init__(self):
        self.session = self._open_with_retry()

    def reopen(self):
        self.session = self._open_with_retry()

    @staticmethod
    def _open_with_retry(retries: int = 20, backoff: float = 3.0, max_backoff: float = 30.0) -> requests.Session:
        """접속 자체가 확률적으로 막히는 구간이 있어(관찰상 절반 이상 실패하는 시간대도 있음)
        지수 백오프로 충분히 오래 재시도한다 — 한 번 뚫리면 이후 조회는 매우 빠르다."""
        wait = backoff
        for attempt in range(retries):
            try:
                return open_session()
            except Exception as e:
                logger.warning(f"[백필] 세션 연결 실패 ({attempt + 1}/{retries}), {wait:.0f}s 후 재시도: {e}")
                time.sleep(wait)
                wait = min(wait * 1.5, max_backoff)
        raise RuntimeError("공홈 세션 연결 반복 실패")


def _fetch_with_retry(round_no: int, box: _SessionBox) -> list[dict]:
    for attempt in range(MAX_RETRIES + 1):
        batch = fetch_draws_around(round_no, session=box.session)
        if batch:
            return batch
        if attempt < MAX_RETRIES:
            logger.warning(f"[백필] {round_no}회차 조회 실패, {RETRY_BACKOFF}s 후 재시도 ({attempt + 1}/{MAX_RETRIES})")
            time.sleep(RETRY_BACKOFF)
            try:
                box.reopen()
            except Exception as e:
                logger.warning(f"[백필] 세션 재연결 실패: {e}")
    return []


def backfill(dry_run: bool = False):
    db = get_supabase_admin()

    rows = (
        db.table("draw_results")
        .select("round, " + ", ".join(DETAIL_FIELDS))
        .eq("is_winning", True)
        .execute()
        .data
    )
    missing_rounds = sorted(
        row["round"] for row in rows
        if any(row[f] is None for f in DETAIL_FIELDS)
    )

    if not missing_rounds:
        logger.info("백필할 회차 없음 — 이미 모두 채워져 있음.")
        return

    logger.info(f"백필 대상: {len(missing_rounds)}회차 ({missing_rounds[0]}~{missing_rounds[-1]})")

    pending = set(missing_rounds)
    fetched: dict[int, dict] = {}
    api_calls = 0
    box = _SessionBox()

    while pending:
        cursor = min(pending)
        batch = _fetch_with_retry(cursor, box)
        api_calls += 1

        if not batch:
            logger.warning(f"[백필] {cursor}회차 조회 포기 — 이 회차는 건너뜀")
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
            # cursor 자체가 응답 목록에 없는 경우(드묾) 무한루프 방지
            logger.warning(f"[백필] {cursor}회차가 응답에 없음 — 건너뜀")
            pending.discard(cursor)

        if pending:
            time.sleep(REQUEST_DELAY)

    logger.info(f"공홈 조회 완료: {len(fetched)}/{len(missing_rounds)}회차 확보 (API 호출 {api_calls}회)")

    skipped = sorted(set(missing_rounds) - set(fetched))
    if skipped:
        logger.warning(f"[백필] 공홈에서 끝내 조회 못한 회차: {skipped}")

    if dry_run:
        for r in sorted(fetched):
            logger.info(f"  [dry-run] {r}회차 → {fetched[r]}")
        return

    updated = 0
    for r, data in fetched.items():
        extra = {f: data.get(f) for f in DETAIL_FIELDS if data.get(f) is not None}
        if not extra:
            continue
        db.table("draw_results").update(extra).eq("round", r).eq("is_winning", True).execute()
        updated += 1
        logger.info(f"  {r}회차 업데이트 완료")

    logger.info(f"[백필] DB 업데이트 완료: {updated}회차")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="draw_results 등수별 상세 백필")
    parser.add_argument("--dry-run", action="store_true", help="DB 변경 없이 대상/조회 결과만 출력")
    args = parser.parse_args()
    backfill(args.dry_run)
