"""
카페 크롤링 CSV → Supabase draw_results 동기화

CSV 컬럼: ball_set, round, draw_date, n1~n6, bonus
회차당 2행: 첫 번째 행 = 모의추첨(is_winning=false), 두 번째 행 = 실제 당첨(is_winning=true)

동작:
  - DB에 없는 (round, is_winning) 조합 → INSERT
  - DB에 있지만 ball_set이 NULL인 행 → UPDATE ball_set만 채움

사용법:
  python scripts/sync_cafe_history.py           # 전체 동기화
  python scripts/sync_cafe_history.py --last 5  # 최근 N회차만
"""

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env")

sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from web.database import get_supabase

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

DATA_PATH = PROJECT_ROOT / "data/history_from_cafe.csv"


def sync(last: int | None = None):
    if not DATA_PATH.exists():
        logger.error(f"CSV 파일 없음: {DATA_PATH}")
        sys.exit(1)

    # ── CSV 로드: 회차당 2행 모두 유지, is_winning 컬럼 추가 ──────────
    df = pd.read_csv(
        DATA_PATH, header=None,
        names=["ball_set", "round", "draw_date",
               "n1", "n2", "n3", "n4", "n5", "n6", "bonus"],
        dtype=str,
    )

    # 회차별로 순서 보존 후 첫 행=모의(False), 두 번째 행=실제(True) 마킹
    df["is_winning"] = (
        df.groupby("round").cumcount() > 0  # 0번째=False, 1번째=True
    )
    df = df.sort_values(["round", "is_winning"]).reset_index(drop=True)

    # --last 옵션: 최근 N회차
    if last:
        recent_rounds = df["round"].unique()
        recent_rounds_sorted = sorted(recent_rounds, key=lambda x: int(x))[-last:]
        df = df[df["round"].isin(recent_rounds_sorted)].reset_index(drop=True)

    rounds = df["round"].unique()
    logger.info(f"CSV: {len(rounds)}개 회차, {len(df)}행 ({min(rounds)} ~ {max(rounds)})")

    db = get_supabase()

    # ── DB에서 현재 상태 조회: (round, is_winning) 조합 ──────────────
    existing = db.table("draw_results").select("round, is_winning, ball_set").execute().data
    existing_keys: set[tuple] = {(row["round"], row["is_winning"]) for row in existing}
    existing_no_ballset: set[tuple] = {
        (row["round"], row["is_winning"])
        for row in existing
        if row["ball_set"] is None
    }

    to_insert = []
    to_update = []

    for _, row in df.iterrows():
        r = int(row["round"])
        is_winning = bool(row["is_winning"])
        key = (r, is_winning)

        new_row = {
            "round":      r,
            "draw_date":  str(row["draw_date"]),
            "is_winning": is_winning,
            "ball_set":   int(row["ball_set"]),
            "n1": int(row["n1"]), "n2": int(row["n2"]), "n3": int(row["n3"]),
            "n4": int(row["n4"]), "n5": int(row["n5"]), "n6": int(row["n6"]),
            "bonus":      int(row["bonus"]),
        }

        if key not in existing_keys:
            to_insert.append(new_row)
        elif key in existing_no_ballset:
            to_update.append({"round": r, "is_winning": is_winning, "ball_set": int(row["ball_set"])})

    # ── INSERT ────────────────────────────────────────────────────────
    if to_insert:
        logger.info(f"신규 삽입: {len(to_insert)}행")
        for i in range(0, len(to_insert), 100):
            batch = to_insert[i:i + 100]
            db.table("draw_results").insert(batch).execute()
            logger.info(f"  삽입 {i + len(batch)}/{len(to_insert)} 완료")
    else:
        logger.info("신규 삽입할 행 없음")

    # ── UPDATE ball_set ───────────────────────────────────────────────
    if to_update:
        logger.info(f"ball_set 업데이트: {len(to_update)}행")
        for item in to_update:
            db.table("draw_results") \
              .update({"ball_set": item["ball_set"]}) \
              .eq("round", item["round"]) \
              .eq("is_winning", item["is_winning"]) \
              .execute()
    else:
        logger.info("ball_set 업데이트할 행 없음")

    logger.info("동기화 완료.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="카페 CSV → Supabase 동기화")
    parser.add_argument("--last", type=int, default=None, help="최근 N개 회차만")
    args = parser.parse_args()
    sync(args.last)
