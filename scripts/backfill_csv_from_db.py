"""
DB(draw_results)의 등수별 상세(winners_1~5/total_prize_1~5/total_sales/prize_1~5)를
data/history_from_cafe.csv 로 역백필.

배경: 기존 835~1230회차는 CSV에 이 컬럼들이 아예 없던 시절(10컬럼)에 적재되었고,
scripts/backfill_prize_details.py로 DB만 채웠다. 이 스크립트는 그 DB 값을 CSV에도
반영해 CSV를 26컬럼(ball_set, round, draw_date, n1~n6, bonus, winners_1~5,
total_prize_1~5, total_sales, prize_1~5)으로 맞춘다. 실행 전 원본을 .bak으로 백업한다.

DB가 source of truth이므로 재실행해도 안전(항상 DB 값으로 덮어씀).

사용법:
  python scripts/backfill_csv_from_db.py
"""

import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env")
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from web.database import get_supabase_admin

DATA_PATH = PROJECT_ROOT / "data/history_from_cafe.csv"

DETAIL_FIELDS = [
    "winners_1", "winners_2", "winners_3", "winners_4", "winners_5",
    "total_prize_1", "total_prize_2", "total_prize_3", "total_prize_4", "total_prize_5",
    "total_sales",
    "prize_1", "prize_2", "prize_3", "prize_4", "prize_5",
]


def main():
    # 이미 21컬럼으로 넓혀져 있어도 앞 10개(원본 카페 데이터)만 사용 — 재실행 시에도 안전
    df = pd.read_csv(
        DATA_PATH, header=None,
        names=["ball_set", "round", "draw_date", "n1", "n2", "n3", "n4", "n5", "n6", "bonus"],
        dtype=str,
        usecols=range(10),
    )
    df["is_winning"] = df.groupby("round").cumcount() > 0

    db = get_supabase_admin()
    rows = db.table("draw_results").select("round, is_winning, " + ", ".join(DETAIL_FIELDS)).execute().data
    lookup = {(r["round"], r["is_winning"]): r for r in rows}

    def detail_row(row):
        key = (int(row["round"]), bool(row["is_winning"]))
        db_row = lookup.get(key)
        if not db_row:
            return pd.Series([""] * len(DETAIL_FIELDS))
        return pd.Series(["" if db_row.get(f) is None else str(db_row[f]) for f in DETAIL_FIELDS])

    details = df.apply(detail_row, axis=1)
    details.columns = DETAIL_FIELDS

    out = pd.concat([df.drop(columns=["is_winning"]), details], axis=1)

    backup = DATA_PATH.with_suffix(".csv.bak")
    shutil.copy(DATA_PATH, backup)
    print(f"백업: {backup}")

    out.to_csv(DATA_PATH, header=False, index=False)
    print(f"완료: {len(out)}행, {out.shape[1]}컬럼 → {DATA_PATH}")

    filled = (details["winners_1"] != "").sum()
    print(f"등수별 상세 채워진 행(실제 당첨): {filled}행")


if __name__ == "__main__":
    main()
