"""
Supabase draw_results 초기 데이터 적재 스크립트
history_from_cafe.csv 에서 직접 읽어 draw_results 테이블에 저장
(외부 API 호출 없음)

CSV 컬럼: ball_set, round, draw_date, n1~n6, bonus
단, 같은 회차에 ball_set이 여러 개 있으므로 회차당 1행만 저장

사용법:
  cd /path/to/lotto645
  python scripts/seed_draw_results.py
  python scripts/seed_draw_results.py --last 10  # 최근 N회차만
"""

import argparse
import sys
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from web.database import DrawResult, SessionLocal, init_db

DATA_PATH = Path(__file__).parent.parent / "data/history_from_cafe.csv"


def main(last: int | None = None):
    init_db()

    df = pd.read_csv(
        DATA_PATH, header=None,
        names=["ball_set","round","draw_date","n1","n2","n3","n4","n5","n6","bonus"]
    )

    # 회차당 두 번째 행(실제 당첨번호) 사용 - 첫 번째는 리허설
    deduped = df.drop_duplicates(subset="round", keep="last").sort_values("round")

    if last:
        deduped = deduped.tail(last)

    print(f"총 {len(deduped)}개 회차 적재 시작: {deduped['round'].iloc[0]} ~ {deduped['round'].iloc[-1]}")

    db = SessionLocal()
    ok = skip = 0

    try:
        for _, row in deduped.iterrows():
            r = int(row["round"])

            # 이미 있으면 스킵
            if db.query(DrawResult).filter(DrawResult.round == r).first():
                skip += 1
                continue

            draw = DrawResult(
                round     = r,
                draw_date = date.fromisoformat(str(row["draw_date"])),
                n1=int(row["n1"]), n2=int(row["n2"]), n3=int(row["n3"]),
                n4=int(row["n4"]), n5=int(row["n5"]), n6=int(row["n6"]),
                bonus     = int(row["bonus"]),
            )
            db.add(draw)
            ok += 1

        db.commit()
    finally:
        db.close()

    print(f"완료: 저장 {ok}개 / 스킵(기존) {skip}개")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--last", type=int, default=None, help="최근 N개 회차만")
    args = parser.parse_args()
    main(args.last)
