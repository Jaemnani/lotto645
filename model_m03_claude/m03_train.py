"""
Step 2 - 학습(집계)
데이터: Supabase draw_results (is_winning=true)
모델: BayesianFrequencyModel

학습 전략
  · "학습"이지만 실제로는 공세트별 번호 빈도 집계
  · 실제 당첨번호(is_winning=true)만 기본값
  · 필요 시 모의추첨 포함, 보너스 포함 옵션
  · 결과는 .npz 로 저장
"""

import os
import json
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client

from m03_model import BayesianFrequencyModel

ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_PATH = os.path.join(os.path.dirname(__file__), "best_m03.npz")
META_PATH = os.path.join(os.path.dirname(__file__), "training_meta.json")


def fetch_history(
    from_round: int | None = None,
    include_mock: bool = False,
) -> pd.DataFrame:
    load_dotenv(os.path.join(ROOT, ".env"))
    sb = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
    q = sb.table("draw_results").select(
        "round,draw_date,is_winning,ball_set,n1,n2,n3,n4,n5,n6,bonus"
    )
    if not include_mock:
        q = q.eq("is_winning", True)
    if from_round is not None:
        q = q.gte("round", from_round)
    q = q.order("round").order("ball_set")

    rows, start, step = [], 0, 1000
    while True:
        r = q.range(start, start + step - 1).execute().data
        if not r:
            break
        rows.extend(r)
        if len(r) < step:
            break
        start += step

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["ball_set"]).copy()
    df["ball_set"] = df["ball_set"].astype(int)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-round", type=int, default=None,
                    help="사용할 최소 회차 (기본: 전체)")
    ap.add_argument("--include-mock", action="store_true",
                    help="모의추첨(is_winning=false) 행도 포함")
    ap.add_argument("--include-bonus", action="store_true",
                    help="보너스 번호도 빈도에 포함")
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="Laplace 평활 계수 (기본 1.0)")
    args = ap.parse_args()

    print(f"[{datetime.now().strftime('%H:%M:%S')}] 데이터 로드 중...")
    df = fetch_history(args.from_round, args.include_mock)
    print(f"  총 {len(df)}행  |  회차 {df['round'].min()}~{df['round'].max()}")
    print(f"  공세트 분포:")
    print(df.groupby("ball_set").size().to_string())

    model = BayesianFrequencyModel(alpha=args.alpha).fit(df, args.include_bonus)
    model.save(CKPT_PATH)
    print(f"\n모델 저장: {CKPT_PATH}")

    meta = {
        "trained_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "from_round": args.from_round,
        "include_mock": args.include_mock,
        "include_bonus": args.include_bonus,
        "alpha": args.alpha,
        "num_draws_global": model.num_draws_global,
        "num_draws_per_set": model.num_draws_per_set,
        "round_range": list(model.round_range),
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"메타 저장: {META_PATH}")


if __name__ == "__main__":
    main()
