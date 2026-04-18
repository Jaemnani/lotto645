"""
예측 유의미성 검증 — Walk-forward Backtest

절차:
  1. 회차 t 이전 데이터로만 모델을 fit
  2. 회차 t 의 실제 ball_set 에 대해 예측 상위 6개 산출
  3. 실제 당첨번호와 비교하여 적중 개수 기록
  4. t = start_round .. end_round 반복

비교 baseline:
  - Random    : 1~45 에서 6개 무작위 (기대 hits = 6×(6/45) ≈ 0.800)
  - Frequency : 현재 m03 모델 (공세트별 베이지안 빈도)
"""

import os
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats

from m03_model import BayesianFrequencyModel
from m03_train import fetch_history

NUM_BALLS = 45
NUM_MAIN  = 6


def evaluate(df: pd.DataFrame, start_round: int, alpha: float = 1.0,
             include_bonus: bool = False, seed: int = 42):
    rng = np.random.default_rng(seed)
    rounds = sorted(df["round"].unique())
    rounds = [r for r in rounds if r >= start_round]

    hits_model  = []
    hits_random = []
    per_set_hits = defaultdict(list)

    for r in rounds:
        train_df = df[df["round"] < r]
        test_rows = df[df["round"] == r]
        if len(train_df) < 30:
            continue

        model = BayesianFrequencyModel(alpha=alpha).fit(train_df, include_bonus)

        for _, row in test_rows.iterrows():
            bs = int(row["ball_set"])
            true_nums = {int(row[c]) for c in ["n1","n2","n3","n4","n5","n6"]}

            probs = model.posterior(bs)
            pred_top6 = set((np.argsort(probs)[::-1][:NUM_MAIN] + 1).tolist())
            hits_model.append(len(pred_top6 & true_nums))
            per_set_hits[bs].append(len(pred_top6 & true_nums))

            rand6 = set(rng.choice(range(1, 46), size=6, replace=False).tolist())
            hits_random.append(len(rand6 & true_nums))

    return hits_model, hits_random, per_set_hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-round", type=int, default=None,
                    help="데이터 시작 회차")
    ap.add_argument("--start-round", type=int, default=1000,
                    help="백테스트 시작 회차 (이전 데이터로 학습)")
    ap.add_argument("--include-mock", action="store_true")
    ap.add_argument("--alpha", type=float, default=1.0)
    args = ap.parse_args()

    print("데이터 로드 중...")
    df = fetch_history(args.from_round, args.include_mock)
    print(f"  총 {len(df)}행, 회차 {df['round'].min()}~{df['round'].max()}")

    print(f"\n백테스트: 회차 {args.start_round} ~ {df['round'].max()}")
    hits_m, hits_r, per_set = evaluate(df, args.start_round, args.alpha,
                                       include_bonus=False)

    hm, hr = np.array(hits_m), np.array(hits_r)

    print(f"\n{'='*60}")
    print(f"결과 ({len(hm)} 회차)")
    print(f"{'='*60}")
    print(f"  Model    : 평균 hits = {hm.mean():.4f}  std = {hm.std():.3f}")
    print(f"  Random   : 평균 hits = {hr.mean():.4f}  std = {hr.std():.3f}")
    print(f"  이론 랜덤: 0.8000 (6×6/45)")

    # 쌍체 t-검정
    t, p = stats.ttest_rel(hm, hr)
    print(f"\n  Paired t-test (Model vs Random):")
    print(f"    t = {t:.3f}   p = {p:.4f}")
    if p < 0.05 and t > 0:
        print(f"    → 모델이 랜덤보다 유의하게 우수 (p<0.05)")
    else:
        print(f"    → 유의한 우위 없음")

    # 적중 개수 분포
    print(f"\n  적중 개수 분포:")
    print(f"    hits:  {'  '.join(str(i) for i in range(7))}")
    print(f"    model: " + "  ".join(f"{(hm==i).sum():>2}" for i in range(7)))
    print(f"    random:" + "  ".join(f"{(hr==i).sum():>2}" for i in range(7)))

    # 공세트별
    print(f"\n  공세트별 평균 hits:")
    for bs in sorted(per_set.keys()):
        h = np.array(per_set[bs])
        print(f"    공세트 {bs}: {h.mean():.4f}  (n={len(h)})")


if __name__ == "__main__":
    main()
