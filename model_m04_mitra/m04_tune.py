"""
m04 α(평활 강도) 튜닝
=====================
공세트 편향 posterior 의 평활 강도 α 를 개발 구간(dev) log loss 로 고른다.
α 가 작으면 세트별 빈도를 그대로 믿고(잡음까지 과신), 클수록 균등분포(6/45) 쪽으로 강하게 당긴다.
운영 m03 는 α=1 로 고정이라 잡음을 과신해 균등분포보다 log loss 가 나빴다 (docs/07 §5.3).

튜닝 구간: 백테스트 dev 구간(최근 holdout 30회 앞의 100회)과 겹치지 않도록 **그보다 앞선 n_rounds 회**에서 고른다.
그래야 m04_backtest.py 의 dev 결과가 α 선택에 쓰이지 않은 표본 밖 평가가 된다. (--window dev 로 예전 방식)

사용:
  python m04_tune.py                          # 백테스트 dev 앞 구간에서 본게임만 vs 리허설 합산 α 그리드
  python m04_tune.py --alphas 1 10 30 100 300
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import m04_data                                     # noqa: E402
from m04_features import NUM_BALLS, build           # noqa: E402
from m04_model import normalize_round               # noqa: E402

UNIFORM_LL = -((6 / 45) * np.log(6 / 45) + (39 / 45) * np.log(39 / 45))


def _real_only_post(ft, k, alpha):
    """s_post_m03 는 α=1 고정이라, 본게임만 쓰는 posterior 를 α 로 다시 계산 (s_post_m03 역산)."""
    i = ft.names.index("s_post_m03")
    n_i = ft.names.index("s_n")
    x = ft.per_round[k]
    n_b = x[0, n_i]
    cnt = x[:, i] * (6 * n_b + NUM_BALLS) - 1.0          # α=1 posterior → 카운트 복원
    return (cnt + alpha) / (6 * n_b + alpha * NUM_BALLS)


def evaluate(hist, alphas, n_rounds, holdout, window="pre", dev_rounds=100):
    ft = build(hist, use_rehearsal=True, alpha=1.0)
    evaluable = [k for k in range(len(hist)) if ft.per_round[k] is not None]
    end = len(evaluable) - holdout
    if window == "pre":
        end -= dev_rounds                     # 백테스트 dev 구간 앞까지만
    targets = evaluable[max(0, end - n_rounds):end]

    def ll(p, y):
        return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())

    rows = []
    for a in alphas:
        fa = build(hist, use_rehearsal=True, alpha=a)
        sr = fa.names.index("sr_post")
        real_ll = np.mean([ll(normalize_round(_real_only_post(ft, k, a) * 6), ft.labels[k]) for k in targets])
        comb_ll = np.mean([ll(normalize_round(fa.per_round[k][:, sr] * 6), ft.labels[k]) for k in targets])
        rows.append((a, real_ll, comb_ll))
    return rows, (int(hist.rounds[targets[0]]), int(hist.rounds[targets[-1]]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="csv", choices=["csv", "supabase"])
    ap.add_argument("--alphas", type=float, nargs="+", default=[1, 3, 10, 30, 100, 300, 1000])
    ap.add_argument("--n-rounds", type=int, default=100)
    ap.add_argument("--holdout", type=int, default=30)
    ap.add_argument("--window", default="pre", choices=["pre", "dev"],
                    help="pre: 백테스트 dev 구간 앞에서 튜닝(기본) / dev: dev 구간에서 (표본 내)")
    ap.add_argument("--dev-rounds", type=int, default=100, help="m04_backtest.py 의 --n-rounds 와 맞출 것")
    args = ap.parse_args()

    rows, (r0, r1) = evaluate(m04_data.load(args.source), args.alphas, args.n_rounds, args.holdout,
                              args.window, args.dev_rounds)
    print(f"튜닝 회차 {r0}~{r1} ({args.window}) | 균등분포 log loss {UNIFORM_LL:.5f} (낮을수록 좋음)\n")
    print("| α | 본게임만 | 본게임+리허설 |")
    print("|---|---|---|")
    for a, r, c in rows:
        print(f"| {a:g} | {r:.5f}{' ✓' if r < UNIFORM_LL else ''} | {c:.5f}{' ✓' if c < UNIFORM_LL else ''} |")
    best = min(rows, key=lambda t: t[2])
    print(f"\n리허설 합산 기준 최적 α = {best[0]:g} (log loss {best[2]:.5f}, 균등 대비 {UNIFORM_LL - best[2]:+.5f})")
    print("✓ = 균등분포보다 좋음. 어떤 α 도 ✓ 가 없으면 공세트 편향 신호가 dev 구간에서 확인되지 않은 것.")


if __name__ == "__main__":
    main()
