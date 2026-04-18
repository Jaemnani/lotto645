"""
Step 1 - 빈도 편향 통계 분석

- 전체 및 공세트별 번호 출현 빈도
- 카이제곱 균일성 검정 (p-value)
- 상위/하위 N개 번호 리포트
- 구간별(1-9, 10-19, ...) 분포
- matplotlib 히스토그램 저장
"""

import os
import argparse
import numpy as np
import pandas as pd
from scipy import stats
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from m03_train import fetch_history

NUM_BALLS  = 45
NUM_SETS   = 5
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "analysis_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def count_frequencies(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    nums = df[cols].values.flatten()
    nums = nums[(nums >= 1) & (nums <= NUM_BALLS)].astype(int)
    cnt = Counter(nums.tolist())
    return np.array([cnt.get(i, 0) for i in range(1, NUM_BALLS + 1)], dtype=float)


def print_header(txt: str):
    print("\n" + "=" * 66)
    print(txt)
    print("=" * 66)


def analyze_global(df: pd.DataFrame, cols: list[str]):
    print_header("전체 빈도 분석")
    counts = count_frequencies(df, cols)
    n = counts.sum()
    exp = n / NUM_BALLS
    chi2, pval = stats.chisquare(counts)

    print(f"  관측 총합 : {int(n)}   기대값/번호 : {exp:.1f}")
    print(f"  최다      : 번호 {counts.argmax()+1:>2}  ({int(counts.max())}회, z={(counts.max()-exp)/np.sqrt(exp*(1-1/45)):+.2f})")
    print(f"  최소      : 번호 {counts.argmin()+1:>2}  ({int(counts.min())}회, z={(counts.min()-exp)/np.sqrt(exp*(1-1/45)):+.2f})")
    print(f"  표준편차  : {counts.std():.2f}  (이론 σ ≈ {np.sqrt(exp*(1-1/45)):.2f})")
    print(f"  Chi²      : {chi2:.2f}   p-value : {pval:.4f}")
    print(f"  결론      : {'유의한 편향 있음 (p<0.05)' if pval < 0.05 else '균등분포와 유의차 없음'}")

    # 상위/하위 10
    idx = np.argsort(counts)[::-1]
    print(f"\n  상위 10:")
    for i in idx[:10]:
        print(f"    번호 {i+1:>2} : {int(counts[i]):>4}회")
    print(f"  하위 10:")
    for i in idx[-10:][::-1]:
        print(f"    번호 {i+1:>2} : {int(counts[i]):>4}회")

    return counts


def analyze_by_set(df: pd.DataFrame, cols: list[str]) -> dict:
    print_header("공세트별 빈도 분석")
    print(f"  {'세트':<4} {'회차':<6} {'관측':<6} {'최다':<14} {'최소':<14} {'Chi²':<8} {'p-value'}")
    per_set = {}
    for bs in range(1, NUM_SETS + 1):
        sub = df[df["ball_set"] == bs]
        if len(sub) == 0:
            continue
        counts = count_frequencies(sub, cols)
        if counts.sum() == 0:
            continue
        chi2, pval = stats.chisquare(counts)
        top = counts.argmax() + 1
        bot = counts.argmin() + 1
        print(f"  {bs:<4} {len(sub):<6} {int(counts.sum()):<6} "
              f"{top:>2}({int(counts[top-1])}회){'':<3} "
              f"{bot:>2}({int(counts[bot-1])}회){'':<3} "
              f"{chi2:<8.2f} {pval:.4f}")
        per_set[bs] = counts
    return per_set


def plot_distribution(counts: np.ndarray, title: str, filename: str):
    fig, ax = plt.subplots(figsize=(14, 4))
    balls = np.arange(1, NUM_BALLS + 1)
    expected = counts.sum() / NUM_BALLS
    colors = ['#d62728' if c >= expected else '#1f77b4' for c in counts]
    ax.bar(balls, counts, color=colors)
    ax.axhline(expected, color='black', linestyle='--', linewidth=0.8,
               label=f'expected={expected:.1f}')
    ax.set_xlabel("ball number")
    ax.set_ylabel("frequency")
    ax.set_title(title)
    ax.set_xticks(balls)
    ax.tick_params(axis='x', labelsize=7)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=100)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-round", type=int, default=None)
    ap.add_argument("--include-mock", action="store_true")
    ap.add_argument("--include-bonus", action="store_true")
    args = ap.parse_args()

    df = fetch_history(args.from_round, args.include_mock)
    cols = ["n1","n2","n3","n4","n5","n6"] + (["bonus"] if args.include_bonus else [])

    print(f"데이터: {len(df)}행  |  회차 {df['round'].min()}~{df['round'].max()}  "
          f"|  include_mock={args.include_mock}  include_bonus={args.include_bonus}")

    global_counts = analyze_global(df, cols)
    per_set = analyze_by_set(df, cols)

    plot_distribution(global_counts,
                      f"global frequency (round {df['round'].min()}~{df['round'].max()})",
                      "freq_global.png")
    for bs, counts in per_set.items():
        plot_distribution(counts, f"ball_set {bs}", f"freq_set{bs}.png")

    print(f"\n그래프 저장: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
