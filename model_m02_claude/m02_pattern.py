"""
추출 패턴 분석
==============
n1~n6이 추출 순서대로 기록되어 있다는 가정 하에:

1. 위치별 번호 분포  : 1번째~6번째로 나온 번호의 범위 패턴
2. 위치별 홀짝 패턴  : 각 위치에서 홀수/짝수 비율
3. 연속 등장 분석    : 이전 회차 번호가 다음 회차에 재등장하는 비율
4. 전이 행렬        : 이번 회차에 X가 나오면 다음 회차에 Y가 나올 확률
5. 간격(Gap) 분석   : 각 번호가 몇 회차 만에 다시 나오는가
6. 공세트별 비교     : 위 분석을 공세트로 구분

사용법:
  python m02_pattern.py
  python m02_pattern.py --ball_set 3   # 특정 공세트만
"""

import os
import argparse
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_PATH  = os.path.join(os.path.dirname(__file__), "../data/history_from_cafe.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "pattern_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_BALLS   = 45
POS_COLS    = ["n1","n2","n3","n4","n5","n6"]  # 추출 순서


def load_data(path):
    df = pd.read_csv(path, header=None,
                     names=["ball_set","round","draw_date","n1","n2","n3","n4","n5","n6","bonus"])
    return df.sort_values(["round","ball_set"]).reset_index(drop=True)


# ── 1. 위치별 번호 분포 ────────────────────────────────────────────────────────
def analyze_position_distribution(df, ball_set=None, tag="전체"):
    subset = df[df["ball_set"] == ball_set] if ball_set else df

    print(f"\n{'='*60}")
    print(f"[ {tag} | 위치별 번호 분포 ]")
    print(f"  (n1=1번째 추출, n6=6번째 추출)")
    print(f"{'='*60}")
    print(f"  {'위치':>4} {'평균':>7} {'중앙값':>7} {'표준편차':>8} {'범위':>12}  판정")

    pos_means = []
    for i, col in enumerate(POS_COLS, 1):
        vals = subset[col].values
        mean = vals.mean()
        med  = np.median(vals)
        std  = vals.std()
        lo, hi = vals.min(), vals.max()
        pos_means.append(mean)
        trend = ""
        if mean < 18:
            trend = "← 낮은 번호 경향"
        elif mean > 28:
            trend = "← 높은 번호 경향"
        print(f"  {i:>4}번째  {mean:>7.2f}  {med:>7.1f}  {std:>8.2f}  [{lo:2d}~{hi:2d}]  {trend}")

    # 단조 경향 검정 (Jonckheere-Terpstra 대신 Spearman)
    corr, p = stats.spearmanr(range(1, 7), pos_means)
    print(f"\n  위치↔평균번호 Spearman 상관: r={corr:.3f}, p={p:.4f}")
    if p < 0.05:
        direction = "앞 위치일수록 낮은 번호" if corr < 0 else "앞 위치일수록 높은 번호"
        print(f"  ★ 유의미한 위치 경향 있음: {direction}")
    else:
        print(f"  → 위치별 번호 편향 없음 (랜덤)")

    # 시각화
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    for i, (col, ax) in enumerate(zip(POS_COLS, axes)):
        vals = subset[col].values
        ax.hist(vals, bins=range(1, 47), color="#3498db", alpha=0.75, edgecolor="white")
        ax.axvline(vals.mean(), color="red", linestyle="--", linewidth=1.5,
                   label=f"μ={vals.mean():.1f}")
        ax.axvline(23, color="grey", linestyle=":", linewidth=1, label="중간(23)")
        ax.set_title(f"{i+1}번째 추출 번호")
        ax.set_xlabel("번호")
        ax.set_ylabel("빈도")
        ax.legend(fontsize=8)
    fig.suptitle(f"위치별 번호 분포 [{tag}]", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fname = f"position_dist_{tag.replace(' ','_')}.png"
    plt.savefig(os.path.join(OUTPUT_DIR, fname), dpi=120)
    plt.close()
    print(f"  [저장] {fname}")
    return pos_means


# ── 2. 위치별 홀짝 패턴 ────────────────────────────────────────────────────────
def analyze_position_odd_even(df, ball_set=None, tag="전체"):
    subset = df[df["ball_set"] == ball_set] if ball_set else df

    print(f"\n[ {tag} | 위치별 홀짝 비율 ]")
    print(f"  {'위치':>4} {'홀수%':>8} {'짝수%':>8}")
    for i, col in enumerate(POS_COLS, 1):
        vals = subset[col].values
        odd_pct = 100 * (vals % 2 == 1).mean()
        print(f"  {i:>4}번째  {odd_pct:>8.1f}  {100-odd_pct:>8.1f}")


# ── 3. 연속 등장 분석 ─────────────────────────────────────────────────────────
def analyze_recurrence(df, ball_set=None, tag="전체"):
    subset = df[df["ball_set"] == ball_set] if ball_set else df
    subset = subset.reset_index(drop=True)

    print(f"\n[ {tag} | 이전 회차 번호 재등장 비율 ]")
    print(f"  (주번호 6개 기준)")

    overlap_counts = []
    for i in range(1, len(subset)):
        prev = set(subset.iloc[i-1][POS_COLS])
        curr = set(subset.iloc[i][POS_COLS])
        overlap_counts.append(len(prev & curr))

    arr = np.array(overlap_counts)
    print(f"  평균 재등장 수: {arr.mean():.3f}개/회차")
    print(f"  랜덤 기대값:    {6*6/45:.3f}개/회차  (6×6/45)")
    print(f"  표준편차: {arr.std():.3f}")

    dist = {k: int((arr == k).sum()) for k in range(7)}
    print(f"  재등장 수 분포: {dist}")

    # 기대 분포 (초기하 분포)
    expected = {}
    n, K, k_draw = 45, 6, 6
    from math import comb
    total = comb(n, k_draw)
    for k in range(7):
        if k > K or k > k_draw:
            expected[k] = 0
        else:
            expected[k] = comb(K, k) * comb(n-K, k_draw-k) / total * len(arr)
    exp_arr = [expected[k] for k in range(7)]
    obs_arr = [dist[k] for k in range(7)]
    chi2, p = stats.chisquare(obs_arr, f_exp=exp_arr)
    print(f"  초기하 분포 대비 chi2={chi2:.2f}, p={p:.4f}")
    if p < 0.05:
        print(f"  ★ 재등장 패턴이 랜덤과 다름")
    else:
        print(f"  → 재등장은 랜덤 수준")
    return arr.mean()


# ── 4. 전이 행렬 ─────────────────────────────────────────────────────────────
def analyze_transition(df, ball_set=None, tag="전체", top_n=10):
    subset = df[df["ball_set"] == ball_set] if ball_set else df
    subset = subset.reset_index(drop=True)

    print(f"\n[ {tag} | 전이 패턴: X가 나온 후 다음 회차에 Y가 나올 빈도 ]")

    mat = np.zeros((NUM_BALLS, NUM_BALLS), dtype=int)
    for i in range(1, len(subset)):
        prev_nums = [int(subset.iloc[i-1][c]) for c in POS_COLS]
        curr_nums = [int(subset.iloc[i][c])   for c in POS_COLS]
        for p in prev_nums:
            for c in curr_nums:
                if 1 <= p <= NUM_BALLS and 1 <= c <= NUM_BALLS:
                    mat[p-1, c-1] += 1

    # 상위 전이 쌍
    triu_flat = mat.flatten()
    top_idx = np.argsort(triu_flat)[::-1][:top_n]
    print(f"  상위 {top_n}개 전이 쌍 (이전→다음):")
    for idx in top_idx:
        r, c = divmod(idx, NUM_BALLS)
        if r != c:  # 자기 자신 제외
            print(f"    {r+1:2d} → {c+1:2d}  {mat[r,c]}회")

    # 히트맵 저장
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(mat, cmap="YlOrRd", aspect="auto")
    ax.set_xlabel("다음 회차 번호")
    ax.set_ylabel("이전 회차 번호")
    ax.set_title(f"전이 행렬 [{tag}]")
    plt.colorbar(im, ax=ax)
    fname = f"transition_{tag.replace(' ','_')}.png"
    plt.savefig(os.path.join(OUTPUT_DIR, fname), dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  [저장] {fname}")
    return mat


# ── 5. 간격(Gap) 분석 ────────────────────────────────────────────────────────
def analyze_gap(df, ball_set=None, tag="전체"):
    subset = df[df["ball_set"] == ball_set] if ball_set else df
    subset = subset.reset_index(drop=True)

    print(f"\n[ {tag} | 번호별 재등장 간격(Gap) 분석 ]")

    gaps_all = []
    last_seen = {}
    cold_numbers = {}   # 마지막 등장 이후 가장 오래된 번호

    for i, row in subset.iterrows():
        nums = [int(row[c]) for c in POS_COLS]
        for n in nums:
            if n in last_seen:
                gaps_all.append(i - last_seen[n])
            last_seen[n] = i

    gaps_all = np.array(gaps_all)
    print(f"  평균 재등장 간격: {gaps_all.mean():.2f}회차")
    print(f"  중앙값: {np.median(gaps_all):.1f}  표준편차: {gaps_all.std():.2f}")
    print(f"  최장 간격: {gaps_all.max()}회차  최단: {gaps_all.min()}회차")

    # 공세트별로 오래 안 나온 번호 (cold numbers)
    last_seen_abs = {}
    for i, row in subset.iterrows():
        for c in POS_COLS:
            n = int(row[c])
            last_seen_abs[n] = i
    all_nums = set(range(1, NUM_BALLS + 1))
    appeared = set(last_seen_abs.keys())
    never = all_nums - appeared
    if never:
        print(f"  미등장 번호: {sorted(never)}")

    # 오래 안 나온 상위 10개
    cold = sorted(last_seen_abs.items(), key=lambda x: x[1])[:10]
    total = len(subset)
    print(f"  오래 안 나온 번호 (cold):")
    for num, last_i in cold:
        gap = total - last_i
        print(f"    {num:2d}번 → {gap}회차 전 마지막 등장")

    # 간격 히스토그램
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(gaps_all, bins=range(1, int(gaps_all.max())+2), color="#9b59b6",
            alpha=0.75, edgecolor="white")
    ax.axvline(gaps_all.mean(), color="red", linestyle="--",
               linewidth=1.5, label=f"mean={gaps_all.mean():.1f}")
    ax.set_xlabel("재등장 간격 (회차 수)")
    ax.set_ylabel("빈도")
    ax.set_title(f"번호 재등장 간격 분포 [{tag}]")
    ax.legend()
    plt.tight_layout()
    fname = f"gap_dist_{tag.replace(' ','_')}.png"
    plt.savefig(os.path.join(OUTPUT_DIR, fname), dpi=120)
    plt.close()
    print(f"  [저장] {fname}")


# ── 6. 공세트별 패턴 비교 요약 ───────────────────────────────────────────────
def compare_ball_sets(df):
    print(f"\n{'='*60}")
    print(f"[ 공세트별 패턴 비교 요약 ]")
    print(f"{'='*60}")
    print(f"  {'Set':>4} {'1번째μ':>8} {'6번째μ':>8} {'재등장μ':>9} {'경향':>20}")

    for bs in sorted(df["ball_set"].unique()):
        sub = df[df["ball_set"] == bs].reset_index(drop=True)
        m1 = sub["n1"].mean()
        m6 = sub["n6"].mean()

        overlaps = []
        for i in range(1, len(sub)):
            prev = set(sub.iloc[i-1][POS_COLS])
            curr = set(sub.iloc[i][POS_COLS])
            overlaps.append(len(prev & curr))
        rec_mean = np.mean(overlaps)

        diff = m6 - m1
        if abs(diff) < 1:
            trend = "위치 차이 없음"
        elif diff > 0:
            trend = f"후반 번호 ↑ ({diff:+.1f})"
        else:
            trend = f"전반 번호 ↑ ({diff:+.1f})"

        print(f"  {bs:>4} {m1:>8.2f} {m6:>8.2f} {rec_mean:>9.3f} {trend:>20}")


# ── 진입점 ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ball_set", type=int, default=None,
                        help="특정 공세트만 분석 (미입력시 전체)")
    parser.add_argument("--data_path", type=str, default=DATA_PATH)
    args = parser.parse_args()

    df = load_data(args.data_path)

    if args.ball_set:
        tag = f"공세트 {args.ball_set}"
        sub = df[df["ball_set"] == args.ball_set]
        analyze_position_distribution(sub, tag=tag)
        analyze_position_odd_even(sub, tag=tag)
        analyze_recurrence(sub, tag=tag)
        analyze_transition(sub, tag=tag)
        analyze_gap(sub, tag=tag)
    else:
        # 전체 분석
        analyze_position_distribution(df, tag="전체")
        analyze_position_odd_even(df, tag="전체")
        analyze_recurrence(df, tag="전체")
        analyze_transition(df, tag="전체")
        analyze_gap(df, tag="전체")

        # 공세트별 비교
        compare_ball_sets(df)

        # 공세트별 위치 분포 요약
        print(f"\n[ 공세트별 위치별 평균 번호 ]")
        print(f"  {'Set':>4}  " + "  ".join(f"{i}번째" for i in range(1, 7)))
        for bs in sorted(df["ball_set"].unique()):
            sub = df[df["ball_set"] == bs]
            means = [sub[c].mean() for c in POS_COLS]
            print(f"  {bs:>4}  " + "  ".join(f"{m:6.2f}" for m in means))

    print(f"\n패턴 분석 완료. 저장 위치: {OUTPUT_DIR}")
