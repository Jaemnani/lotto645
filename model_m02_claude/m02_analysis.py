"""
Step 1: 공세트번호별 추첨 패턴 분석
- 번호 빈도 분포 (chi-square 균일성 검정 포함)
- 구간별 분포 (1-9, 10-19, 20-29, 30-39, 40-45)
- 홀짝 비율
- 합계 분포
- 번호 공출현 상관관계
"""

import numpy as np
import pandas as pd
from scipy import stats
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

# ── 데이터 로드 ────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/history_from_cafe.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "analysis_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, header=None,
                     names=["ball_set", "round", "draw_date", "n1", "n2", "n3", "n4", "n5", "n6", "bonus"])
    return df

# ── 헬퍼 함수 ─────────────────────────────────────────────────────────────────
def get_number_cols():
    return ["n1", "n2", "n3", "n4", "n5", "n6", "bonus"]

def freq_array(rows: pd.DataFrame) -> np.ndarray:
    """1~45 각 번호의 등장 횟수를 배열로 반환"""
    counts = np.zeros(45, dtype=int)
    for col in get_number_cols():
        for v in rows[col]:
            if 1 <= v <= 45:
                counts[v - 1] += 1
    return counts

def main_freq_array(rows: pd.DataFrame) -> np.ndarray:
    """보너스 제외, 주번호(n1~n6)만"""
    counts = np.zeros(45, dtype=int)
    for col in ["n1", "n2", "n3", "n4", "n5", "n6"]:
        for v in rows[col]:
            if 1 <= v <= 45:
                counts[v - 1] += 1
    return counts

def draw_sums(rows: pd.DataFrame) -> list:
    """회차별 주번호 합계"""
    return (rows[["n1","n2","n3","n4","n5","n6"]]).sum(axis=1).tolist()

def odd_even_ratio(rows: pd.DataFrame) -> tuple:
    nums = rows[["n1","n2","n3","n4","n5","n6"]].values.flatten()
    odd = int(np.sum(nums % 2 == 1))
    even = int(np.sum(nums % 2 == 0))
    return odd, even

def range_distribution(rows: pd.DataFrame) -> dict:
    """구간별 번호 출현 비율 (주번호만)"""
    nums = rows[["n1","n2","n3","n4","n5","n6"]].values.flatten()
    ranges = {
        "1-9":   int(np.sum((nums >= 1)  & (nums <= 9))),
        "10-19": int(np.sum((nums >= 10) & (nums <= 19))),
        "20-29": int(np.sum((nums >= 20) & (nums <= 29))),
        "30-39": int(np.sum((nums >= 30) & (nums <= 39))),
        "40-45": int(np.sum((nums >= 40) & (nums <= 45))),
    }
    return ranges

def cooccurrence_matrix(rows: pd.DataFrame) -> np.ndarray:
    """45x45 공출현 빈도 행렬 (주번호만)"""
    mat = np.zeros((45, 45), dtype=int)
    for _, row in rows.iterrows():
        nums = sorted([row[c] - 1 for c in ["n1","n2","n3","n4","n5","n6"] if 1 <= row[c] <= 45])
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                mat[nums[i], nums[j]] += 1
                mat[nums[j], nums[i]] += 1
    return mat

# ── 분석 함수 ──────────────────────────────────────────────────────────────────
def analyze_frequency(df: pd.DataFrame):
    """공세트별 번호 빈도 분석 + 균일성 검정"""
    ball_sets = sorted(df["ball_set"].unique())
    all_freqs = {}

    print("=" * 60)
    print("[ 공세트별 번호 빈도 분석 (주번호 + 보너스) ]")
    print("=" * 60)

    for bs in ball_sets:
        rows = df[df["ball_set"] == bs]
        freq = freq_array(rows)
        all_freqs[bs] = freq

        total = freq.sum()
        expected = total / 45.0
        chi2, p = stats.chisquare(freq, f_exp=[expected] * 45)

        top5 = (np.argsort(freq)[-5:][::-1] + 1).tolist()
        bot5 = (np.argsort(freq)[:5] + 1).tolist()

        print(f"\n공세트 {bs} | 총 추출 {int(total)}개 | 회차 수 {len(rows)}")
        print(f"  평균 빈도: {expected:.2f} (균일 기대값)")
        print(f"  chi2={chi2:.2f}, p={p:.4f} → {'★ 유의미한 편차 있음' if p < 0.05 else '균일 분포 범위'}")
        print(f"  상위 5개 번호: {top5}")
        print(f"  하위 5개 번호: {bot5}")

    # ── 번호별 빈도 비교 히트맵 저장 ───────────────────────────────────────────
    fig, axes = plt.subplots(len(ball_sets), 1, figsize=(14, 3 * len(ball_sets)), sharex=True)
    for ax, bs in zip(axes, ball_sets):
        freq = all_freqs[bs]
        colors = ["#e74c3c" if f == freq.max() else "#3498db" for f in freq]
        ax.bar(range(1, 46), freq, color=colors, alpha=0.85)
        ax.axhline(freq.mean(), color="orange", linestyle="--", linewidth=1.2, label="평균")
        ax.set_ylabel(f"Set {bs}", fontsize=11, fontweight="bold")
        ax.set_ylim(0, freq.max() * 1.25)
        ax.legend(fontsize=8, loc="upper right")
    axes[-1].set_xlabel("번호 (1~45)", fontsize=11)
    fig.suptitle("공세트별 번호 출현 빈도", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "freq_by_ballset.png"), dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\n[저장] freq_by_ballset.png")
    return all_freqs


def analyze_sum_distribution(df: pd.DataFrame):
    """공세트별 주번호 합계 분포"""
    ball_sets = sorted(df["ball_set"].unique())

    print("\n" + "=" * 60)
    print("[ 공세트별 주번호 합계 분포 ]")
    print("=" * 60)

    fig, ax = plt.subplots(figsize=(10, 5))
    all_sums = {}
    for bs in ball_sets:
        rows = df[df["ball_set"] == bs]
        sums = draw_sums(rows)
        all_sums[bs] = sums
        mean_s = np.mean(sums)
        std_s = np.std(sums)
        print(f"  공세트 {bs}: 평균합={mean_s:.1f}, 표준편차={std_s:.1f}, "
              f"min={min(sums)}, max={max(sums)}")
        ax.hist(sums, bins=20, alpha=0.5, label=f"Set {bs}")

    ax.axvline(138, color="black", linestyle="--", linewidth=1.5, label="이론평균(138)")
    ax.set_xlabel("주번호 합계")
    ax.set_ylabel("빈도")
    ax.set_title("공세트별 주번호 합계 분포")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "sum_distribution.png"), dpi=120)
    plt.close()
    print(f"[저장] sum_distribution.png")

    # Kruskal-Wallis 검정 (공세트 간 합계 차이)
    groups = [all_sums[bs] for bs in ball_sets]
    stat, p = stats.kruskal(*groups)
    print(f"\n  Kruskal-Wallis 검정 (공세트 간 합계 차이): stat={stat:.3f}, p={p:.4f}")
    print(f"  → {'★ 공세트간 합계 분포 유의미하게 다름' if p < 0.05 else '공세트간 합계 분포 차이 없음'}")


def analyze_odd_even(df: pd.DataFrame):
    """공세트별 홀짝 비율"""
    ball_sets = sorted(df["ball_set"].unique())

    print("\n" + "=" * 60)
    print("[ 공세트별 홀짝 비율 ]")
    print("=" * 60)

    labels, odd_ratios, even_ratios = [], [], []
    for bs in ball_sets:
        rows = df[df["ball_set"] == bs]
        odd, even = odd_even_ratio(rows)
        total = odd + even
        print(f"  공세트 {bs}: 홀수={odd}({100*odd/total:.1f}%), 짝수={even}({100*even/total:.1f}%)")
        labels.append(f"Set {bs}")
        odd_ratios.append(100 * odd / total)
        even_ratios.append(100 * even / total)

    x = np.arange(len(ball_sets))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - 0.2, odd_ratios, 0.4, label="홀수", color="#e74c3c", alpha=0.85)
    ax.bar(x + 0.2, even_ratios, 0.4, label="짝수", color="#3498db", alpha=0.85)
    ax.axhline(50, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("비율 (%)")
    ax.set_title("공세트별 홀짝 비율")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "odd_even.png"), dpi=120)
    plt.close()
    print(f"[저장] odd_even.png")


def analyze_range(df: pd.DataFrame):
    """공세트별 구간 분포"""
    ball_sets = sorted(df["ball_set"].unique())
    range_keys = ["1-9", "10-19", "20-29", "30-39", "40-45"]

    print("\n" + "=" * 60)
    print("[ 공세트별 번호 구간 분포 (주번호) ]")
    print("=" * 60)

    fig, axes = plt.subplots(1, len(ball_sets), figsize=(14, 4), sharey=True)
    for ax, bs in zip(axes, ball_sets):
        rows = df[df["ball_set"] == bs]
        rd = range_distribution(rows)
        total = sum(rd.values())
        pcts = [100 * rd[k] / total for k in range_keys]
        print(f"  공세트 {bs}: " + ", ".join(f"{k}={100*rd[k]/total:.1f}%" for k in range_keys))
        ax.bar(range_keys, pcts, color=["#1abc9c","#3498db","#9b59b6","#e67e22","#e74c3c"], alpha=0.85)
        ax.set_title(f"Set {bs}")
        ax.set_ylim(0, 40)
        ax.tick_params(axis="x", rotation=30)
    axes[0].set_ylabel("비율 (%)")
    fig.suptitle("공세트별 번호 구간 분포", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "range_dist.png"), dpi=120)
    plt.close()
    print(f"[저장] range_dist.png")


def analyze_cooccurrence(df: pd.DataFrame):
    """공세트별 번호 공출현 상관 히트맵 (상위 페어 출력)"""
    ball_sets = sorted(df["ball_set"].unique())

    print("\n" + "=" * 60)
    print("[ 공세트별 공출현 상위 번호 쌍 ]")
    print("=" * 60)

    fig, axes = plt.subplots(1, len(ball_sets), figsize=(18, 4))
    for ax, bs in zip(axes, ball_sets):
        rows = df[df["ball_set"] == bs]
        mat = cooccurrence_matrix(rows)

        # 상위 5 쌍
        triu = np.triu(mat, k=1)
        top_pairs = []
        flat = triu.flatten()
        top_idx = np.argsort(flat)[-5:][::-1]
        for idx in top_idx:
            r, c = divmod(idx, 45)
            top_pairs.append((r + 1, c + 1, mat[r, c]))
        print(f"  공세트 {bs} 상위 공출현 쌍: " +
              ", ".join(f"({a},{b})={cnt}" for a, b, cnt in top_pairs))

        im = ax.imshow(mat, cmap="YlOrRd", aspect="auto")
        ax.set_title(f"Set {bs}", fontsize=10)
        ax.set_xlabel("번호")
        if bs == ball_sets[0]:
            ax.set_ylabel("번호")
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("공세트별 공출현 빈도 행렬 (주번호)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "cooccurrence.png"), dpi=120)
    plt.close()
    print(f"[저장] cooccurrence.png")


def analyze_consecutive(df: pd.DataFrame):
    """공세트별 연속번호 포함 비율"""
    ball_sets = sorted(df["ball_set"].unique())

    print("\n" + "=" * 60)
    print("[ 공세트별 연속번호 포함 비율 ]")
    print("=" * 60)

    for bs in ball_sets:
        rows = df[df["ball_set"] == bs]
        count = 0
        for _, row in rows.iterrows():
            nums = sorted([row[c] for c in ["n1","n2","n3","n4","n5","n6"]])
            for i in range(len(nums) - 1):
                if nums[i + 1] - nums[i] == 1:
                    count += 1
                    break
        pct = 100 * count / len(rows)
        print(f"  공세트 {bs}: 연속번호 포함 {count}/{len(rows)} ({pct:.1f}%)")


def compare_ball_sets_kl(all_freqs: dict):
    """공세트 간 KL 발산 비교 (번호 분포 유사도)"""
    ball_sets = sorted(all_freqs.keys())
    eps = 1e-9

    print("\n" + "=" * 60)
    print("[ 공세트 간 KL 발산 (번호 분포 유사도) ]")
    print("  → 값이 클수록 두 공세트의 번호 분포가 다름")
    print("=" * 60)

    for i in range(len(ball_sets)):
        for j in range(i + 1, len(ball_sets)):
            bs_i, bs_j = ball_sets[i], ball_sets[j]
            p = all_freqs[bs_i] / (all_freqs[bs_i].sum() + eps)
            q = all_freqs[bs_j] / (all_freqs[bs_j].sum() + eps)
            kl_ij = float(stats.entropy(p + eps, q + eps))
            kl_ji = float(stats.entropy(q + eps, p + eps))
            print(f"  Set{bs_i}→Set{bs_j}: KL={kl_ij:.4f}, 역방향: {kl_ji:.4f}")


def summary_recommendation(df: pd.DataFrame):
    """분석 종합 및 모델 설계 권고"""
    ball_sets = sorted(df["ball_set"].unique())
    print("\n" + "=" * 60)
    print("[ 종합 분석 결과 및 모델 설계 권고 ]")
    print("=" * 60)

    for bs in ball_sets:
        rows = df[df["ball_set"] == bs]
        n_draws = len(rows)
        freq = main_freq_array(rows)
        chi2, p = stats.chisquare(freq, f_exp=[freq.sum() / 45.0] * 45)
        sums = draw_sums(rows)
        odd, even = odd_even_ratio(rows)
        total_oe = odd + even
        print(f"\n  공세트 {bs} | 샘플 {n_draws}개")
        print(f"    - 빈도 균일성: chi2={chi2:.2f}, p={p:.4f} "
              f"{'(편차 有)' if p < 0.05 else '(균일)'}")
        print(f"    - 합계: μ={np.mean(sums):.1f}, σ={np.std(sums):.1f}")
        print(f"    - 홀짝: {100*odd/total_oe:.1f}% / {100*even/total_oe:.1f}%")

    print("\n[ 모델 설계 권고 ]")
    print("  · 공세트별 번호 분포가 균일에서 유의미하게 벗어나는 세트가 존재할 경우")
    print("    → 공세트를 조건으로 주는 조건부 예측 모델 필요")
    print("  · 샘플 수가 공세트당 80~160개 수준으로 적음")
    print("    → Shared LSTM Backbone + Ball-Set Conditioning 구조 권장")
    print("    → 세트별 독립 모델은 데이터 부족으로 과적합 위험")
    print("  · 역사 시퀀스를 LSTM으로 인코딩하고,")
    print("    다음 회차의 공세트 번호를 임베딩으로 주입하여 예측")


# ── 진입점 ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_data(DATA_PATH)
    print(f"전체 데이터: {len(df)}행, 회차 {df['round'].min()}~{df['round'].max()}")
    print(f"공세트 분포:\n{df['ball_set'].value_counts().sort_index().to_string()}\n")

    all_freqs = analyze_frequency(df)
    analyze_sum_distribution(df)
    analyze_odd_even(df)
    analyze_range(df)
    analyze_cooccurrence(df)
    analyze_consecutive(df)
    compare_ball_sets_kl(all_freqs)
    summary_recommendation(df)

    print(f"\n분석 완료. 이미지 저장 위치: {OUTPUT_DIR}")
