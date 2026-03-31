"""
예측 유의미성 검증
===================
세 가지 접근:

1. Walk-forward Backtesting
   - 시간 순서를 지키며 과거 회차를 예측, 실제 당첨 번호와 비교
   - 데이터 누수(look-ahead bias) 없음

2. Baseline 비교 (3종)
   - Random    : 1~45에서 6개 무작위 선택
   - Frequency : 해당 공세트 과거 빈도 상위 6개 고정 선택
   - Model     : BallSetLSTM 예측 상위 6개

3. 통계적 유의성 검정
   - 이항 검정(Binomial Test): "랜덤보다 더 많이 맞혔는가?"
   - 기대 적중 개수 (랜덤): 6 × (6/45) ≈ 0.800 개/회차
   - Calibration: 모델 확률 → 실제 출현율 일치도
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
from scipy import stats
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from m02_model import BallSetLSTM

# ── 경로 ───────────────────────────────────────────────────────────────────────
DATA_PATH   = os.path.join(os.path.dirname(__file__), "../data/history_from_cafe.csv")
CKPT_PATH   = os.path.join(os.path.dirname(__file__), "best_m02.pth")
OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), "eval_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_BALLS   = 45
NUM_SETS    = 5
RANDOM_SEED = 42

# 랜덤 선택의 이론적 기대 적중 수 (주번호 6개 중 6개)
# P(맞힘) = 6/45,  6번 시도 → E = 6 × (6/45)
RANDOM_EXPECTED_HITS = 6 * (6 / NUM_BALLS)   # ≈ 0.800


# ── 데이터 로드 ────────────────────────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, header=None,
                     names=["ball_set", "round", "draw_date", "n1", "n2", "n3", "n4", "n5", "n6", "bonus"])
    return df.sort_values(["round", "ball_set"]).reset_index(drop=True)


def numbers_of_row(row) -> set:
    """주번호 6개만 반환 (보너스 제외)"""
    return {int(row[c]) for c in ["n1", "n2", "n3", "n4", "n5", "n6"]}


def onehot(nums, n=NUM_BALLS) -> np.ndarray:
    arr = np.zeros(n, dtype=np.float32)
    for v in nums:
        if 1 <= v <= n:
            arr[v - 1] = 1.0
    return arr


def row_onehot(row) -> np.ndarray:
    """주번호 + 보너스 원-핫"""
    nums = [int(row[c]) for c in ["n1","n2","n3","n4","n5","n6","bonus"]]
    return onehot(nums)


# ── 모델 로드 ──────────────────────────────────────────────────────────────────
def load_model(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    cfg  = ckpt["config"]
    model = BallSetLSTM(
        num_balls     = cfg["num_balls"],
        num_ball_sets = cfg["num_ball_sets"],
        emb_dim       = cfg["emb_dim"],
        hidden_size   = cfg["hidden_size"],
        num_layers    = cfg["num_layers"],
        dropout       = 0.0,   # 평가 시 dropout 비활성
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, cfg


# ── Baseline 예측기 ────────────────────────────────────────────────────────────
def predict_random(rng: np.random.Generator) -> set:
    return set(rng.choice(NUM_BALLS, size=6, replace=False) + 1)


def predict_frequency(history: pd.DataFrame, target_ball_set: int) -> set:
    """해당 공세트의 과거 주번호 빈도 상위 6개"""
    subset = history[history["ball_set"] == target_ball_set]
    if len(subset) == 0:
        # 데이터 없으면 전체 빈도 사용
        subset = history
    freq = np.zeros(NUM_BALLS, dtype=int)
    for col in ["n1","n2","n3","n4","n5","n6"]:
        for v in subset[col]:
            if 1 <= v <= NUM_BALLS:
                freq[v - 1] += 1
    top6 = set(np.argsort(freq)[-6:] + 1)
    return top6


def predict_model(model, history: pd.DataFrame, target_ball_set: int,
                  win_size: int, device: str) -> set:
    """모델 예측 상위 6개"""
    rows = history.tail(win_size).reset_index(drop=True)
    pad  = win_size - len(rows)

    oh_list = []
    bs_list = []
    for _, row in rows.iterrows():
        oh_list.append(row_onehot(row))
        bs_list.append(int(row["ball_set"]) - 1)

    # 데이터 부족 시 앞에 0 패딩
    if pad > 0:
        oh_list  = [np.zeros(NUM_BALLS, dtype=np.float32)] * pad + oh_list
        bs_list  = [0] * pad + bs_list

    x_oh = torch.from_numpy(np.array(oh_list)).unsqueeze(0).to(device)   # (1,W,45)
    x_bs = torch.tensor([bs_list], dtype=torch.long).to(device)           # (1,W)
    t_bs = torch.tensor([target_ball_set - 1], dtype=torch.long).to(device)

    with torch.no_grad():
        prob = model(x_oh, x_bs, t_bs)[0].cpu().numpy()

    top6 = set(np.argsort(prob)[-6:] + 1)
    return top6


# ── Walk-forward Backtesting ───────────────────────────────────────────────────
def walk_forward_backtest(df: pd.DataFrame, model, cfg: dict,
                          device: str, start_idx: int = None):
    """
    start_idx 이후 각 행을 순서대로 예측하고 실제와 비교.
    start_idx 기본값: 전체의 20% 지점 (최소 win_size 이후)
    """
    win_size = cfg["win_size"]
    rng = np.random.default_rng(RANDOM_SEED)

    if start_idx is None:
        start_idx = max(win_size, int(len(df) * 0.2))

    print(f"\n{'='*60}")
    print(f"Walk-forward Backtesting")
    print(f"  평가 구간: index {start_idx} ~ {len(df)-1}  ({len(df)-start_idx}회차)")
    print(f"  랜덤 기대 적중: {RANDOM_EXPECTED_HITS:.3f}개/회차")
    print(f"{'='*60}")

    records = []  # (ball_set, model_hits, freq_hits, rand_hits, model_prob)

    # 공세트별 확률 누적 (Calibration용)
    prob_buckets = defaultdict(list)   # bucket_center → [(pred_prob, actual)]

    for i in range(start_idx, len(df)):
        row        = df.iloc[i]
        history    = df.iloc[:i]
        target_bs  = int(row["ball_set"])
        actual     = numbers_of_row(row)

        pred_model = predict_model(model, history, target_bs, win_size, device)
        pred_freq  = predict_frequency(history, target_bs)
        pred_rand  = predict_random(rng)

        hits_model = len(pred_model & actual)
        hits_freq  = len(pred_freq  & actual)
        hits_rand  = len(pred_rand  & actual)

        # Calibration 데이터 수집
        rows_tail = history.tail(win_size).reset_index(drop=True)
        pad = win_size - len(rows_tail)
        oh_list = [row_onehot(r) for _, r in rows_tail.iterrows()]
        bs_list = [int(r["ball_set"]) - 1 for _, r in rows_tail.iterrows()]
        if pad > 0:
            oh_list = [np.zeros(NUM_BALLS, dtype=np.float32)] * pad + oh_list
            bs_list = [0] * pad + bs_list
        x_oh = torch.from_numpy(np.array(oh_list)).unsqueeze(0).to(device)
        x_bs = torch.tensor([bs_list], dtype=torch.long).to(device)
        t_bs = torch.tensor([target_bs - 1], dtype=torch.long).to(device)
        with torch.no_grad():
            prob_all = model(x_oh, x_bs, t_bs)[0].cpu().numpy()

        actual_oh = onehot(actual)
        for num_idx in range(NUM_BALLS):
            bucket = round(float(prob_all[num_idx]) * 10) / 10  # 0.0~1.0, step 0.1
            prob_buckets[bucket].append((float(prob_all[num_idx]), int(actual_oh[num_idx])))

        records.append({
            "ball_set":    target_bs,
            "model_hits":  hits_model,
            "freq_hits":   hits_freq,
            "rand_hits":   hits_rand,
        })

    rec = pd.DataFrame(records)

    # ── 전체 통계 ────────────────────────────────────────────────────────────
    n      = len(rec)
    m_mean = rec["model_hits"].mean()
    f_mean = rec["freq_hits"].mean()
    r_mean = rec["rand_hits"].mean()

    print(f"\n[ 전체 평균 적중 수 (주번호 6개 기준) ]")
    print(f"  모델(BallSetLSTM): {m_mean:.4f} 개/회차")
    print(f"  빈도 기반 Baseline: {f_mean:.4f} 개/회차")
    print(f"  랜덤 Baseline:      {r_mean:.4f} 개/회차")
    print(f"  이론 기대값(랜덤):   {RANDOM_EXPECTED_HITS:.4f} 개/회차")

    # ── 이항 검정 ────────────────────────────────────────────────────────────
    # 귀무가설: 모델 적중이 랜덤과 다르지 않다 (p_success = 6/45 per number)
    # 각 회차에서 6번의 독립 시도, p = 6/45
    total_trials = n * 6
    p_random     = 6 / NUM_BALLS

    for label, col in [("모델", "model_hits"), ("빈도기반", "freq_hits")]:
        total_hits = int(rec[col].sum())
        binom_res  = stats.binomtest(total_hits, total_trials, p_random, alternative="greater")
        print(f"\n[ 이항 검정: {label} vs 랜덤 ]")
        print(f"  총 시도: {total_trials}번  총 적중: {total_hits}번")
        print(f"  관찰 적중률: {total_hits/total_trials:.4f}  기대(랜덤): {p_random:.4f}")
        print(f"  p-value: {binom_res.pvalue:.6f}")
        if binom_res.pvalue < 0.05:
            print(f"  ★ 랜덤보다 유의미하게 더 맞힘 (p<0.05)")
        elif binom_res.pvalue < 0.10:
            print(f"  ▲ 약한 경향성 있음 (p<0.10)")
        else:
            print(f"  ✗ 랜덤과 차이 없음")

    # ── 공세트별 분석 ────────────────────────────────────────────────────────
    print(f"\n[ 공세트별 평균 적중 수 ]")
    print(f"  {'Set':>4} {'n':>5} {'Model':>8} {'Freq':>8} {'Rand':>8}")
    for bs in sorted(rec["ball_set"].unique()):
        sub = rec[rec["ball_set"] == bs]
        print(f"  {bs:>4} {len(sub):>5} "
              f"{sub['model_hits'].mean():>8.4f} "
              f"{sub['freq_hits'].mean():>8.4f} "
              f"{sub['rand_hits'].mean():>8.4f}")

    # ── 적중 분포 히스토그램 ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    bins = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
    for ax, col, title, color in [
        (axes[0], "model_hits", "Model", "#e74c3c"),
        (axes[1], "freq_hits",  "Frequency Baseline", "#3498db"),
        (axes[2], "rand_hits",  "Random Baseline", "#95a5a6"),
    ]:
        ax.hist(rec[col], bins=bins, color=color, alpha=0.8, edgecolor="white")
        ax.axvline(rec[col].mean(), color="black", linestyle="--",
                   linewidth=1.5, label=f"mean={rec[col].mean():.3f}")
        ax.axvline(RANDOM_EXPECTED_HITS, color="orange", linestyle=":",
                   linewidth=1.5, label=f"theory={RANDOM_EXPECTED_HITS:.3f}")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Hits per draw")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Count")
    fig.suptitle("Hit Distribution Comparison", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "hit_distribution.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"\n[저장] hit_distribution.png")

    # ── Calibration Plot ─────────────────────────────────────────────────────
    bucket_centers = sorted(prob_buckets.keys())
    mean_preds, mean_actuals, sizes = [], [], []
    for bc in bucket_centers:
        items = prob_buckets[bc]
        if len(items) < 5:
            continue
        mean_preds.append(np.mean([x[0] for x in items]))
        mean_actuals.append(np.mean([x[1] for x in items]))
        sizes.append(len(items))

    if mean_preds:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
        sc = ax.scatter(mean_preds, mean_actuals, c=sizes, cmap="Blues",
                        s=80, zorder=5, edgecolors="grey", linewidths=0.5)
        plt.colorbar(sc, ax=ax, label="Sample count")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Actual frequency")
        ax.set_title("Model Calibration\n(closer to diagonal = better)", fontsize=11)
        ax.legend()
        plt.tight_layout()
        path = os.path.join(OUTPUT_DIR, "calibration.png")
        plt.savefig(path, dpi=120)
        plt.close()
        print(f"[저장] calibration.png")

    return rec


# ── 해석 가이드 출력 ───────────────────────────────────────────────────────────
def print_interpretation_guide():
    print(f"\n{'='*60}")
    print("[ 결과 해석 가이드 ]")
    print(f"{'='*60}")
    print("""
  1. 평균 적중 수
     - 랜덤 이론값: 0.800 개/회차
     - 모델 > 빈도기반 > 랜덤  순서면 모델이 의미있음
     - 0.800 을 크게 넘어도 여전히 로또 당첨은 어려움

  2. 이항 검정 p-value
     - p < 0.05 : 랜덤보다 유의미하게 더 맞힘
     - p < 0.10 : 약한 경향성 (데이터 부족 가능성)
     - p >= 0.10: 랜덤과 차이 없음 → 모델 재검토 필요

  3. Calibration Plot
     - 대각선에 가까울수록 예측 확률이 실제 출현율과 일치
     - 위쪽으로 치우치면 과소예측, 아래쪽이면 과대예측

  4. 공세트별 차이
     - 특정 공세트에서 모델이 더 잘 맞힌다면
       → 해당 공세트의 물리적 편향이 실재할 가능성

  5. 현실적 기대치
     - 이론적으로 로또는 완전 랜덤 설계
     - 물리적 질량 차이로 인한 편향이 있어도 매우 미미
     - 모델 우위가 통계적으로 유의미하더라도
       당첨 기대값(EV)은 여전히 음수임을 인식할 것
""")


# ── 진입점 ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",      type=str, default=CKPT_PATH)
    parser.add_argument("--data_path", type=str, default=DATA_PATH)
    parser.add_argument("--start_pct", type=float, default=0.2,
                        help="전체 데이터 중 평가 시작 비율 (기본 0.2 = 앞 20%% 이후)")
    args = parser.parse_args()

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    df    = load_data(args.data_path)
    model, cfg = load_model(args.ckpt, device)

    start_idx = max(cfg["win_size"], int(len(df) * args.start_pct))
    rec = walk_forward_backtest(df, model, cfg, device, start_idx=start_idx)

    print_interpretation_guide()
    print(f"\n평가 완료. 결과 저장: {OUTPUT_DIR}")
