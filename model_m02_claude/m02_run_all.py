"""
전체 공세트(1~5) 번호 추출 통합 실행
======================================
모든 공세트에 대해:
  - 상위 15개 번호 풀
  - 전략 1: 확률 상위 조합
  - 전략 2: 확률 가중 랜덤 조합
  - 전략 3: 구간 균형 조합
  - cold 번호 포함 조합 (오래 안 나온 번호 반영)

사용법:
  python m02_run_all.py
  python m02_run_all.py --top_n 12 --top_k 5 --sample_m 10
"""

import argparse
import os
import itertools
import numpy as np
import pandas as pd
import torch
from m02_model import BallSetLSTM

DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/history_from_cafe.csv")
CKPT_PATH = os.path.join(os.path.dirname(__file__), "best_m02.pth")
NUM_BALLS = 45
POS_COLS  = ["n1","n2","n3","n4","n5","n6"]


# ── 공통 함수 ──────────────────────────────────────────────────────────────────
def load_model(ckpt_path, device):
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=True)
    cfg   = ckpt["config"]
    model = BallSetLSTM(
        num_balls=cfg["num_balls"], num_ball_sets=cfg["num_ball_sets"],
        emb_dim=cfg["emb_dim"], hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"], dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, cfg


def get_prob(model, df, ball_set, win_size, device):
    rows = df.tail(win_size).reset_index(drop=True)
    pad  = win_size - len(rows)
    oh_list, bs_list = [], []
    for _, row in rows.iterrows():
        arr = np.zeros(NUM_BALLS, dtype=np.float32)
        for c in ["n1","n2","n3","n4","n5","n6","bonus"]:
            v = int(row[c])
            if 1 <= v <= NUM_BALLS:
                arr[v-1] = 1.0
        oh_list.append(arr)
        bs_list.append(int(row["ball_set"]) - 1)
    if pad > 0:
        oh_list = [np.zeros(NUM_BALLS, dtype=np.float32)] * pad + oh_list
        bs_list = [0] * pad + bs_list
    x_oh = torch.from_numpy(np.array(oh_list)).unsqueeze(0).to(device)
    x_bs = torch.tensor([bs_list], dtype=torch.long).to(device)
    t_bs = torch.tensor([ball_set - 1], dtype=torch.long).to(device)
    with torch.no_grad():
        prob = model(x_oh, x_bs, t_bs)[0].cpu().numpy()
    return prob


def get_cold_numbers(df, ball_set, top_cold=5):
    """해당 공세트에서 가장 오래 안 나온 번호"""
    sub = df[df["ball_set"] == ball_set].reset_index(drop=True)
    last_seen = {}
    for i, row in sub.iterrows():
        for c in POS_COLS:
            last_seen[int(row[c])] = i
    total = len(sub)
    # 등장한 적 없는 번호는 gap = total
    all_nums = {n: last_seen.get(n, -1) for n in range(1, NUM_BALLS + 1)}
    cold = sorted(all_nums.items(), key=lambda x: x[1])[:top_cold]
    return [n for n, _ in cold]


def score_combo(nums, prob):
    return sum(prob[n - 1] for n in nums)


def build_scored(top_nums, prob):
    combos = list(itertools.combinations(top_nums, 6))
    scored = sorted([(c, score_combo(c, prob)) for c in combos],
                    key=lambda x: x[1], reverse=True)
    return scored


# ── 전략별 출력 ────────────────────────────────────────────────────────────────
def print_strategy1(scored, top_k):
    print(f"\n  [전략 1] 확률 상위 {top_k}개 조합")
    print(f"  {'순위':>4}  {'조합':<32}  스코어")
    print(f"  {'-'*52}")
    for rank, (combo, sc) in enumerate(scored[:top_k], 1):
        print(f"  {rank:>4}  {str(list(combo)):<32}  {sc*100:.2f}%")


def print_strategy2(scored, sample_m, seed):
    rng     = np.random.default_rng(seed)
    weights = np.array([sc for _, sc in scored])
    weights = weights / weights.sum()
    chosen  = sorted(rng.choice(len(scored), size=min(sample_m, len(scored)),
                                replace=False, p=weights))
    print(f"\n  [전략 2] 확률 가중 랜덤 {sample_m}개 조합")
    print(f"  {'순위':>4}  {'조합':<32}  스코어")
    print(f"  {'-'*52}")
    for rank, idx in enumerate(chosen, 1):
        combo, sc = scored[idx]
        print(f"  {rank:>4}  {str(list(combo)):<32}  {sc*100:.2f}%")


def print_strategy3(scored, top_k):
    balanced = [(c, sc) for c, sc in scored
                if sum(1 for n in c if n <= 15) >= 1
                and sum(1 for n in c if 16 <= n <= 30) >= 1
                and sum(1 for n in c if n >= 31) >= 1]
    print(f"\n  [전략 3] 구간 균형(저-중-고 각 1개 이상) 상위 {top_k}개")
    print(f"  {'순위':>4}  {'조합':<32}  스코어  저-중-고")
    print(f"  {'-'*58}")
    for rank, (combo, sc) in enumerate(balanced[:top_k], 1):
        lo  = sum(1 for n in combo if n <= 15)
        mid = sum(1 for n in combo if 16 <= n <= 30)
        hi  = sum(1 for n in combo if n >= 31)
        print(f"  {rank:>4}  {str(list(combo)):<32}  {sc*100:.2f}%  {lo}-{mid}-{hi}")


def print_strategy4(scored, cold_nums, top_k):
    """cold 번호 중 최소 1개 포함 조합"""
    cold_set  = set(cold_nums)
    with_cold = [(c, sc) for c, sc in scored
                 if len(set(c) & cold_set) >= 1]
    print(f"\n  [전략 4] Cold 번호({cold_nums}) 중 1개 이상 포함 상위 {top_k}개")
    print(f"  {'순위':>4}  {'조합':<32}  스코어  cold포함")
    print(f"  {'-'*58}")
    for rank, (combo, sc) in enumerate(with_cold[:top_k], 1):
        included = sorted(set(combo) & cold_set)
        print(f"  {rank:>4}  {str(list(combo)):<32}  {sc*100:.2f}%  {included}")


# ── 메인 ──────────────────────────────────────────────────────────────────────
def main(top_n, top_k, sample_m, seed, ball_set=None):
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")

    df = pd.read_csv(DATA_PATH, header=None,
                     names=["ball_set","round","draw_date","n1","n2","n3","n4","n5","n6","bonus"])
    df = df.sort_values(["round","ball_set"]).reset_index(drop=True)

    model, cfg = load_model(CKPT_PATH, device)
    win_size   = cfg["win_size"]

    print(f"\n{'#'*70}")
    print(f"  로또 번호 추출  |  상위 {top_n}개 풀  |  C({top_n},6)={len(list(itertools.combinations(range(top_n),6)))}가지 조합")
    print(f"{'#'*70}")

    results = {}

    ball_sets = [ball_set] if ball_set else list(range(1, 6))

    for bs in ball_sets:
        print(f"\n{'='*70}")
        print(f"  공세트 {bs}")
        print(f"{'='*70}")

        prob      = get_prob(model, df, bs, win_size, device)
        top_idx   = np.argsort(prob)[::-1][:top_n]
        top_nums  = sorted((top_idx + 1).tolist())
        cold_nums = get_cold_numbers(df, bs, top_cold=3)

        # 번호 풀 출력
        print(f"\n  번호 풀 (상위 {top_n}개): {top_nums}")
        print(f"  확률: " + "  ".join(f"{n}={prob[n-1]*100:.1f}%" for n in top_nums))
        print(f"  Cold 번호 (오래 미등장): {cold_nums}")

        scored = build_scored(top_nums, prob)

        print_strategy1(scored, top_k)
        print_strategy2(scored, sample_m, seed)
        print_strategy3(scored, top_k)
        print_strategy4(scored, cold_nums, top_k)

        results[bs] = {
            "pool": top_nums,
            "cold": cold_nums,
            "top1": [list(c) for c, _ in scored[:top_k]],
        }

    # ── 요약 테이블 ──────────────────────────────────────────────────────────
    print(f"\n{'#'*70}")
    print(f"  전체 요약  |  공세트별 전략 1 최상위 번호")
    print(f"{'#'*70}")
    print(f"  {'Set':>4}  {'번호 풀':40}  {'Cold':15}  최고 조합")
    print(f"  {'-'*90}")
    for bs in ball_sets:
        r = results[bs]
        print(f"  {bs:>4}  {str(r['pool']):<40}  {str(r['cold']):<15}  {r['top1'][0]}")

    print(f"\n{'#'*70}")
    print(f"  구매 가이드")
    print(f"  전략 1: 가장 확률 높은 조합 집중 구매")
    print(f"  전략 2: 다양성 확보 (가중 랜덤)")
    print(f"  전략 3: 번호 구간 분산")
    print(f"  전략 4: 오래 안 나온 번호 포함")
    print(f"{'#'*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top_n",    type=int, default=15,  help="번호 풀 크기 (기본 15)")
    parser.add_argument("--top_k",    type=int, default=5,   help="각 전략 출력 개수 (기본 5)")
    parser.add_argument("--sample_m", type=int, default=10,  help="전략2 샘플 수 (기본 10)")
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--ball_set", type=int, default=None, help="특정 공세트만 (생략 시 1~5 전체)")
    args = parser.parse_args()
    main(args.top_n, args.top_k, args.sample_m, args.seed, args.ball_set)
