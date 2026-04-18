"""
전체 공세트(1~5) 번호 추출 통합 실행

모든 공세트에 대해:
  - 상위 pool(기본 15)개 번호 풀
  - 전략 1: 사후확률 상위 조합 (log-prob 합 최대)
  - 전략 2: 사후확률 가중 랜덤 샘플링 (분산 확보)
  - 전략 3: 구간 균형 (1-9, 10-19, 20-29, 30-39, 40-45)
  - 전략 4: Cold 번호 포함 (하위 N개 중 1개를 강제 포함)

사용법:
  python m03_run_all.py
  python m03_run_all.py --pool 15 --top_k 3
"""

import os
import json
import random
import argparse
import itertools
from datetime import datetime

import numpy as np

from m03_model import BayesianFrequencyModel, NUM_BALLS, NUM_SETS

CKPT_PATH   = os.path.join(os.path.dirname(__file__), "best_m03.npz")
RESULT_PATH = os.path.join(os.path.dirname(__file__), "pipeline_result.json")

NUM_MAIN = 6


def strategy1_top_prob(probs: np.ndarray, pool: int, top_k: int):
    """확률 상위 조합 (log-prob 합 최대)"""
    top_idx = np.argsort(probs)[::-1][:pool]
    top_nums = sorted((top_idx + 1).tolist())
    log_p = np.log(probs)
    scored = [(sum(log_p[n-1] for n in c), list(c))
              for c in itertools.combinations(top_nums, NUM_MAIN)]
    scored.sort(reverse=True)
    return [combo for _, combo in scored[:top_k]]


def strategy2_weighted_random(probs: np.ndarray, pool: int, top_k: int, seed: int = 42):
    """가중 랜덤 샘플링: 상위 pool 에서 확률 비례 추출"""
    rng = np.random.default_rng(seed)
    top_idx = np.argsort(probs)[::-1][:pool]
    p = probs[top_idx]
    p = p / p.sum()
    results = []
    for _ in range(top_k):
        chosen = rng.choice(top_idx, size=NUM_MAIN, replace=False, p=p)
        results.append(sorted((chosen + 1).tolist()))
    return results


def strategy3_balanced(probs: np.ndarray, pool: int, top_k: int):
    """구간 균형: 1-9, 10-19, 20-29, 30-39, 40-45 에서 고루 선택"""
    buckets = [(1, 9), (10, 19), (20, 29), (30, 39), (40, 45)]
    top_idx = np.argsort(probs)[::-1][:pool]
    top_nums = sorted((top_idx + 1).tolist())

    def bucket_of(n: int) -> int:
        for i, (a, b) in enumerate(buckets):
            if a <= n <= b:
                return i
        return -1

    log_p = np.log(probs)
    candidates = []
    for combo in itertools.combinations(top_nums, NUM_MAIN):
        buckets_seen = {bucket_of(n) for n in combo}
        if len(buckets_seen) >= 4:  # 구간 4개 이상 포함
            candidates.append((sum(log_p[n-1] for n in combo), list(combo)))
    if not candidates:
        return strategy1_top_prob(probs, pool, top_k)
    candidates.sort(reverse=True)
    return [c for _, c in candidates[:top_k]]


def strategy4_cold_included(probs: np.ndarray, pool: int, top_k: int, cold_pool: int = 5):
    """Cold 번호(하위 cold_pool 개 중 1개) 포함 + 상위 pool 에서 5개"""
    asc = np.argsort(probs)
    cold_nums = (asc[:cold_pool] + 1).tolist()
    hot_nums = sorted((asc[::-1][:pool] + 1).tolist())
    log_p = np.log(probs)

    scored = []
    for cold in cold_nums:
        for combo in itertools.combinations([n for n in hot_nums if n != cold], NUM_MAIN - 1):
            full = sorted([cold] + list(combo))
            score = sum(log_p[n-1] for n in full)
            scored.append((score, full))
    scored.sort(reverse=True)
    seen, out = set(), []
    for _, c in scored:
        key = tuple(c)
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
        if len(out) >= top_k:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=int, default=15, help="번호 풀 크기 (기본 15)")
    ap.add_argument("--top_k", type=int, default=3, help="전략별 상위 K개 출력")
    ap.add_argument("--global", dest="use_global", action="store_true",
                    help="공세트 무시하고 글로벌 확률 사용")
    args = ap.parse_args()

    if not os.path.exists(CKPT_PATH):
        print(f"[ERROR] 모델 파일 없음: {CKPT_PATH}")
        print("        먼저 m03_train.py 를 실행하세요.")
        raise SystemExit(1)

    model = BayesianFrequencyModel.load(CKPT_PATH)
    print(f"모델 로드: {CKPT_PATH}")
    print(f"  회차 범위: {model.round_range[0]}~{model.round_range[1]}")
    print(f"  총 추첨: {model.num_draws_global}")
    print(f"  공세트별: {model.num_draws_per_set}")
    print(f"  α (Laplace): {model.alpha}")

    result = {
        "predicted_at": datetime.now().isoformat(),
        "config": {"pool": args.pool, "top_k": args.top_k, "global": args.use_global},
        "tickets": {},
    }

    targets = [None] if args.use_global else list(range(1, NUM_SETS + 1))
    for bs in targets:
        probs = model.posterior(bs)
        label = "global" if bs is None else f"ball_set_{bs}"

        print(f"\n{'='*60}")
        print(f"{label}")
        print(f"{'='*60}")
        top_idx = np.argsort(probs)[::-1][:args.pool]
        pool_str = ", ".join(f"{int(n+1)}({probs[n]*100:.2f}%)" for n in top_idx)
        print(f"  상위 {args.pool}: {pool_str}")

        s1 = strategy1_top_prob(probs, args.pool, args.top_k)
        s2 = strategy2_weighted_random(probs, args.pool, args.top_k, seed=42 + (bs or 0))
        s3 = strategy3_balanced(probs, args.pool, args.top_k)
        s4 = strategy4_cold_included(probs, args.pool, args.top_k)

        for name, combos in [("전략1 확률상위", s1),
                             ("전략2 가중랜덤", s2),
                             ("전략3 구간균형", s3),
                             ("전략4 cold포함", s4)]:
            print(f"\n  [{name}]")
            for i, c in enumerate(combos, 1):
                print(f"    {i}. {c}")

        result["tickets"][label] = {
            "strategy_1_top_prob": s1,
            "strategy_2_weighted_random": s2,
            "strategy_3_balanced": s3,
            "strategy_4_cold_included": s4,
        }

    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장: {RESULT_PATH}")


if __name__ == "__main__":
    main()
