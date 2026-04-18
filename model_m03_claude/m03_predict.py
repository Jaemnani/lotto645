"""
공세트별 1장씩 최종 추천 번호 생성
==================================
전략:
  1. "일정 수준 이상 자주 나오는" 번호만 hot 풀로 선정
     - threshold 모드: 기대값 × ratio 이상 (기본 1.0 = 평균 이상)
     - top 모드     : 상위 N개 (기본 15)
  2. hot 풀 내에서 사후확률 가중 샘플링 → 6개 번호 추출
     - 자주 나오는 번호에 더 높은 가중치
     - 완전 결정적(top-1)이 아닌 확률적 다양성 확보

사용법:
  python m03_predict.py
  python m03_predict.py --pool-mode threshold --ratio 1.0
  python m03_predict.py --pool-mode top --top 15
  python m03_predict.py --mode deterministic   # 상위 조합 고정
  python m03_predict.py --samples 5            # 공세트별 5장 다양화
  python m03_predict.py --seed 42
"""

import os
import json
import argparse
import itertools
from datetime import datetime

import numpy as np
import pytz

from m03_model import BayesianFrequencyModel, NUM_BALLS, NUM_SETS

CKPT_PATH   = os.path.join(os.path.dirname(__file__), "best_m03.npz")
RESULT_PATH = os.path.join(os.path.dirname(__file__), "prediction.json")

NUM_MAIN = 6


def build_pool(probs: np.ndarray, mode: str, top: int, ratio: float,
               counts: np.ndarray) -> np.ndarray:
    """hot 풀 인덱스 반환 (1~45 의 번호 인덱스)"""
    if mode == "top":
        return np.argsort(probs)[::-1][:top]
    elif mode == "threshold":
        expected = counts.sum() / NUM_BALLS
        mask = counts >= (expected * ratio)
        pool = np.where(mask)[0]
        if len(pool) < NUM_MAIN:
            # 임계값이 너무 높아서 6개 미만이면 상위 NUM_MAIN 로 폴백
            pool = np.argsort(probs)[::-1][:NUM_MAIN]
        return pool
    else:
        raise ValueError(f"unknown pool mode: {mode}")


def sample_combo(probs: np.ndarray, pool: np.ndarray,
                 rng: np.random.Generator) -> list[int]:
    """pool 에서 확률 가중 샘플링으로 6개 번호 추출"""
    w = probs[pool]
    w = w / w.sum()
    chosen = rng.choice(pool, size=NUM_MAIN, replace=False, p=w)
    return sorted((chosen + 1).tolist())


def best_combo(probs: np.ndarray, pool: np.ndarray) -> list[int]:
    """pool 내 log-prob 합 최대 조합 (결정적)"""
    log_p = np.log(probs)
    best, best_score = None, -np.inf
    for c in itertools.combinations((pool + 1).tolist(), NUM_MAIN):
        s = sum(log_p[n-1] for n in c)
        if s > best_score:
            best_score = s
            best = sorted(list(c))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-mode", choices=["threshold", "top"], default="threshold",
                    help="풀 선정 방식 (기본 threshold)")
    ap.add_argument("--ratio", type=float, default=1.0,
                    help="threshold 모드: 기대값 대비 비율 (기본 1.0 = 평균 이상)")
    ap.add_argument("--top", type=int, default=15,
                    help="top 모드: 상위 N개 (기본 15)")
    ap.add_argument("--mode", choices=["sample", "deterministic"], default="sample",
                    help="샘플링 방식 (기본 sample = 가중랜덤)")
    ap.add_argument("--samples", type=int, default=1,
                    help="공세트별 생성 티켓 수 (기본 1)")
    ap.add_argument("--seed", type=int, default=None,
                    help="샘플링 시드 (기본: 시간 기반)")
    args = ap.parse_args()

    if not os.path.exists(CKPT_PATH):
        print(f"[ERROR] 모델 없음: {CKPT_PATH}  (먼저 m03_train.py 실행)")
        raise SystemExit(1)

    model = BayesianFrequencyModel.load(CKPT_PATH)
    kst = pytz.timezone("Asia/Seoul")
    ts = datetime.now(kst)
    rng = np.random.default_rng(args.seed)

    print(f"{'='*62}")
    print(f"m03 베이지안 빈도 모델 — 공세트별 추천 번호")
    print(f"{'='*62}")
    print(f"  학습 회차: {model.round_range[0]}~{model.round_range[1]}")
    print(f"  총 추첨  : {model.num_draws_global}회")
    print(f"  α        : {model.alpha}")
    print(f"  pool     : {args.pool_mode} "
          f"(ratio={args.ratio})" if args.pool_mode == "threshold"
          else f"  pool     : {args.pool_mode} (top={args.top})")
    print(f"  mode     : {args.mode}   samples/set: {args.samples}")
    print(f"  seed     : {args.seed}")
    print(f"  생성시각 : {ts.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print()

    tickets: dict[str, list] = {}
    for bs in range(1, NUM_SETS + 1):
        probs = model.posterior(bs)
        counts = model.counts_per_set[bs]
        pool = build_pool(probs, args.pool_mode, args.top, args.ratio, counts)

        pool_info = ", ".join(
            f"{n+1}({int(counts[n])}회)"
            for n in sorted(pool, key=lambda i: -probs[i])
        )
        expected = counts.sum() / NUM_BALLS
        print(f"  공세트 {bs}  (기대값 {expected:.1f}회 | pool 크기 {len(pool)})")
        print(f"    hot pool: {pool_info}")

        combos = []
        for i in range(args.samples):
            if args.mode == "deterministic" and i == 0:
                combo = best_combo(probs, pool)
            else:
                combo = sample_combo(probs, pool, rng)
            combos.append(combo)
            freqs = [int(counts[n-1]) for n in combo]
            tag = "(top)" if args.mode == "deterministic" and i == 0 else ""
            print(f"    티켓 {i+1}{tag}: {combo}   freq {freqs}")
        print()

        tickets[f"ball_set_{bs}"] = combos[0] if args.samples == 1 else combos

    result = {
        "predicted_at": ts.isoformat(),
        "model": "m03_bayesian_frequency",
        "round_range": list(model.round_range),
        "config": {
            "pool_mode": args.pool_mode,
            "ratio": args.ratio,
            "top": args.top,
            "mode": args.mode,
            "samples": args.samples,
            "seed": args.seed,
            "alpha": model.alpha,
        },
        "tickets": tickets,
    }
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"저장: {RESULT_PATH}")


if __name__ == "__main__":
    main()
