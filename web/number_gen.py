"""
m03 (BayesianFrequencyModel) 기반 번호 생성 모듈
Oracle 서버에서 모델을 메모리 싱글턴으로 유지하고, 재학습 후 reload_model()로 교체.
"""

import itertools
import logging
import os
import random
import sys
import threading
from datetime import datetime

import numpy as np

# model_m03_claude 모듈 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model_m03_claude"))
from m03_model import BayesianFrequencyModel, NUM_BALLS, NUM_SETS  # noqa: E402

logger = logging.getLogger(__name__)

CKPT_PATH = os.path.join(os.path.dirname(__file__), "..", "model_m03_claude", "best_m03.npz")

_model: BayesianFrequencyModel | None = None
_loaded_at: datetime | None = None
_lock = threading.Lock()


def _load_model():
    global _model, _loaded_at
    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(
            f"모델 파일 없음: {CKPT_PATH}\n"
            f"먼저 python model_m03_claude/m03_train.py 를 실행하세요."
        )
    _model = BayesianFrequencyModel.load(CKPT_PATH)
    _loaded_at = datetime.now()
    logger.info(
        f"[number_gen] m03 로드 완료  회차 {_model.round_range[0]}~{_model.round_range[1]}  "
        f"α={_model.alpha}  추첨수={_model.num_draws_global}"
    )


def reload_model() -> dict:
    """재학습 후 호출. 현재 싱글턴을 새 체크포인트로 교체."""
    with _lock:
        _load_model()
    return get_model_info()


def get_model_info() -> dict:
    """현재 로드된 모델의 메타 정보."""
    with _lock:
        if _model is None:
            return {"loaded": False}
        return {
            "loaded": True,
            "round_range": list(_model.round_range),
            "num_draws_global": _model.num_draws_global,
            "num_draws_per_set": {k: int(v) for k, v in _model.num_draws_per_set.items()},
            "alpha": _model.alpha,
            "loaded_at": _loaded_at.isoformat() if _loaded_at else None,
        }


def _get_prob(ball_set: int) -> np.ndarray:
    with _lock:
        if _model is None:
            _load_model()
        return _model.posterior(ball_set)


def _get_cold_numbers(ball_set: int, top_cold: int = 5) -> list[int]:
    """최근 출현이 가장 적은 번호 top_cold 개 (m03는 시계열 정보가 없어서 빈도 하위로 대체)"""
    with _lock:
        if _model is None:
            _load_model()
        counts = _model.counts_per_set[ball_set]
    # 빈도 낮은 순
    idx = np.argsort(counts)[:top_cold]
    return sorted((idx + 1).tolist())


def generate_numbers(
    ball_set: int,
    strategy: int,
    top_n: int = 15,
    seed: int | None = None,
) -> dict:
    """
    번호 생성

    Parameters
    ----------
    ball_set : 1-5 고정, 0이면 랜덤 선택
    strategy : 1 확률상위 | 2 가중랜덤 | 3 구간균형 | 4 Cold포함
    top_n    : 번호 풀 크기 (기본 15)
    seed     : strategy 2 랜덤 시드
    """
    if ball_set == 0:
        ball_set = random.randint(1, NUM_SETS)

    prob     = _get_prob(ball_set)
    top_idx  = np.argsort(prob)[::-1][:top_n]
    top_nums = sorted((top_idx + 1).tolist())

    if strategy == 1:
        combos  = list(itertools.combinations(top_nums, 6))
        scored  = sorted(combos, key=lambda c: sum(prob[n - 1] for n in c), reverse=True)
        numbers = list(scored[0])

    elif strategy == 2:
        combos  = list(itertools.combinations(top_nums, 6))
        weights = np.array([sum(prob[n - 1] for n in c) for c in combos])
        weights /= weights.sum()
        rng     = np.random.default_rng(seed)
        idx     = rng.choice(len(combos), p=weights)
        numbers = list(combos[idx])

    elif strategy == 3:
        combos   = list(itertools.combinations(top_nums, 6))
        balanced = [
            c for c in combos
            if sum(1 for n in c if n <= 15) >= 1
            and sum(1 for n in c if 16 <= n <= 30) >= 1
            and sum(1 for n in c if n >= 31) >= 1
        ]
        pool    = balanced if balanced else combos
        scored  = sorted(pool, key=lambda c: sum(prob[n - 1] for n in c), reverse=True)
        numbers = list(scored[0])

    elif strategy == 4:
        cold     = set(_get_cold_numbers(ball_set, top_cold=5))
        ext_pool = sorted(set(top_nums) | cold)
        combos   = list(itertools.combinations(ext_pool, 6))
        filtered = [c for c in combos if set(c) & cold]
        pool     = filtered if filtered else combos
        scored   = sorted(pool, key=lambda c: sum(prob[n - 1] for n in c), reverse=True)
        numbers  = list(scored[0])

    else:
        raise ValueError(f"strategy는 1-4 사이여야 합니다: {strategy}")

    return {
        "ball_set": ball_set,
        "strategy": strategy,
        "numbers":  sorted(numbers),
        "pool":     top_nums,
    }
