"""
기존 LSTM 모델을 이용한 번호 생성 모듈
Oracle Cloud 서버에서 모델을 메모리에 로드하고 재사용
"""

import itertools
import logging
import os
import random
import sys

import numpy as np
import pandas as pd
import torch

# model_m02_claude 모듈 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model_m02_claude"))
from m02_model import BallSetLSTM  # noqa: E402

logger = logging.getLogger(__name__)

DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/history_from_cafe.csv")
CKPT_PATH = os.path.join(os.path.dirname(__file__), "../model_m02_claude/best_m02.pth")
NUM_BALLS = 45
POS_COLS  = ["n1", "n2", "n3", "n4", "n5", "n6"]

# 싱글턴: 서버 기동 시 1회만 로드
_model  = None
_cfg    = None
_device = None


def _load_model():
    global _model, _cfg, _device
    if _model is not None:
        return

    _device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    ckpt   = torch.load(CKPT_PATH, map_location=_device, weights_only=True)
    _cfg   = ckpt["config"]
    _model = BallSetLSTM(
        num_balls    = _cfg["num_balls"],
        num_ball_sets= _cfg["num_ball_sets"],
        emb_dim      = _cfg["emb_dim"],
        hidden_size  = _cfg["hidden_size"],
        num_layers   = _cfg["num_layers"],
        dropout      = 0.0,
    ).to(_device)
    _model.load_state_dict(ckpt["model_state"])
    _model.eval()
    logger.info(f"[number_gen] 모델 로드 완료 ({_device})")


def _get_prob(ball_set: int) -> np.ndarray:
    _load_model()
    df = pd.read_csv(
        DATA_PATH, header=None,
        names=["ball_set","round","draw_date","n1","n2","n3","n4","n5","n6","bonus"],
    )
    df = df.sort_values(["round", "ball_set"]).reset_index(drop=True)

    win_size = _cfg["win_size"]
    rows     = df.tail(win_size).reset_index(drop=True)
    pad      = win_size - len(rows)

    oh_list, bs_list = [], []
    for _, row in rows.iterrows():
        arr = np.zeros(NUM_BALLS, dtype=np.float32)
        for c in ["n1","n2","n3","n4","n5","n6","bonus"]:
            v = int(row[c])
            if 1 <= v <= NUM_BALLS:
                arr[v - 1] = 1.0
        oh_list.append(arr)
        bs_list.append(int(row["ball_set"]) - 1)

    if pad > 0:
        oh_list = [np.zeros(NUM_BALLS, dtype=np.float32)] * pad + oh_list
        bs_list = [0] * pad + bs_list

    x_oh = torch.from_numpy(np.array(oh_list)).unsqueeze(0).to(_device)
    x_bs = torch.tensor([bs_list], dtype=torch.long).to(_device)
    t_bs = torch.tensor([ball_set - 1], dtype=torch.long).to(_device)

    with torch.no_grad():
        prob = _model(x_oh, x_bs, t_bs)[0].cpu().numpy()
    return prob


def _get_cold_numbers(ball_set: int, top_cold: int = 5) -> list[int]:
    df = pd.read_csv(
        DATA_PATH, header=None,
        names=["ball_set","round","draw_date","n1","n2","n3","n4","n5","n6","bonus"],
    )
    sub = df[df["ball_set"] == ball_set].reset_index(drop=True)
    last_seen: dict[int, int] = {}
    for i, row in sub.iterrows():
        for c in POS_COLS:
            last_seen[int(row[c])] = int(i)
    all_nums = {n: last_seen.get(n, -1) for n in range(1, NUM_BALLS + 1)}
    cold = sorted(all_nums.items(), key=lambda x: x[1])[:top_cold]
    return [n for n, _ in cold]


def generate_numbers(
    ball_set: int,
    strategy: int,
    top_n: int = 15,
    seed: int | None = None,
) -> dict:
    """
    번호 생성 메인 함수

    Parameters
    ----------
    ball_set : 1-5 고정, 0이면 랜덤 선택
    strategy : 1 확률상위 | 2 가중랜덤 | 3 구간균형 | 4 Cold포함
    top_n    : 번호 풀 크기 (기본 15)
    seed     : 랜덤 시드 (strategy 2 재현용)

    Returns
    -------
    {"ball_set": int, "strategy": int, "numbers": [6개], "pool": [top_n개]}
    """
    if ball_set == 0:
        ball_set = random.randint(1, 5)

    prob     = _get_prob(ball_set)
    top_idx  = np.argsort(prob)[::-1][:top_n]
    top_nums = sorted((top_idx + 1).tolist())

    if strategy == 1:
        # 확률 합산 최대 조합
        combos  = list(itertools.combinations(top_nums, 6))
        scored  = sorted(combos, key=lambda c: sum(prob[n - 1] for n in c), reverse=True)
        numbers = list(scored[0])

    elif strategy == 2:
        # 확률 가중 랜덤 조합
        combos  = list(itertools.combinations(top_nums, 6))
        weights = np.array([sum(prob[n - 1] for n in c) for c in combos])
        weights /= weights.sum()
        rng     = np.random.default_rng(seed)
        idx     = rng.choice(len(combos), p=weights)
        numbers = list(combos[idx])

    elif strategy == 3:
        # 저-중-고 구간 균형 (각 구간 1개 이상)
        combos   = list(itertools.combinations(top_nums, 6))
        balanced = [
            c for c in combos
            if sum(1 for n in c if n <= 15) >= 1
            and sum(1 for n in c if 16 <= n <= 30) >= 1
            and sum(1 for n in c if n >= 31) >= 1
        ]
        pool   = balanced if balanced else combos
        scored = sorted(pool, key=lambda c: sum(prob[n - 1] for n in c), reverse=True)
        numbers = list(scored[0])

    elif strategy == 4:
        # Cold 번호 1개 이상 포함
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
