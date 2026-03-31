"""
Step 2 - 모델 정의
구조: Shared LSTM Backbone + Ball-Set Conditional Head

입력:
  - x         : (batch, seq_len, 45)  과거 추첨 번호 원-핫 시퀀스
  - ball_set  : (batch,)              다음 회차 공세트 번호 (0-indexed: 0~4)

출력:
  - (batch, 45) 각 번호의 출현 확률 (sigmoid)

설계 근거:
  · 공세트당 샘플이 ~150개 수준 → 세트별 독립 모델은 과적합
  · LSTM 이 전체 시계열 패턴을 학습, Ball-Set Embedding 이 세트별 편향 보정
  · 보너스 번호 포함 7개를 multi-label 로 예측, 상위 6개 → 주번호 추천
"""

import torch
import torch.nn as nn


class BallSetLSTM(nn.Module):
    """
    Shared LSTM Backbone + Ball-Set Conditional Head

    Parameters
    ----------
    num_balls     : int, 번호 범위 (기본 45)
    num_ball_sets : int, 공세트 수 (기본 5)
    emb_dim       : int, 공세트 임베딩 차원
    hidden_size   : int, LSTM 은닉층 크기
    num_layers    : int, LSTM 층 수
    dropout       : float, LSTM / FC dropout
    """

    def __init__(
        self,
        num_balls: int = 45,
        num_ball_sets: int = 5,
        emb_dim: int = 16,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_balls = num_balls
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # ── LSTM backbone ─────────────────────────────────────────────────────
        # 입력: 원-핫 번호(45) + 공세트 원-핫(num_ball_sets) per time step
        lstm_input_dim = num_balls + num_ball_sets
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # ── 다음 회차 공세트 임베딩 ───────────────────────────────────────────
        self.ball_set_emb = nn.Embedding(num_ball_sets, emb_dim)

        # ── 예측 헤드 ─────────────────────────────────────────────────────────
        head_input = hidden_size + emb_dim
        self.head = nn.Sequential(
            nn.Linear(head_input, head_input),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_input, num_balls),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,          # (B, T, 45)  과거 번호 원-핫
        hist_ball_sets: torch.Tensor,   # (B, T)      각 타임스텝의 공세트 (0-indexed)
        target_ball_set: torch.Tensor,  # (B,)        예측할 회차의 공세트
    ) -> torch.Tensor:
        """
        Returns
        -------
        prob : (B, 45)  각 번호의 출현 확률
        """
        B, T, _ = x.shape

        # 공세트 원-핫 → 히스토리에 concat
        bs_onehot = torch.zeros(B, T, self.ball_set_emb.num_embeddings,
                                device=x.device)
        bs_onehot.scatter_(2, hist_ball_sets.unsqueeze(-1), 1.0)  # (B, T, 5)

        lstm_in = torch.cat([x, bs_onehot], dim=-1)  # (B, T, 50)

        lstm_out, _ = self.lstm(lstm_in)              # (B, T, H)
        last_hidden = lstm_out[:, -1, :]              # (B, H)

        # 다음 회차 공세트 임베딩
        target_emb = self.ball_set_emb(target_ball_set)  # (B, emb_dim)

        combined = torch.cat([last_hidden, target_emb], dim=-1)  # (B, H+emb)
        prob = self.head(combined)                                 # (B, 45)
        return prob


# ── 유틸 함수 ──────────────────────────────────────────────────────────────────
def count_hits(pred_top6: torch.Tensor, target_onehot: torch.Tensor) -> float:
    """예측 상위 6개 번호와 정답 번호의 평균 일치 개수"""
    B = pred_top6.shape[0]
    total = 0
    for i in range(B):
        pred_set = set(pred_top6[i].tolist())
        target_nums = (target_onehot[i].nonzero(as_tuple=True)[0] + 1).tolist()
        total += len(pred_set & set(target_nums))
    return total / B
