"""
Step 2 - 모델 정의
구조: Bayesian Frequency Model (베이지안 빈도 모델)

가설
  · 물리적 공의 질량/마모 편차로 특정 번호가 장기적으로 편향 출현한다
  · LSTM 같은 시계열 패턴 학습 대신, 단순 빈도 통계 + 베이지안 평활로 접근
  · 공세트(1~5)는 서로 다른 물리 공 세트 → 세트별 독립 빈도 추정

사후확률
  p_i = (count_i + α) / (N + α × 45)
    count_i : 번호 i의 과거 출현 횟수
    N       : 총 관측 수
    α       : Laplace 평활 계수 (0이면 순수 빈도, 1이면 uniform prior)
"""

import numpy as np

NUM_BALLS = 45
NUM_SETS  = 5


class BayesianFrequencyModel:
    """
    공세트별 베이지안 빈도 모델

    Parameters
    ----------
    alpha : float, Laplace 평활 계수 (default 1.0)
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        # 공세트별 번호 카운트: {1: np.array(45,), ..., 5: np.array(45,)}
        self.counts_per_set: dict[int, np.ndarray] = {}
        # 글로벌 카운트
        self.counts_global: np.ndarray | None = None
        self.num_draws_per_set: dict[int, int] = {}
        self.num_draws_global: int = 0
        self.round_range: tuple[int, int] | None = None

    def fit(self, df, include_bonus: bool = False):
        """
        df 컬럼: ball_set, round, draw_date, n1~n6, (bonus)
        """
        cols = ["n1","n2","n3","n4","n5","n6"] + (["bonus"] if include_bonus else [])
        self.counts_global = np.zeros(NUM_BALLS, dtype=np.float64)
        for bs in range(1, NUM_SETS + 1):
            self.counts_per_set[bs] = np.zeros(NUM_BALLS, dtype=np.float64)
            self.num_draws_per_set[bs] = 0

        for _, row in df.iterrows():
            bs = int(row["ball_set"])
            for c in cols:
                v = int(row[c])
                if 1 <= v <= NUM_BALLS:
                    self.counts_global[v - 1] += 1
                    if 1 <= bs <= NUM_SETS:
                        self.counts_per_set[bs][v - 1] += 1
            if 1 <= bs <= NUM_SETS:
                self.num_draws_per_set[bs] += 1

        self.num_draws_global = len(df)
        self.round_range = (int(df["round"].min()), int(df["round"].max()))
        return self

    def posterior(self, ball_set: int | None = None) -> np.ndarray:
        """
        공세트별(혹은 글로벌) 사후 출현 확률 (45,)
        ball_set=None 이면 글로벌.
        """
        counts = self.counts_global if ball_set is None else self.counts_per_set[ball_set]
        return (counts + self.alpha) / (counts.sum() + self.alpha * NUM_BALLS)

    def save(self, path: str):
        np.savez(
            path,
            alpha=self.alpha,
            counts_global=self.counts_global,
            counts_per_set=np.stack([self.counts_per_set[bs] for bs in range(1, NUM_SETS+1)]),
            num_draws_per_set=np.array([self.num_draws_per_set[bs] for bs in range(1, NUM_SETS+1)]),
            num_draws_global=self.num_draws_global,
            round_min=self.round_range[0],
            round_max=self.round_range[1],
        )

    @classmethod
    def load(cls, path: str) -> "BayesianFrequencyModel":
        d = np.load(path)
        m = cls(alpha=float(d["alpha"]))
        m.counts_global = d["counts_global"]
        for i in range(NUM_SETS):
            m.counts_per_set[i + 1] = d["counts_per_set"][i]
            m.num_draws_per_set[i + 1] = int(d["num_draws_per_set"][i])
        m.num_draws_global = int(d["num_draws_global"])
        m.round_range = (int(d["round_min"]), int(d["round_max"]))
        return m
