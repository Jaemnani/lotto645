"""
m04 feature 생성 (point-in-time)
=================================
한 행 = (회차 t, 번호 i, 공세트 b). 라벨 = 번호 i 가 회차 t 본번호 6개에 포함되었는가.

모든 feature 는 회차 t **이전** 회차만으로 계산한다. 회차를 앞에서부터 한 번 훑으면서
"t 를 보기 직전의 누적 상태"로 feature 를 만든 뒤에 t 를 상태에 반영하므로 구조적으로 누수가 없다.
(test_m04.py 의 누수 테스트가 이를 검증)

가설 → 컬럼
  H1 공세트 편향 (m03 가설) : ball_set, s_n, s_post_m03, s_cnt_10, s_gap  (+ 옵션 r_post)
  H2 최근 흐름(hot)          : g_cnt_10 / 30 / 100, prev_hit, prev_bonus
  H3 오래 안 나옴(cold)      : g_gap, s_gap
  H4 번호 자체 성질(대조군)  : number, odd, zone, last_digit
"""

from dataclasses import dataclass, field

import numpy as np

from m04_data import NUM_BALLS, NUM_SETS, History

ALPHA = 1.0             # m03 와 같은 Laplace 평활
GAP_CAP = 100           # 한 번도 안 나왔으면 이 값으로 캡

BASE_FEATURES = [
    "number", "odd", "zone", "last_digit",
    "g_rate", "g_cnt_10", "g_cnt_30", "g_cnt_100", "g_gap", "prev_hit", "prev_bonus",
    "ball_set", "s_n", "s_post_m03", "s_cnt_10", "s_gap",
]
REHEARSAL_FEATURES = ["r_n", "r_post"]


def feature_names(use_rehearsal: bool = False) -> list[str]:
    return BASE_FEATURES + (REHEARSAL_FEATURES if use_rehearsal else [])


_NUM   = np.arange(1, NUM_BALLS + 1)
_STATIC = np.stack([
    _NUM,
    _NUM % 2,
    np.minimum((_NUM - 1) // 15, 2),
    _NUM % 10,
], axis=1).astype(np.float32)


@dataclass
class _State:
    """회차 t 직전까지의 누적 상태"""
    n: int = 0
    g_cnt: np.ndarray = field(default_factory=lambda: np.zeros(NUM_BALLS))
    g_hist: list = field(default_factory=list)                 # 회차별 hits
    g_last: np.ndarray = field(default_factory=lambda: np.full(NUM_BALLS, -1))
    prev_hit: np.ndarray = field(default_factory=lambda: np.zeros(NUM_BALLS))
    prev_bonus: np.ndarray = field(default_factory=lambda: np.zeros(NUM_BALLS))
    s_n: np.ndarray = field(default_factory=lambda: np.zeros(NUM_SETS + 1, dtype=int))
    s_cnt: np.ndarray = field(default_factory=lambda: np.zeros((NUM_SETS + 1, NUM_BALLS)))
    s_hist: dict = field(default_factory=lambda: {b: [] for b in range(1, NUM_SETS + 1)})
    s_last: np.ndarray = field(default_factory=lambda: np.full((NUM_SETS + 1, NUM_BALLS), -1))
    r_n: np.ndarray = field(default_factory=lambda: np.zeros(NUM_SETS + 1, dtype=int))
    r_cnt: np.ndarray = field(default_factory=lambda: np.zeros((NUM_SETS + 1, NUM_BALLS)))

    def _window(self, hist: list, k: int) -> np.ndarray:
        if not hist:
            return np.zeros(NUM_BALLS)
        return np.sum(hist[-k:], axis=0)

    def features(self, b: int, use_rehearsal: bool) -> np.ndarray:
        """공세트 b 를 가정했을 때 45개 번호의 feature (45, F)"""
        g_gap = np.where(self.g_last < 0, GAP_CAP, np.minimum(self.n - self.g_last, GAP_CAP))
        n_b = self.s_n[b]
        s_gap = np.where(self.s_last[b] < 0, GAP_CAP, np.minimum(n_b - self.s_last[b], GAP_CAP))
        s_post = (self.s_cnt[b] + ALPHA) / (6 * n_b + ALPHA * NUM_BALLS)
        cols = [
            _STATIC,
            np.stack([
                self.g_cnt / max(self.n, 1),
                self._window(self.g_hist, 10),
                self._window(self.g_hist, 30),
                self._window(self.g_hist, 100),
                g_gap,
                self.prev_hit,
                self.prev_bonus,
                np.full(NUM_BALLS, b),
                np.full(NUM_BALLS, n_b),
                s_post,
                self._window(self.s_hist[b], 10),
                s_gap,
            ], axis=1),
        ]
        if use_rehearsal:
            r_post = (self.r_cnt[b] + ALPHA) / (6 * self.r_n[b] + ALPHA * NUM_BALLS)
            cols.append(np.stack([np.full(NUM_BALLS, self.r_n[b]), r_post], axis=1))
        return np.concatenate(cols, axis=1).astype(np.float32)

    def update(self, hits: np.ndarray, bonus: int, b: int, reh: np.ndarray | None):
        h = hits.astype(float)
        self.g_cnt += h
        self.g_hist.append(h)
        self.g_last[hits] = self.n
        self.n += 1
        self.prev_hit = h
        self.prev_bonus = np.zeros(NUM_BALLS)
        self.prev_bonus[bonus - 1] = 1
        if b > 0:
            self.s_cnt[b] += h
            self.s_hist[b].append(h)
            self.s_last[b][hits] = self.s_n[b]
            self.s_n[b] += 1
            if reh is not None:
                self.r_cnt[b] += reh
                self.r_n[b] += 1


@dataclass
class FeatureTable:
    """
    per_round[k]  : 회차 k 를 실제 공세트로 본 feature (45, F) — 공세트 모르면 None
    labels[k]     : (45,) 0/1
    next_query    : 마지막 회차 다음 회차용, 공세트 1~5 가정 (5, 45, F)
    """
    names: list[str]
    rounds: np.ndarray
    ball_set: np.ndarray
    per_round: list
    labels: np.ndarray
    next_query: np.ndarray


def build(hist: History, use_rehearsal: bool = False) -> FeatureTable:
    st = _State()
    per_round = []
    for k in range(len(hist)):
        b = int(hist.ball_set[k])
        per_round.append(st.features(b, use_rehearsal) if b > 0 else None)
        reh = hist.reh_hits[k] if hist.has_reh[k] else None
        st.update(hist.hits[k], int(hist.bonus[k]), b, reh)
    next_query = np.stack([st.features(b, use_rehearsal) for b in range(1, NUM_SETS + 1)])
    return FeatureTable(
        names=feature_names(use_rehearsal),
        rounds=hist.rounds,
        ball_set=hist.ball_set,
        per_round=per_round,
        labels=hist.hits.astype(int),
        next_query=next_query,
    )


def context(ft: FeatureTable, before_idx: int, max_rounds: int) -> tuple[np.ndarray, np.ndarray]:
    """회차 인덱스 before_idx 미만에서, 공세트가 있는 최근 max_rounds 회차를 컨텍스트(학습) 행으로 모은다."""
    idx = [k for k in range(before_idx) if ft.per_round[k] is not None]
    idx = idx[-max_rounds:]
    if not idx:
        raise ValueError("컨텍스트로 쓸 회차가 없음")
    X = np.concatenate([ft.per_round[k] for k in idx])
    y = np.concatenate([ft.labels[k] for k in idx])
    return X, y


def context_same_set(ft: FeatureTable, before_idx: int, max_rounds: int, b: int) -> tuple[np.ndarray, np.ndarray]:
    """공세트 b 인 회차만 컨텍스트로 (세트 편향 가설을 직접 보는 설정)"""
    idx = [k for k in range(before_idx) if ft.ball_set[k] == b and ft.per_round[k] is not None]
    idx = idx[-max_rounds:]
    if not idx:
        raise ValueError(f"공세트 {b} 컨텍스트 없음")
    X = np.concatenate([ft.per_round[k] for k in idx])
    y = np.concatenate([ft.labels[k] for k in idx])
    return X, y
