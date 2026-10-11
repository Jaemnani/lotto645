"""
m04 인기도 모델 — "남들이 많이 고르는 번호" 추정
=================================================
1등 확률은 어떤 조합이든 같지만, 당첨금은 같은 조합을 산 사람끼리 나눈다.
그래서 **남들이 덜 고르는 조합**을 사면 당첨 시 받는 기대 금액이 커진다.

추정 방법
  1. 회차별 5등(3개 일치) 당첨 게임 수 / 기대값(판매 게임 수 × 5등 확률) 의 로그 = 그 회차 당첨번호의 인기
     · 판매 게임 수 ≈ 총 당첨금(데이터의 total_sales 컬럼) ÷ 0.5 ÷ 1,000원 (법정 환급률 약 50%)
       — 상수 배율은 중심화·절편에 흡수되므로 결과(β, γ)에는 영향 없음
     · 5등은 회차당 수백만 게임이라 잡음이 거의 없다 (1등 12명 남짓과 대조)
  2. 그 값을 번호 45개 indicator 로 ridge 회귀 → 번호별 인기 β
  3. 교차적합(자기 회차를 안 본) 인기점수로 1등 당첨자 수를 Poisson 회귀 → γ
     · γ = 인기점수 1단위당 1등 공동당첨자 배수의 로그 (과대산포 보정한 표준오차 포함)

조합 c 의 1등 공동당첨 배수 ≈ exp(γ · (score(c) − 평균 당첨조합 점수))

검증 (2026-10, 733~1230회 498회):
  · 5등 기반 β 가 학습에 안 쓴 최근 100회 5등 초과 당첨을 상관 0.88 로 예측
  · γ 의 z ≈ 7 (과대산포 보정 후 ≈ 5): 인기 하위 1% 조합의 1등 공동당첨 ≈ 평균의 0.8배

사용:
  python m04_popularity.py          # 적합 + 진단 출력
"""

import argparse
import os
import sys
from dataclasses import dataclass
from math import comb

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import m04_data                      # noqa: E402
from m04_data import NUM_BALLS       # noqa: E402

N_COMBOS = comb(45, 6)
GRADE_PROB = {
    1: 1 / N_COMBOS,
    2: 6 / N_COMBOS,
    3: 6 * 38 / N_COMBOS,
    4: comb(6, 4) * comb(39, 2) / N_COMBOS,
    5: comb(6, 3) * comb(39, 3) / N_COMBOS,
}
TICKET_PRICE = 1000
PAYOUT_RATIO = 0.5      # 총 당첨금 / 판매액 (관측: 등수별 관측/기대 당첨 비가 전 등수 ≈ 1.0 이 되는 값)


@dataclass
class PopularityModel:
    beta: np.ndarray          # (45,) 번호별 인기 (로그 초과 당첨, 클수록 인기)
    ref: float                # 과거 당첨조합 인기점수 평균 (= 무작위 조합의 평균)
    gamma: float              # 인기점수 1단위당 1등 공동당첨 로그 배수
    gamma_se: float           # 과대산포 보정 표준오차
    n_rounds: int
    grade: int

    def score(self, combos) -> np.ndarray:
        """combos: (n, 6) 1-based 번호 → 인기점수 (클수록 인기)"""
        c = np.asarray(combos, dtype=int)
        return self.beta[c - 1].sum(axis=-1)

    def jackpot_share(self, combos) -> np.ndarray:
        """1등 공동당첨자 기대 배수 (1.0 = 평균 조합, 작을수록 당첨금을 덜 나눔)"""
        return np.exp(self.gamma * (self.score(combos) - self.ref))


def _games(hist) -> np.ndarray:
    """회차별 판매 게임 수 추정"""
    return hist.prize_pool / PAYOUT_RATIO / TICKET_PRICE


def _excess(hist, grade):
    w = hist.winners[:, grade - 1]
    games = _games(hist)
    ok = np.isfinite(w) & np.isfinite(games) & (games > 0)
    y = np.log((w[ok] + 0.5) / (games[ok] * GRADE_PROB[grade]))
    return ok, y - y.mean()


def fit(hist, grade: int = 5, ridge: float = 30.0, folds: int = 5, seed: int = 0) -> PopularityModel:
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold

    ok, y = _excess(hist, grade)
    if ok.sum() < 50:
        raise ValueError(f"당첨자/판매액 데이터가 부족함 ({ok.sum()}회)")
    X = hist.hits[ok].astype(float)

    # 교차적합 인기점수 — 자기 회차의 당첨 정보를 보지 않은 점수로 1등 효과를 잰다
    sc = np.zeros(len(y))
    coefs = []
    for tr, te in KFold(folds, shuffle=True, random_state=seed).split(X):
        m = Ridge(alpha=ridge).fit(X[tr], y[tr])
        sc[te] = m.predict(X[te])
        coefs.append(m.coef_)

    # 1등 Poisson: log E[w1] = log(games/N) + a + γ·sc   (뉴턴법)
    w1 = hist.winners[ok, 0]
    off = np.log(_games(hist)[ok] * GRADE_PROB[1])
    A = np.c_[np.ones(len(sc)), sc - sc.mean()]
    b = np.zeros(2)
    for _ in range(100):
        mu = np.exp(off + A @ b)
        step = np.linalg.solve(A.T @ (A * mu[:, None]), A.T @ (w1 - mu))
        b += step
        if np.abs(step).max() < 1e-10:
            break
    mu = np.exp(off + A @ b)
    phi = max(1.0, float(((w1 - mu) ** 2 / mu).sum() / (len(w1) - 2)))      # 과대산포
    se = float(np.sqrt(np.linalg.inv(A.T @ (A * mu[:, None]))[1, 1] * phi))

    # γ 는 폴드 모델(데이터 4/5 로 적합, 더 강하게 축소된) 점수로 추정했으므로, 새 조합도 같은 축소 수준인
    # 폴드 계수 평균으로 점수화해야 γ 의 척도가 맞는다 (전체 적합 계수를 쓰면 효과가 ~10% 과장됨)
    beta = np.mean(coefs, axis=0)
    ref = float((X @ beta).mean())
    return PopularityModel(beta=beta, ref=ref, gamma=float(b[1]), gamma_se=se,
                           n_rounds=int(ok.sum()), grade=grade)


def holdout_check(hist, grade: int = 5, holdout: int = 100, ridge: float = 30.0) -> float:
    """최근 holdout 회차를 빼고 β 를 학습 → 그 회차의 초과 당첨을 얼마나 맞히는지 (상관)"""
    from sklearn.linear_model import Ridge

    ok, y = _excess(hist, grade)
    X = hist.hits[ok].astype(float)
    m = Ridge(alpha=ridge).fit(X[:-holdout], y[:-holdout])
    return float(np.corrcoef(m.predict(X[-holdout:]), y[-holdout:])[0, 1])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="csv", choices=["csv", "supabase"])
    args = ap.parse_args()

    hist = m04_data.load(args.source)
    pm = fit(hist)
    order = np.argsort(pm.beta)
    rng = np.random.default_rng(0)
    rand = np.sort(np.argsort(rng.random((200_000, NUM_BALLS)), axis=1)[:, :6] + 1, axis=1)
    share = pm.jackpot_share(rand)

    print(f"학습 {pm.n_rounds}회 ({int(hist.rounds[0])}~{int(hist.rounds[-1])}), 기준 {pm.grade}등")
    print(f"최근 100회 holdout 예측 상관: {holdout_check(hist):+.2f}")
    print(f"γ = {pm.gamma:.2f} ± {pm.gamma_se:.2f} (z={pm.gamma / pm.gamma_se:.1f}, 과대산포 보정)")
    print("인기 많은 번호:", [int(j + 1) for j in order[::-1][:10]])
    print("인기 적은 번호:", [int(j + 1) for j in order[:10]])
    print("무작위 조합의 1등 공동당첨 배수 분포:",
          " ".join(f"하위{q}% x{np.percentile(share, q):.2f}" for q in (1, 10, 50, 90, 99)))


if __name__ == "__main__":
    main()
