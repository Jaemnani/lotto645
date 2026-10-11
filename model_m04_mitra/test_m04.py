"""
m04 테스트  —  python -m pytest model_m04_mitra -q
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model_m03_claude"))

import m04_data                                                  # noqa: E402
from m04_features import build, context, context_same_set, feature_names   # noqa: E402
from m04_model import M03Backend, UniformBackend, normalize_round, to_posterior   # noqa: E402


@pytest.fixture(scope="module")
def hist():
    return m04_data.load_csv()


def test_csv_parsing(hist):
    # 733~1230 회차 연속, 공세트 있는 회차 = 리허설 있는 회차
    assert hist.rounds[0] == 733
    assert (np.diff(hist.rounds) == 1).all()
    assert (hist.hits.sum(axis=1) == 6).all()
    assert ((hist.ball_set > 0) == hist.has_reh).all()
    assert (hist.reh_hits[hist.has_reh].sum(axis=1) == 6).all()


def test_no_future_leakage(hist):
    """회차 k 의 feature 는 k 이후 회차를 아무리 바꿔도 변하지 않아야 한다."""
    k = len(hist) - 50
    for use_reh in (False, True):
        base = build(hist, use_rehearsal=use_reh)
        tampered = hist.head(len(hist))
        rng = np.random.default_rng(1)
        for j in range(k, len(hist)):
            tampered.hits[j] = False
            tampered.hits[j, rng.choice(45, 6, replace=False)] = True
            tampered.reh_hits[j] = False
            tampered.reh_hits[j, rng.choice(45, 6, replace=False)] = True
            tampered.bonus[j] = rng.integers(1, 46)
            if j > k:   # 회차 k 의 공세트는 "가정 조건"이라 유지, 결과(번호)만 바꾼다
                tampered.ball_set[j] = rng.integers(1, 6)
        other = build(tampered, use_rehearsal=use_reh)
        for j in range(k + 1):
            if base.per_round[j] is not None:
                np.testing.assert_array_equal(base.per_round[j], other.per_round[j])


def test_same_round_rehearsal_not_used(hist):
    """같은 회차 리허설은 구매 마감 뒤에 알 수 있으므로 그 회차 feature 에 들어가면 안 된다."""
    k = len(hist) - 1
    tampered = hist.head(len(hist))
    tampered.reh_hits[k] = ~tampered.reh_hits[k]
    a = build(hist, use_rehearsal=True).per_round[k]
    b = build(tampered, use_rehearsal=True).per_round[k]
    np.testing.assert_array_equal(a, b)


def test_m03_equivalence(hist):
    """feature 의 s_post_m03 는 운영 m03(BayesianFrequencyModel) posterior 와 같아야 한다."""
    from m03_model import BayesianFrequencyModel

    ft = build(hist)
    rows = [
        {"ball_set": int(hist.ball_set[i]), "round": int(hist.rounds[i]),
         **{f"n{j + 1}": int(n) for j, n in enumerate(np.flatnonzero(hist.hits[i]) + 1)}}
        for i in range(len(hist)) if hist.ball_set[i] > 0
    ]
    m03 = BayesianFrequencyModel(alpha=1.0).fit(pd.DataFrame(rows))
    col = feature_names().index("s_post_m03")
    for b in range(1, 6):
        np.testing.assert_allclose(ft.next_query[b - 1][:, col], m03.posterior(b), rtol=1e-6)


def test_context_shape_and_limits(hist):
    ft = build(hist)
    X, y = context(ft, len(hist), 110)
    assert X.shape == (110 * 45, len(ft.names)) and X.shape[0] <= 5120
    assert len(ft.names) <= 50
    assert y.sum() == 110 * 6
    Xs, ys = context_same_set(ft, len(hist), 60, 4)
    assert Xs.shape[0] == 60 * 45
    assert (Xs[:, ft.names.index("ball_set")] == 4).all()


def test_normalization():
    p = normalize_round(np.linspace(0.05, 0.3, 45))
    assert abs(p.sum() - 6) < 1e-9 and (p > 0).all() and (p < 1).all()
    post = to_posterior(p)
    assert abs(post.sum() - 1) < 1e-12


def test_baseline_backends(hist):
    ft = build(hist)
    X, y = context(ft, len(hist), 20)
    q = ft.next_query[0]
    u = normalize_round(UniformBackend().fit(X, y).predict(q))
    np.testing.assert_allclose(u, 6 / 45)
    m = M03Backend().fit(X, y).predict(q)
    assert abs(m.sum() - 6) < 1e-6


def test_mitra_backend_is_seeded(monkeypatch):
    """Mitra 전처리가 전역 np.random 을 쓰므로, 같은 입력이면 같은 결과 + 전역 RNG 상태는 보존되어야 한다."""
    import types

    class FakeMitra:
        def __init__(self, **kw):
            self.kw = kw

        def fit(self, X, y):   # AutoGluon Preprocessor.determine_mirror 와 같은 방식
            self.mirror = np.random.choice([1, -1], size=(1, X.shape[1]))
            return self

        def predict_proba(self, X):
            s = 1 / (1 + np.exp(-(X * self.mirror).sum(axis=1) / 100))
            return np.stack([1 - s, s], axis=1)

    mod = types.ModuleType("sklearn_interface")
    mod.MitraClassifier = FakeMitra
    for name in ("autogluon", "autogluon.tabular", "autogluon.tabular.models",
                 "autogluon.tabular.models.mitra", "autogluon.tabular.models.mitra.sklearn_interface"):
        monkeypatch.setitem(sys.modules, name, mod if name.endswith("sklearn_interface") else types.ModuleType(name))
    from m04_model import MitraBackend

    X = np.random.default_rng(0).random((90, 16))
    y = np.array([0, 1] * 45)
    np.random.seed(123)
    before = np.random.random()
    np.random.seed(123)
    a = MitraBackend(device="cpu").fit(X, y).predict(X)
    b = MitraBackend(device="cpu").fit(X, y).predict(X)
    np.testing.assert_array_equal(a, b)
    assert np.random.random() == before   # 전역 RNG 흐름을 건드리지 않음


def test_load_supabase_pages_past_1000(monkeypatch):
    """postgrest range() 는 offset/limit 을 누적하므로 페이지마다 새 쿼리여야 1,000행 넘어도 정상."""
    import types

    rows = [{"round": 733 + i // 2, "is_winning": bool(i % 2), "ball_set": 1,
             "n1": 1, "n2": 2, "n3": 3, "n4": 4, "n5": 5, "n6": 6, "bonus": 7} for i in range(2500)]

    class Q:
        def __init__(self):
            self.ranges = []

        def select(self, *_):
            return self

        def order(self, *_):
            return self

        def range(self, a, b):
            self.ranges.append((a, b))
            return self

        def execute(self):
            a, b = self.ranges[0]          # 누적된 파라미터 중 첫 값이 쓰이는 최악의 경우를 흉내
            return types.SimpleNamespace(data=rows[a:b + 1])

    class SB:
        def table(self, _):
            return Q()

    fake_supabase = types.ModuleType("supabase")
    fake_supabase.create_client = lambda *_: SB()
    monkeypatch.setitem(sys.modules, "supabase", fake_supabase)
    h = m04_data.load_supabase()
    assert len(h) == 1250 and h.rounds[-1] == 733 + 1249


# ── v2: 리허설 합산·α·재현성 칼럼 / 인기도 / 티켓 선택 ──────────────────────────────

def test_v2_features_alpha_and_agree(hist):
    from m04_features import build, feature_names
    names = feature_names(True)
    assert names[-4:] == ["r_n", "r_post", "sr_post", "agree"] and len(names) <= 50
    a1 = build(hist, use_rehearsal=True, alpha=1.0).next_query
    a100 = build(hist, use_rehearsal=True, alpha=100.0).next_query
    sr = names.index("sr_post")
    # α 가 클수록 균등(1/45) 쪽으로 당겨진다
    assert np.abs(a100[:, :, sr] - 1 / 45).max() < np.abs(a1[:, :, sr] - 1 / 45).max()
    np.testing.assert_allclose(a100[:, :, sr].sum(axis=1), 1.0, rtol=1e-5)
    # s_post_m03 는 α 와 무관 (운영 m03 와 동등성 유지)
    m = names.index("s_post_m03")
    np.testing.assert_array_equal(a1[:, :, m], a100[:, :, m])
    # agree: 한쪽 표본만 치우치면 0
    from m04_features import _agree
    z = _agree(np.array([20.0] + [0] * 44), 30, np.array([0.0] * 45), 30)
    assert z[0] == 0


def test_m03r_backend_requires_rehearsal(hist):
    from m04_features import build
    from m04_model import M03RBackend, make_backend
    with pytest.raises(ValueError):
        make_backend("m03r")
    ft = build(hist, use_rehearsal=True, alpha=100.0)
    p = M03RBackend(use_rehearsal=True).predict(ft.next_query[0])
    assert abs(p.sum() - 6) < 1e-4


@pytest.fixture(scope="module")
def pop(hist):
    from m04_popularity import fit
    return fit(hist)


def test_popularity_model_is_predictive(hist, pop):
    from m04_popularity import holdout_check
    assert holdout_check(hist) > 0.5                 # 학습 안 한 최근 100회 5등 초과당첨 예측
    assert pop.gamma > 0 and pop.gamma / pop.gamma_se > 2
    low = np.array([[29, 36, 41, 42, 43, 45]])
    high = np.array([[3, 7, 9, 11, 12, 17]])
    assert pop.jackpot_share(low)[0] < 1 < pop.jackpot_share(high)[0]


def test_pick_constraints(hist, pop):
    from m04_pick import pick
    rates = np.full(45, 6 / 45)
    tickets = pick(rates, pop, hist, n=5, max_overlap=2, samples=50_000, seed=1)
    past = {tuple(np.flatnonzero(h) + 1) for h in hist.hits}
    assert len(tickets) == 5
    for i, t in enumerate(tickets):
        nums = t["numbers"]
        assert nums == sorted(nums) and len(set(nums)) == 6 and 1 <= nums[0] and nums[-1] <= 45
        assert tuple(nums) not in past
        assert t["co_winner_multiplier"] < 1                 # 균등 확률이면 인기 회피만으로 고름
        for u in tickets[:i]:
            assert len(set(nums) & set(u["numbers"])) <= 2


def test_pick_weight_zero_follows_model(hist, pop):
    from m04_pick import pick
    rates = np.full(45, (6 - 6 * 0.3) / 39)
    fav = [1, 2, 3, 4, 5, 6]
    rates[np.array(fav) - 1] = 0.3                         # 모델이 1~6 을 강하게 선호
    t = pick(rates, pop, hist, n=1, weight=0.0, samples=200_000, seed=0)[0]
    assert len(set(t["numbers"]) & set(fav)) >= 5           # 인기 무시하면 모델 선호 그대로


# ── 교차 검토 / Codex 지적 회귀 테스트 ──────────────────────────────────────────

def test_conditional_poisson_reproduces_inclusion_and_is_neutral_when_uniform():
    from m04_pick import _inclusion, combo_log_ratio, conditional_poisson_weights
    rng = np.random.default_rng(3)
    r = rng.uniform(0.08, 0.2, 45)
    r = r / r.sum() * 6
    w = conditional_poisson_weights(r)
    np.testing.assert_allclose(_inclusion(w), r, atol=1e-10)
    u = conditional_poisson_weights(np.full(45, 6 / 45))
    np.testing.assert_allclose(combo_log_ratio(np.array([[1, 2, 3, 4, 5, 6], [40, 41, 42, 43, 44, 45]]), u), 0, atol=1e-9)


def test_pick_warns_when_it_cannot_fill_n(hist, pop, capsys):
    from m04_pick import pick
    t = pick(np.full(45, 6 / 45), pop, hist, n=8, max_overlap=0, samples=50_000, seed=0)
    assert len(t) < 8 and "경고" in capsys.readouterr().out


def test_pick_rejects_stale_prediction(hist, tmp_path):
    import argparse
    import json
    from m04_pick import load_probs
    f = tmp_path / "prediction.json"
    f.write_text(json.dumps({"target_round": int(hist.rounds[-1]), "probs": {}}))
    args = argparse.Namespace(backend=None, prediction=str(f), allow_stale=False)
    with pytest.raises(SystemExit):
        load_probs(args, hist)


def test_backtest_uniform_baseline_uses_exact_expectation():
    from m04_backtest import score
    y = np.zeros(45)
    y[:6] = 1
    s = score(np.full(45, 6 / 45), y, np.random.default_rng(0))
    assert s["top15_hits"] == 2.0 and abs(s["top6_hits"] - 0.8) < 1e-12
