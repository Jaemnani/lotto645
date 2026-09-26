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
