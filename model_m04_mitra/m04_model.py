"""
m04 확률 백엔드
================
모든 백엔드는 같은 인터페이스:  fit(X, y) → predict(X) → 번호별 출현확률 p (0~1)

  mitra    : Amazon Mitra-v2 (autogluon/mitra-classifier-2). 기본은 ICL only(fine_tune=False).
  logreg   : 로지스틱 회귀 — 파이프라인/feature 검증용 기준선 (가중치 다운로드 불필요)
  m03      : feature 의 s_post_m03 를 그대로 사용 (= 운영 중인 m03 와 동일한 확률)
  uniform  : 모든 번호 6/45 — "패턴 없음" 기준선

한 회차 안에서 p 의 합은 6 이어야 하므로 normalize_round() 로 맞춘 뒤 쓴다.
"""

import numpy as np

from m04_features import feature_names

BASE_RATE = 6 / 45
MITRA_V2_CLS = "autogluon/mitra-classifier-2"


class UniformBackend:
    name = "uniform"

    def __init__(self, **_):
        pass

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.full(len(X), BASE_RATE)


class M03Backend:
    name = "m03"

    def __init__(self, use_rehearsal: bool = False, **_):
        self.col = feature_names(use_rehearsal).index("s_post_m03")

    def fit(self, X, y):
        return self

    def predict(self, X):
        return X[:, self.col] * 6


class LogRegBackend:
    name = "logreg"

    def __init__(self, C: float = 0.05, **_):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        self.model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict_proba(X)[:, 1]


def _auto_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class MitraBackend:
    """
    autogluon.tabular[mitra] 의 sklearn 인터페이스를 직접 사용.
    (TabularPredictor 경유 시 max_rows=10000 / max_classes=10 제한과 bagging 이 끼어들어 제외)
    """
    name = "mitra"

    def __init__(
        self,
        hf_model: str = MITRA_V2_CLS,
        fine_tune: bool = False,
        fine_tune_steps: int = 50,
        n_estimators: int = 1,
        shuffle_features: bool = False,
        device: str = "auto",
        seed: int = 0,
        **_,
    ):
        from autogluon.tabular.models.mitra.sklearn_interface import MitraClassifier

        self.seed = seed
        self.model = MitraClassifier(
            hf_model=hf_model,
            fine_tune=fine_tune,
            fine_tune_steps=fine_tune_steps,
            n_estimators=n_estimators,
            shuffle_features=shuffle_features,
            device=_auto_device() if device == "auto" else device,
            metric="log_loss",
            seed=seed,
            verbose=False,
        )

    def fit(self, X, y):
        # AutoGluon Mitra 전처리(random_mirror_x)가 시드 없는 전역 np.random 으로 feature 부호를 뒤집는다.
        # 같은 회차를 매일 다시 예측해도 결과가 같도록 전역 RNG 를 잠깐 고정했다가 되돌린다.
        state = np.random.get_state()
        np.random.seed(self.seed)
        try:
            self.model.fit(X, y.astype(int))
        finally:
            np.random.set_state(state)
        return self

    def predict(self, X):
        return self.model.predict_proba(X)[:, 1]


BACKENDS = {b.name: b for b in (UniformBackend, M03Backend, LogRegBackend, MitraBackend)}


def make_backend(name: str, **kw):
    if name not in BACKENDS:
        raise ValueError(f"backend 는 {list(BACKENDS)} 중 하나: {name}")
    return BACKENDS[name](**kw)


def normalize_round(p: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """한 회차 45개 확률의 합을 6 으로 맞춘다 (본번호가 정확히 6개이므로)."""
    p = np.clip(np.asarray(p, dtype=float), eps, None)
    return np.clip(p * 6 / p.sum(), eps, 1 - eps)


def to_posterior(p: np.ndarray) -> np.ndarray:
    """번호별 확률(합 6) → m03 posterior 와 같은 스케일(합 1). web/number_gen 이 쓰는 형태."""
    p = np.asarray(p, dtype=float)
    return p / p.sum()
