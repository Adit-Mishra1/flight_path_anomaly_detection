from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

try:
    from sklearn.linear_model import SGDClassifier
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


@dataclass
class ModelScore:
    score: float
    is_anomaly: bool
    details: dict


class OnlineBinaryAnomalyModel:
    def __init__(self, n_features: int, threshold: float = 0.65):
        self.n_features = n_features
        self.threshold = threshold
        self._fallback_center = np.zeros(n_features)
        self._fallback_var = np.ones(n_features)
        self._fallback_n = 0

        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.clf = (
            SGDClassifier(loss="log_loss", alpha=0.0005, random_state=42)
            if SKLEARN_AVAILABLE
            else None
        )
        self._initialized = False

    def update(self, x: np.ndarray, y: int) -> float:
        x = x.reshape(1, -1)

        if not SKLEARN_AVAILABLE:
            self._fallback_update(x[0])
            return float(self._fallback_score(x[0]))

        self.scaler.partial_fit(x)
        x_scaled = self.scaler.transform(x)
        classes = np.array([0, 1])
        if not self._initialized:
            self.clf.partial_fit(x_scaled, np.array([y]), classes=classes)
            self._initialized = True
        else:
            self.clf.partial_fit(x_scaled, np.array([y]))

        p = self.clf.predict_proba(x_scaled)[0][1]
        return float(p)

    def score(self, x: np.ndarray) -> float:
        if not SKLEARN_AVAILABLE or not self._initialized:
            return float(self._fallback_score(x))
        x_scaled = self.scaler.transform(x.reshape(1, -1))
        return float(self.clf.predict_proba(x_scaled)[0][1])

    def evaluate(self, x: np.ndarray) -> ModelScore:
        score = self.score(x)
        return ModelScore(score=score, is_anomaly=score >= self.threshold, details={})

    def _fallback_update(self, x: np.ndarray) -> None:
        self._fallback_n += 1
        if self._fallback_n == 1:
            self._fallback_center = x
            return
        delta = x - self._fallback_center
        self._fallback_center += delta / self._fallback_n
        delta2 = x - self._fallback_center
        self._fallback_var += delta * delta2

    def _fallback_score(self, x: np.ndarray) -> float:
        std = np.sqrt(np.maximum(self._fallback_var / max(self._fallback_n - 1, 1), 1e-3))
        z = np.abs((x - self._fallback_center) / std)
        raw = float(np.mean(z))
        return 1.0 - math.exp(-raw / 3.0)

class TinyAutoencoder:
    """Minimal online autoencoder-style learner for reconstruction-error anomaly scoring."""

    def __init__(self, input_dim: int, hidden_dim: int = 4, lr: float = 0.001):
        rng = np.random.default_rng(42)
        self.w1 = rng.normal(0, 0.1, size=(input_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        self.w2 = rng.normal(0, 0.1, size=(hidden_dim, input_dim))
        self.b2 = np.zeros(input_dim)
        self.lr = lr
        self.err_ema = 1.0

    def train_and_score(self, x: np.ndarray) -> float:
        h = np.tanh(x @ self.w1 + self.b1)
        y = h @ self.w2 + self.b2
        err_vec = y - x
        mse = float(np.mean(err_vec**2))

        # Backprop for one-sample SGD update.
        dy = 2.0 * err_vec / len(x)
        dw2 = np.outer(h, dy)
        db2 = dy
        dh = dy @ self.w2.T
        dz1 = dh * (1.0 - h**2)
        dw1 = np.outer(x, dz1)
        db1 = dz1

        self.w2 -= self.lr * dw2
        self.b2 -= self.lr * db2
        self.w1 -= self.lr * dw1
        self.b1 -= self.lr * db1

        self.err_ema = 0.95 * self.err_ema + 0.05 * mse
        norm = mse / max(self.err_ema, 1e-6)
        return float(1.0 - math.exp(-norm))
