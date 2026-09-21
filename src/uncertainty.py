"""Precomputed, inference-cheap uncertainty and novelty detection.

Everything in this module is fit **once at training time** and stored in the
model bundle, so serving a prediction stays a single model forward pass plus
O(log n) / O(d^2) arithmetic - no extra model calls, no per-request lag.

Two capabilities are added on top of the bare probability:

* **Out-of-fold residual band** (a K-fold analogue of the jackknife interval).
  k-fold out-of-fold predicted probabilities give absolute residuals
  ``|y - p|`` against the binary label; at inference the point probability is
  widened by one global half-width, the (1 - ``alpha``) residual quantile.
  Read as a set of labels it coincides with a conformal label set (score
  ``1 - p_true``), so the target is coverage of the *label*, approximately -
  it is not the CV+ interval, which would need the fold models at query time.
* **Robust-Mahalanobis out-of-distribution score**. A Ledoit-Wolf-shrunk
  Gaussian is fit on the encoded training features; a live record beyond a
  high training-distance quantile is flagged as *unlike the cohort the model
  learned from*. This matters here because every source dataset is small and
  demographically narrow (see ``data/README.md``).

Both are model-agnostic: they only need an estimator exposing sklearn-style
``fit`` / ``predict_proba`` (TabPFN and the ``LogisticRegression`` test
stand-in both qualify).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold

EstimatorFactory = Callable[[], object]


# --------------------------------------------------------------------------- #
# Cross-validated out-of-fold probabilities (shared basis for the band + the
# learned decision threshold below)
# --------------------------------------------------------------------------- #
def oof_probabilities(
    estimator_factory: EstimatorFactory,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
) -> np.ndarray:
    """Length-``n`` out-of-fold ``P(y = 1)``, a fresh estimator fit per fold.

    ``n_splits`` is capped at the size of the smallest class so this stays
    valid on the tiny, imbalanced clinical splits used here.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).astype(int)
    min_class = int(np.bincount(y).min()) if len(y) else 0
    n_splits = max(2, min(n_splits, min_class))

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof = np.zeros(len(y), dtype=float)
    for train_idx, val_idx in skf.split(X, y):
        est = estimator_factory()
        est.fit(X[train_idx], y[train_idx])
        oof[val_idx] = est.predict_proba(X[val_idx])[:, 1]
    return oof


# --------------------------------------------------------------------------- #
# Out-of-fold residual probability band
# --------------------------------------------------------------------------- #
def conformal_residuals(oof_proba: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Absolute-residual nonconformity scores ``|y - p|``, sorted ascending."""
    y = np.asarray(y).astype(float)
    p = np.asarray(oof_proba, dtype=float)
    return np.sort(np.abs(y - p))


def conformal_halfwidth(residuals: np.ndarray, alpha: float = 0.10) -> float:
    """Finite-sample ``(1 - alpha)`` quantile of the residuals (Vovk rank).

    Uses the ``ceil((n + 1)(1 - alpha))``-th order statistic; if ``alpha`` is
    too small for the calibration set size the widest available residual is
    returned (coverage degrades gracefully rather than raising).
    """
    residuals = np.asarray(residuals, dtype=float)
    n = len(residuals)
    if n == 0:
        return float("nan")
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    k = min(max(k, 1), n)
    return float(residuals[k - 1])


def probability_band(p_hat: float, residuals: np.ndarray, alpha: float = 0.10) -> tuple[float, float]:
    """Widen a point probability to a ``[lo, hi]`` residual band, clipped to [0, 1]."""
    h = conformal_halfwidth(residuals, alpha)
    if not np.isfinite(h):
        return (float(p_hat), float(p_hat))
    return (float(max(0.0, p_hat - h)), float(min(1.0, p_hat + h)))


# --------------------------------------------------------------------------- #
# Learned decision threshold (replaces a hard-coded 0.5 cutoff)
# --------------------------------------------------------------------------- #
def youden_threshold(oof_proba: np.ndarray, y: np.ndarray) -> float:
    """Probability cutoff maximising Youden's J (= sensitivity + specificity - 1).

    Computed on the out-of-fold predictions so it does not peek at the test
    split. Clamped away from the degenerate 0/1 ends.
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(oof_proba, dtype=float)
    if len(np.unique(y)) < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(y, p)
    j = tpr - fpr
    t = float(thr[int(np.argmax(j))])
    if not np.isfinite(t):
        return 0.5
    return float(min(max(t, 0.01), 0.99))


# --------------------------------------------------------------------------- #
# Out-of-distribution / novelty detection
# --------------------------------------------------------------------------- #
@dataclass
class OODDetector:
    """Robust-Mahalanobis novelty detector over encoded training features.

    Picklable (plain NumPy arrays only). Fit on the same feature matrix the
    model trains on, stored in the model bundle, and queried per prediction.
    """

    mean: np.ndarray
    precision: np.ndarray  # inverse covariance (Ledoit-Wolf shrunk -> always invertible)
    ref_distances_sorted: np.ndarray  # Mahalanobis distances of the training rows, sorted
    quantile: float = 0.975

    @classmethod
    def fit(cls, X: np.ndarray, quantile: float = 0.975) -> "OODDetector":
        X = np.asarray(X, dtype=float)
        lw = LedoitWolf().fit(X)
        diff = X - lw.location_
        d = np.sqrt(np.clip(np.einsum("ij,jk,ik->i", diff, lw.precision_, diff), 0.0, None))
        return cls(
            mean=np.asarray(lw.location_, dtype=float),
            precision=np.asarray(lw.precision_, dtype=float),
            ref_distances_sorted=np.sort(d),
            quantile=float(quantile),
        )

    @property
    def threshold(self) -> float:
        return float(np.quantile(self.ref_distances_sorted, self.quantile))

    def distance(self, x: np.ndarray) -> float:
        diff = np.asarray(x, dtype=float).ravel() - self.mean
        return float(np.sqrt(max(0.0, diff @ self.precision @ diff)))

    def distances(self, X: np.ndarray) -> np.ndarray:
        diff = np.asarray(X, dtype=float) - self.mean
        return np.sqrt(np.clip(np.einsum("ij,jk,ik->i", diff, self.precision, diff), 0.0, None))

    def novelty_score(self, x: np.ndarray) -> float:
        """Fraction of training rows nearer the centroid than ``x`` (0 = typical, 1 = extreme)."""
        d = self.distance(x)
        n = len(self.ref_distances_sorted)
        return float(np.searchsorted(self.ref_distances_sorted, d) / n) if n else 0.0

    def is_ood(self, x: np.ndarray) -> bool:
        return self.distance(x) > self.threshold
