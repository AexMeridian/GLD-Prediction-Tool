"""Learned component #3, per CLAUDE.md: "a regularized logistic regression
... predicting P(round trip is net profitable after learned delay and fees)
from the pattern features. Acts as an extra entry gate: enter only if
P >= FILTER_MIN_PROB. Must beat the no-filter engine on held-out data by
PROMOTION_MARGIN."

No sklearn in this project's dependencies (numpy/scipy only), so the
logistic regression is fit directly with `scipy.optimize.minimize` on an
L2-regularized log-loss objective -- a small, closed problem that doesn't
need a full ML library.

Features come from `learning/patterns.py`'s `RoundTripFeatures`, one-hot
encoded per dimension bucket (a small, fixed vocabulary, unlike raw
minute-of-window which is used as a numeric feature directly).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from gold_edge.learning.patterns import DIMENSIONS, RoundTripFeatures

# minute_of_window is numeric; every other RoundTripFeatures dimension is
# categorical and gets one-hot encoded.
_CATEGORICAL_DIMENSIONS = tuple(d for d in DIMENSIONS if d != "minute_of_window")


def _vocabulary(features: Sequence[RoundTripFeatures]) -> dict[str, list[str]]:
    vocab: dict[str, list[str]] = {}
    for dim in _CATEGORICAL_DIMENSIONS:
        values = sorted({f.value(dim) for f in features})
        vocab[dim] = values
    return vocab


def _feature_names(vocab: dict[str, list[str]]) -> list[str]:
    names = ["minute_of_window"]
    for dim in _CATEGORICAL_DIMENSIONS:
        names.extend(f"{dim}={v}" for v in vocab[dim])
    return names


def _encode_one(features: RoundTripFeatures, vocab: dict[str, list[str]]) -> list[float]:
    row = [float(features.minute_of_window)]
    for dim in _CATEGORICAL_DIMENSIONS:
        value = features.value(dim)
        row.extend(1.0 if v == value else 0.0 for v in vocab[dim])
    return row


def encode_features(
    features: Sequence[RoundTripFeatures], vocab: dict[str, list[str]] | None = None
) -> tuple[np.ndarray, dict[str, list[str]]]:
    """Builds the design matrix and (if not given) the vocabulary to encode
    with. Pass the TRAINING vocabulary back in when encoding held-out or
    live data, so a bucket value never seen in training simply encodes to
    all-zero for that dimension rather than growing new columns."""
    vocab = vocab or _vocabulary(features)
    matrix = np.array([_encode_one(f, vocab) for f in features], dtype=float)
    return matrix, vocab


@dataclass(frozen=True)
class TradeFilterModel:
    feature_names: list[str]
    vocab: dict[str, list[str]]
    weights: list[float]
    bias: float
    mean: list[float]
    std: list[float]

    def _standardize(self, x: np.ndarray) -> np.ndarray:
        std = np.where(np.array(self.std) == 0, 1.0, np.array(self.std))
        return (x - np.array(self.mean)) / std

    def predict_proba(self, features: RoundTripFeatures) -> float:
        row, _ = encode_features([features], self.vocab)
        x = self._standardize(row[0])
        z = float(np.dot(x, self.weights) + self.bias)
        return float(1.0 / (1.0 + np.exp(-z)))

    def predict_proba_many(self, features: Sequence[RoundTripFeatures]) -> list[float]:
        rows, _ = encode_features(features, self.vocab)
        x = self._standardize(rows)
        z = x @ np.array(self.weights) + self.bias
        return (1.0 / (1.0 + np.exp(-z))).tolist()


def _neg_log_likelihood_l2(
    params: np.ndarray, x: np.ndarray, y: np.ndarray, l2: float
) -> tuple[float, np.ndarray]:
    weights, bias = params[:-1], params[-1]
    z = x @ weights + bias
    z = np.clip(z, -30, 30)
    p = 1.0 / (1.0 + np.exp(-z))
    eps = 1e-9
    nll = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
    penalty = l2 * np.sum(weights**2) / len(y)
    loss = nll + penalty

    grad_z = p - y
    grad_w = (x.T @ grad_z) / len(y) + 2 * l2 * weights / len(y)
    grad_b = np.mean(grad_z)
    grad = np.concatenate([grad_w, [grad_b]])
    return float(loss), grad


def fit_trade_filter(
    features: Sequence[RoundTripFeatures], profitable: Sequence[bool], l2: float = 1.0
) -> TradeFilterModel:
    """`profitable[i]` is whether round trip i's realistic net P&L (after
    the learned delay profile and fees) was positive. Features are
    standardized before fitting so L2 regularization penalizes all
    dimensions comparably regardless of scale (minute_of_window vs. a 0/1
    one-hot column)."""
    if len(features) != len(profitable):
        raise ValueError("features and profitable must be the same length")
    if len(features) < 2:
        raise ValueError("need at least 2 training examples to fit a trade filter")

    raw_x, vocab = encode_features(features)
    y = np.array([1.0 if p else 0.0 for p in profitable])
    mean = raw_x.mean(axis=0)
    std = raw_x.std(axis=0)
    std_safe = np.where(std == 0, 1.0, std)
    x = (raw_x - mean) / std_safe

    n_features = x.shape[1]
    x0 = np.zeros(n_features + 1)
    result = minimize(_neg_log_likelihood_l2, x0, args=(x, y, l2), jac=True, method="L-BFGS-B")
    weights, bias = result.x[:-1], result.x[-1]

    return TradeFilterModel(
        feature_names=_feature_names(vocab),
        vocab=vocab,
        weights=weights.tolist(),
        bias=float(bias),
        mean=mean.tolist(),
        std=std.tolist(),
    )


def filter_beats_baseline(
    baseline_net_pnl: float, filtered_net_pnl: float, promotion_margin: float
) -> bool:
    """CLAUDE.md: "Must beat the no-filter engine on held-out data by
    PROMOTION_MARGIN." Both P&Ls should already be computed on the SAME
    held-out round trips (the filter applied as a post-hoc gate, keeping
    only round trips it would have allowed) -- this function just applies
    the margin test, not the P&L computation itself."""
    return filtered_net_pnl >= baseline_net_pnl + promotion_margin
