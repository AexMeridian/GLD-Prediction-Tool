"""Learned components #1 and #2, per CLAUDE.md:

1. Fair-value calibrator: isotonic regression mapping raw `fair_yes` to a
   calibrated probability, fit on settled windows and on markout-implied
   values. "Used only if it improves held-out log-loss/Brier" -- this
   module computes both metrics and leaves the promotion decision to the
   caller (`should_promote_calibrator`), never applying itself silently.

2. Volatility scaling: a multiplier on sigma by vol regime and time of day
   that minimizes held-out log-loss, evaluated the same way.

Both are supervised on (raw_probability, target) pairs where the target is
either a settled outcome (0/1) or a markout-implied probability (a later
market_mid, itself an estimate of P(YES) once prices ~= probabilities) --
CLAUDE.md explicitly allows fitting on either source, and isotonic
regression works the same way for a continuous target as for binary.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import isotonic_regression


@dataclass(frozen=True)
class IsotonicCalibrator:
    """A monotonic increasing step function fit on training data; `predict`
    linearly interpolates between the fitted training points and clamps at
    the ends, so it can be evaluated on any raw probability in [0, 1], not
    just ones seen during training."""

    x_thresholds: list[float]
    y_values: list[float]

    def predict(self, raw_probability: float) -> float:
        return float(np.interp(raw_probability, self.x_thresholds, self.y_values))

    def predict_many(self, raw_probabilities: Sequence[float]) -> list[float]:
        return [self.predict(p) for p in raw_probabilities]


def fit_isotonic_calibrator(
    raw_probabilities: Sequence[float], targets: Sequence[float]
) -> IsotonicCalibrator:
    """`targets` may be 0/1 settlement outcomes or markout-implied
    probabilities (a later market_mid) -- both are valid supervision for
    "does raw_fair systematically over/understate the true probability."""
    if len(raw_probabilities) != len(targets):
        raise ValueError("raw_probabilities and targets must be the same length")
    if not raw_probabilities:
        raise ValueError("need at least one training pair to fit a calibrator")

    x = np.asarray(raw_probabilities, dtype=float)
    y = np.asarray(targets, dtype=float)
    order = np.argsort(x, kind="stable")
    x_sorted = x[order]
    y_sorted = y[order]

    result = isotonic_regression(y_sorted, increasing=True)
    fitted = np.clip(result.x, 0.0, 1.0)

    # np.interp requires strictly non-decreasing x for well-defined lookups;
    # duplicate raw_probability values are fine (isotonic already averaged
    # their targets via pool-adjacent-violators), interp just uses the last
    # matching point, which is consistent since PAVA makes ties equal anyway.
    return IsotonicCalibrator(x_thresholds=x_sorted.tolist(), y_values=fitted.tolist())


def brier_score(probabilities: Sequence[float], outcomes: Sequence[float]) -> float:
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    return float(np.mean((p - y) ** 2))


def log_loss(probabilities: Sequence[float], outcomes: Sequence[float], eps: float = 1e-6) -> float:
    p = np.clip(np.asarray(probabilities, dtype=float), eps, 1 - eps)
    y = np.asarray(outcomes, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def should_promote_calibrator(
    raw_probabilities: Sequence[float],
    calibrated_probabilities: Sequence[float],
    outcomes: Sequence[float],
) -> bool:
    """CLAUDE.md: "Used only if it improves held-out log-loss/Brier."
    Requires improvement on BOTH metrics -- a calibrator that trades one off
    against the other isn't an unambiguous win, and per CLAUDE.md's
    anti-overfitting stance, ties should default to keeping the simpler
    (raw) model."""
    raw_ll = log_loss(raw_probabilities, outcomes)
    cal_ll = log_loss(calibrated_probabilities, outcomes)
    raw_brier = brier_score(raw_probabilities, outcomes)
    cal_brier = brier_score(calibrated_probabilities, outcomes)
    return cal_ll < raw_ll and cal_brier < raw_brier


@dataclass(frozen=True)
class VolMultiplierTable:
    """A multiplier on sigma, keyed by (vol_regime, session) -- the same
    bucket labels learning/patterns.py uses, so a live lookup and a pattern
    report always agree on what bucket a moment falls into. Falls back to
    1.0 (no scaling) for a combination that was never seen or never had
    enough held-out data to fit."""

    multipliers: dict[tuple[str, str], float]

    def get(self, vol_regime: str, session: str) -> float:
        return self.multipliers.get((vol_regime, session), 1.0)


DEFAULT_MULTIPLIER_GRID: tuple[float, ...] = (0.7, 0.85, 1.0, 1.15, 1.3, 1.5)


def fit_vol_multipliers(
    bucketed: dict[tuple[str, str], list[tuple[float, float, float]]],
    min_bucket_n: int,
    grid: Sequence[float] = DEFAULT_MULTIPLIER_GRID,
) -> VolMultiplierTable:
    """`bucketed` maps (vol_regime, session) -> list of (sigma, standardized_z,
    outcome) triples, where `standardized_z` is whatever the fair-value
    model's z-score input was (ln(S/S0) / (sigma * sqrt(tau))) BEFORE
    scaling -- multiplying the candidate factor back onto that z, then
    running it through the normal CDF, is how each grid candidate's implied
    fair_yes is reconstructed without re-deriving the whole model here.
    Picks, per bucket with n >= min_bucket_n, the multiplier minimizing
    held-out log-loss; buckets without enough data are omitted (get()
    then falls back to 1.0)."""
    from scipy.stats import norm

    multipliers: dict[tuple[str, str], float] = {}
    for bucket_key, triples in bucketed.items():
        if len(triples) < min_bucket_n:
            continue
        z_values = np.array([z for _, z, _ in triples])
        outcomes = np.array([o for _, _, o in triples])
        best_factor = 1.0
        best_ll = float("inf")
        for factor in grid:
            implied_fair = norm.cdf(z_values / factor) if factor > 0 else norm.cdf(z_values)
            ll = log_loss(implied_fair.tolist(), outcomes.tolist())
            if ll < best_ll:
                best_ll = ll
                best_factor = factor
        multipliers[bucket_key] = best_factor
    return VolMultiplierTable(multipliers=multipliers)
