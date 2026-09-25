"""A richer market+model blend, with a strict protocol against fooling ourselves.

Base blend: logit(p) = w0 + w1*logit(market mid) + w2*logit(model). Candidate
extras (all known at the signal boundary, never later): how far into the
window we are, market momentum, spread, activity, vol at other time scales,
time of day. Each set is scored by expanding walk-forward on the DEV period
only; a final HOLDOUT (the most recent days) is scored once, after the choice
is made. Regularisation pulls the fit toward "trust the market", so more
features cannot silently turn into overfitting.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import minimize

from gold_edge.learning.blend import _logit
from gold_edge.learning.calibrator import brier_score, log_loss
from gold_edge.learning.market_study import StudyRow, _cluster_bootstrap

BASE = ("logit_mid", "logit_fair")
WINDOW = 15.0


def feature_value(row: StudyRow, name: str) -> float:
    lm, lf = _logit(row.quote.yes_mid), _logit(row.fair_yes)
    if name == "logit_mid":
        return lm
    if name == "logit_fair":
        return lf
    if name == "fair_x_late":  # model matters more as the window runs out
        return lf * row.minute / WINDOW
    if name == "mid_x_late":
        return lm * row.minute / WINDOW
    if name == "spread":
        return float(row.quote.yes_ask - row.quote.yes_bid)
    return row.extras.get(name, 0.0)


@dataclass(frozen=True)
class FeatureBlend:
    names: tuple[str, ...]
    weights: tuple[float, ...]  # intercept first
    means: tuple[float, ...]
    scales: tuple[float, ...]

    def _x(self, rows: Sequence[StudyRow]) -> np.ndarray:
        raw = np.array([[feature_value(r, n) for n in self.names] for r in rows])
        x = (raw - np.array(self.means)) / np.array(self.scales)
        return np.hstack([np.ones((len(rows), 1)), x])

    def predict(self, rows: Sequence[StudyRow]) -> list[float]:
        z = self._x(rows) @ np.array(self.weights)
        return [float(1 / (1 + math.exp(-min(max(v, -30), 30)))) for v in z]


def fit_feature_blend(
    rows: Sequence[StudyRow],
    names: Sequence[str],
    l2_base: float = 1.0,
    l2_extra: float = 5.0,
) -> FeatureBlend:
    """Base features stay unstandardised (so the identity prior "weight 1 on
    the market" means what it says); extras are standardised and shrunk to 0."""
    names = tuple(names)
    raw = np.array([[feature_value(r, n) for n in names] for r in rows])
    means = np.array([0.0 if n in BASE else raw[:, i].mean() for i, n in enumerate(names)])
    scales = np.array(
        [1.0 if n in BASE else (raw[:, i].std() or 1.0) for i, n in enumerate(names)]
    )
    x = np.hstack([np.ones((len(rows), 1)), (raw - means) / scales])
    y = np.array([r.outcome for r in rows])
    prior = np.array([0.0] + [1.0 if n == "logit_mid" else 0.0 for n in names])
    pen = np.array([0.0] + [l2_base if n in BASE else l2_extra for n in names])

    def loss(w: np.ndarray) -> float:
        z = x @ w
        return float(np.sum(np.logaddexp(0, z) - y * z) + np.sum(pen * (w - prior) ** 2))

    res = minimize(loss, prior, method="L-BFGS-B")
    return FeatureBlend(names, tuple(float(v) for v in res.x), tuple(means), tuple(scales))


@dataclass(frozen=True)
class Scored:
    names: tuple[str, ...]
    n_rows: int
    n_windows: int
    market_log_loss: float
    log_loss: float
    market_brier: float
    brier: float
    diff_vs_market: float  # market - model Brier; positive => better than market
    ci_low: float
    ci_high: float
    rows: list[StudyRow]  # held-out rows, fair_yes replaced by the blend probability


def _score(names: Sequence[str], scored: list[StudyRow], n_boot: int, seed: int) -> Scored | None:
    if not scored:
        return None
    y = [r.outcome for r in scored]
    mids = [r.quote.yes_mid for r in scored]
    ps = [r.fair_yes for r in scored]
    by_window: dict[str, list[float]] = {}
    for r in scored:
        by_window.setdefault(r.ticker, []).append(
            (r.quote.yes_mid - r.outcome) ** 2 - (r.fair_yes - r.outcome) ** 2
        )
    point, lo, hi = _cluster_bootstrap(by_window, n_boot, random.Random(seed))
    return Scored(
        tuple(names), len(scored), len(by_window), log_loss(mids, y), log_loss(ps, y),
        brier_score(mids, y), brier_score(ps, y), point, lo, hi, scored,
    )


def _by_day(rows: Sequence[StudyRow]) -> dict[str, list[StudyRow]]:
    out: dict[str, list[StudyRow]] = {}
    for r in sorted(rows, key=lambda r: r.quote.receive_time):
        out.setdefault(r.quote.receive_time.date().isoformat(), []).append(r)
    return out


def walk_forward(
    rows: Sequence[StudyRow],
    names: Sequence[str],
    min_train_windows: int = 300,
    n_boot: int = 500,
    seed: int = 0,
) -> Scored | None:
    """Fit on every strictly earlier day, score the next day."""
    days = _by_day(rows)
    train: list[StudyRow] = []
    scored: list[StudyRow] = []
    for day in sorted(days):
        if len({r.ticker for r in train}) >= min_train_windows:
            model = fit_feature_blend(train, names)
            preds = model.predict(days[day])
            scored.extend(replace(r, fair_yes=p) for r, p in zip(days[day], preds, strict=True))
        train.extend(days[day])
    return _score(names, scored, n_boot, seed)


def fit_and_score(
    dev: Sequence[StudyRow], holdout: Sequence[StudyRow], names: Sequence[str],
    n_boot: int = 1000, seed: int = 0,
) -> Scored | None:
    """Fit once on all of `dev`, score `holdout` -- the one-shot final test."""
    model = fit_feature_blend(dev, names)
    p = model.predict(holdout)
    return _score(names, [replace(r, fair_yes=q) for r, q in zip(holdout, p, strict=True)],
                  n_boot, seed)


def paired_diff(
    a: Scored, b: Scored, n_boot: int = 1000, seed: int = 0
) -> tuple[float, float, float]:
    """Brier(a) - Brier(b) over the same rows, resampling windows: positive
    means b is better than a."""
    key = {(r.ticker, r.minute): r for r in b.rows}
    by_window: dict[str, list[float]] = {}
    for r in a.rows:
        other = key.get((r.ticker, r.minute))
        if other is None:
            continue
        by_window.setdefault(r.ticker, []).append(
            (r.fair_yes - r.outcome) ** 2 - (other.fair_yes - other.outcome) ** 2
        )
    if not by_window:
        return 0.0, 0.0, 0.0
    return _cluster_bootstrap(by_window, n_boot, random.Random(seed))


def split_dev_holdout(
    rows: Sequence[StudyRow], holdout_days: int
) -> tuple[list[StudyRow], list[StudyRow]]:
    days = _by_day(rows)
    order = sorted(days)
    cut = set(order[-holdout_days:]) if holdout_days else set()
    dev = [r for d in order if d not in cut for r in days[d]]
    hold = [r for d in order if d in cut for r in days[d]]
    return dev, hold
