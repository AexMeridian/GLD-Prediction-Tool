"""Does the model add information the market price doesn't already have?

Kalshi's mid is itself a probability forecast. The honest test of "can we
predict movement" is whether a logistic blend of logit(market mid) and
logit(model fair value), fit only on EARLIER days, predicts LATER days better
than the market mid alone. If it can't, the model has nothing to sell; if it
can, the blend's held-out probabilities are what a trade filter should use.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import minimize

from gold_edge.learning.calibrator import brier_score, log_loss
from gold_edge.learning.market_study import StudyRow, _cluster_bootstrap

_EPS = 1e-4


def _logit(p: float) -> float:
    p = min(max(p, _EPS), 1 - _EPS)
    return math.log(p / (1 - p))


def _features(rows: Sequence[StudyRow]) -> np.ndarray:
    return np.array([[1.0, _logit(r.quote.yes_mid), _logit(r.fair_yes)] for r in rows])


def blend_probability(weights: Sequence[float], market_mid: float, fair_yes: float) -> float:
    """The one blend formula, used by training rows and the live shadow alike:
    p = sigmoid(w0 + w1*logit(market mid) + w2*logit(model fair))."""
    z = weights[0] + weights[1] * _logit(market_mid) + weights[2] * _logit(fair_yes)
    return 1.0 / (1.0 + math.exp(-z))


@dataclass(frozen=True)
class BlendModel:
    weights: tuple[float, float, float]  # intercept, w_market, w_model

    def predict(self, rows: Sequence[StudyRow]) -> list[float]:
        return [blend_probability(self.weights, r.quote.yes_mid, r.fair_yes) for r in rows]


def fit_blend(rows: Sequence[StudyRow], l2: float = 1.0) -> BlendModel:
    """L2-regularised logistic regression, regularised toward the identity
    blend (weight 1 on the market, 0 elsewhere) so with little data it
    degrades to "just trust the market" instead of overfitting."""
    x = _features(rows)
    y = np.array([r.outcome for r in rows])
    prior = np.array([0.0, 1.0, 0.0])

    def loss(w: np.ndarray) -> float:
        z = x @ w
        ll = np.sum(np.logaddexp(0, z) - y * z)
        return float(ll + l2 * np.sum((w - prior) ** 2))

    res = minimize(loss, prior, method="L-BFGS-B")
    return BlendModel(tuple(float(v) for v in res.x))  # type: ignore[arg-type]


@dataclass(frozen=True)
class WalkForwardBlend:
    n_days_scored: int
    n_rows: int
    n_windows: int
    market_log_loss: float
    blend_log_loss: float
    model_log_loss: float
    market_brier: float
    blend_brier: float
    diff_brier: float  # market - blend; positive => blend better
    ci_low: float
    ci_high: float
    last_weights: tuple[float, float, float]
    scored_rows: list[StudyRow]  # held-out rows with fair_yes replaced by the blend


def walk_forward_blend(
    rows: Sequence[StudyRow], min_train_windows: int = 200, n_boot: int = 1000, seed: int = 0
) -> WalkForwardBlend | None:
    """Expanding-window, by calendar day: fit on every strictly earlier day,
    score the day. A day is never in its own training set."""
    by_day: dict[str, list[StudyRow]] = {}
    for r in sorted(rows, key=lambda r: r.quote.receive_time):
        by_day.setdefault(r.quote.receive_time.date().isoformat(), []).append(r)

    train: list[StudyRow] = []
    scored: list[StudyRow] = []
    blend_p: list[float] = []
    weights: tuple[float, float, float] = (0.0, 1.0, 0.0)
    days = 0
    for day in sorted(by_day):
        day_rows = by_day[day]
        if len({r.ticker for r in train}) >= min_train_windows:
            model = fit_blend(train)
            weights = model.weights
            p = model.predict(day_rows)
            scored.extend(replace(r, fair_yes=pp) for r, pp in zip(day_rows, p, strict=True))
            blend_p.extend(p)
            days += 1
        train.extend(day_rows)
    if not scored:
        return None

    y = [r.outcome for r in scored]
    mids = [r.quote.yes_mid for r in scored]
    # scored[i].fair_yes was overwritten by the blend, so recover the raw
    # model probability from the originals in the same order.
    raw = {(r.ticker, r.minute): r.fair_yes for r in rows}
    model_p = [raw[(r.ticker, r.minute)] for r in scored]

    by_window: dict[str, list[float]] = {}
    for r, m in zip(scored, mids, strict=True):
        diff = (m - r.outcome) ** 2 - (r.fair_yes - r.outcome) ** 2
        by_window.setdefault(r.ticker, []).append(diff)
    point, lo, hi = _cluster_bootstrap(by_window, n_boot, random.Random(seed))
    return WalkForwardBlend(
        n_days_scored=days,
        n_rows=len(scored),
        n_windows=len(by_window),
        market_log_loss=log_loss(mids, y),
        blend_log_loss=log_loss(blend_p, y),
        model_log_loss=log_loss(model_p, y),
        market_brier=brier_score(mids, y),
        blend_brier=brier_score(blend_p, y),
        diff_brier=point,
        ci_low=lo,
        ci_high=hi,
        last_weights=weights,
        scored_rows=scored,
    )


def format_blend(w: WalkForwardBlend | None) -> str:
    if w is None:
        return "walk-forward blend: not enough history to train and score yet"
    verdict = (
        "blend beat the market out-of-sample"
        if w.ci_low > 0
        else "market beat the blend"
        if w.ci_high < 0
        else "no reliable difference vs. the market"
    )
    a, b, c = w.last_weights
    return "\n".join(
        [
            f"walk-forward (fit on earlier days, score the next): {w.n_days_scored} days, "
            f"{w.n_windows} windows, {w.n_rows} rows -- all held-out",
            f"log-loss  market={w.market_log_loss:.4f}  blend={w.blend_log_loss:.4f}  "
            f"model-alone={w.model_log_loss:.4f}",
            f"Brier     market={w.market_brier:.4f}  blend={w.blend_brier:.4f}  "
            f"market-minus-blend={w.diff_brier:+.5f}  95% CI [{w.ci_low:+.5f}, {w.ci_high:+.5f}]"
            f"  -> {verdict}",
            f"latest fitted blend: logit(p) = {a:+.3f} + {b:.3f}*logit(market) + "
            f"{c:+.3f}*logit(model)",
        ]
    )
