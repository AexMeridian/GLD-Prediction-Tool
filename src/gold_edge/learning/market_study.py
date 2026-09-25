"""Does the fair-value model beat the real Kalshi market on settled 15-minute
windows? The question that actually decides whether any edge exists.

Live recording captured real Kalshi books and settlements for days that have
no spot-price ticks (the price feeds had silently died). Yahoo's free GLD
1-minute bars can stand in for the missing price series during NYSE hours:
the model only uses ln(S/S0) and a per-minute vol, both scale-free, so GLD's
~$400 level vs spot's ~$4,400 doesn't matter -- only how closely GLD's
minute-to-minute moves track spot's. Two honesty rules follow:

- GLD is a proxy with its own noise. `proxy_agreement` measures how often the
  GLD-implied window outcome equals Kalshi's actual settlement; every result
  below is bounded by that number, and GLD ticks never enter the recorder's
  tables or the live/backtest paths (they are tagged and used only here).
- Timing is lookahead-safe: a signal at minute boundary T only uses GLD bars
  that closed by T, and the Kalshi quote it trades against is the first
  snapshot at or after T + delay (default 1.5s, the human-delay floor).
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import numpy as np

from gold_edge.config import FeesConfig, ModelConfig
from gold_edge.learning.historical_gold import PriceBar
from gold_edge.model.fair_value import compute_fair_value
from gold_edge.model.fees import taker_fee
from gold_edge.model.volatility import VolatilityTracker

WINDOW_MINUTES = 15
MIN_WARMUP_BARS = 10


@dataclass(frozen=True)
class WindowRef:
    ticker: str
    open_time: datetime
    close_time: datetime
    result: str  # "yes" or "no"


@dataclass(frozen=True)
class BookQuote:
    yes_bid: Decimal
    yes_ask: Decimal
    no_bid: Decimal
    no_ask: Decimal
    receive_time: datetime

    @property
    def yes_mid(self) -> float:
        return float((self.yes_bid + self.yes_ask) / 2)


@dataclass(frozen=True)
class StudyRow:
    ticker: str
    minute: int
    fair_yes: float
    quote: BookQuote
    outcome: float  # Kalshi's actual settlement, 1.0 = YES
    extras: dict[str, float] = field(default_factory=dict)  # signal-time features


class GldSeries:
    """GLD 1-minute closes. Yahoo stamps a bar with its START time, so the
    price known at boundary T is the close of the bar that started at T-60s."""

    def __init__(self, bars: Sequence[PriceBar], half_life_s: float, short_horizon_s: float):
        ordered = sorted(bars, key=lambda b: b.timestamp)
        self._closes: dict[datetime, float] = {}
        self._sigma: dict[datetime, float] = {}
        tracker: VolatilityTracker | None = None
        session_bars = 0
        prev_end: datetime | None = None
        for bar in ordered:
            end = bar.timestamp + timedelta(seconds=60)
            self._closes[end] = bar.close
            if prev_end is None or (end - prev_end) > timedelta(minutes=5):
                tracker = VolatilityTracker(half_life_s, short_horizon_s, 1e-6)
                session_bars = 0
            assert tracker is not None
            tracker.update(bar.close, end)
            session_bars += 1
            if session_bars >= MIN_WARMUP_BARS:
                self._sigma[end] = tracker.sigma_per_minute
            prev_end = end

    def price_at(self, t: datetime) -> float | None:
        return self._closes.get(t)

    def sigma_at(self, t: datetime) -> float | None:
        return self._sigma.get(t)


def proxy_agreement(windows: Sequence[WindowRef], gld: GldSeries) -> tuple[int, int]:
    """(windows comparable, windows where GLD's own up/down call matched
    Kalshi's actual settlement). Ties resolve YES, like the contract."""
    compared = agree = 0
    for w in windows:
        p0, p1 = gld.price_at(w.open_time), gld.price_at(w.close_time)
        if p0 is None or p1 is None:
            continue
        compared += 1
        gld_yes = p1 >= p0
        if gld_yes == (w.result == "yes"):
            agree += 1
    return compared, agree


def build_study_rows(
    windows: Sequence[WindowRef],
    gld: GldSeries,
    quote_at: Callable[[str, datetime], BookQuote | None],
    model_cfg: ModelConfig,
    delay_s: float = 1.5,
    extra_at: Callable[[WindowRef, int, datetime, float], dict[str, float]] | None = None,
) -> list[StudyRow]:
    """`extra_at(window, minute, boundary, fair_yes)` may return additional
    features known at the signal boundary (never anything from later)."""
    rows: list[StudyRow] = []
    for w in windows:
        s0 = gld.price_at(w.open_time)
        if s0 is None or gld.price_at(w.close_time) is None:
            continue
        outcome = 1.0 if w.result == "yes" else 0.0
        for m in range(1, WINDOW_MINUTES):
            t = w.open_time + timedelta(minutes=m)
            s, sigma = gld.price_at(t), gld.sigma_at(t)
            if s is None or sigma is None:
                continue
            quote = quote_at(w.ticker, t + timedelta(seconds=delay_s))
            if quote is None:
                continue
            fv = compute_fair_value(
                s, s0, sigma, float(WINDOW_MINUTES - m), model_cfg.min_fair_value,
                model_cfg.max_fair_value,
            )
            extras = extra_at(w, m, t, fv.yes) if extra_at else {}
            rows.append(StudyRow(w.ticker, m, fv.yes, quote, outcome, extras))
    return rows


@dataclass(frozen=True)
class BrierComparison:
    n_rows: int
    n_windows: int
    model_brier: float
    market_brier: float
    diff: float  # market - model; positive => model beat the market
    ci_low: float
    ci_high: float


def _cluster_bootstrap(
    by_window: dict[str, list[float]], n_boot: int, rng: random.Random
) -> tuple[float, float, float]:
    """Resamples whole windows (the independent units), pooled mean per draw."""
    keys = list(by_window)
    sums = np.array([sum(by_window[k]) for k in keys])
    counts = np.array([len(by_window[k]) for k in keys], dtype=float)
    gen = np.random.default_rng(rng.randrange(2**32))
    idx = gen.integers(0, len(keys), size=(n_boot, len(keys)))
    means = np.sort(sums[idx].sum(axis=1) / counts[idx].sum(axis=1))
    point = float(sums.sum() / counts.sum())
    return point, float(means[int(0.025 * n_boot)]), float(means[int(0.975 * n_boot) - 1])


def compare_brier(
    rows: Sequence[StudyRow], n_boot: int = 2000, seed: int = 0
) -> BrierComparison | None:
    if not rows:
        return None
    by_window: dict[str, list[float]] = {}
    model_sq = market_sq = 0.0
    for r in rows:
        m = (r.fair_yes - r.outcome) ** 2
        k = (r.quote.yes_mid - r.outcome) ** 2
        model_sq += m
        market_sq += k
        by_window.setdefault(r.ticker, []).append(k - m)
    point, lo, hi = _cluster_bootstrap(by_window, n_boot, random.Random(seed))
    n = len(rows)
    return BrierComparison(n, len(by_window), model_sq / n, market_sq / n, point, lo, hi)


@dataclass(frozen=True)
class HoldResult:
    threshold: float
    n_trades: int
    win_rate: float
    mean_pnl: float
    ci_low: float
    ci_high: float


def best_entry(
    fair_yes: float,
    yes_ask: Decimal,
    no_ask: Decimal,
    fees_cfg: FeesConfig,
    threshold: float,
) -> tuple[float, str, Decimal] | None:
    """The single entry rule shared by the study and the live shadow: buy the
    side with the larger gap = fair - ask - entry_fee, if it clears
    `threshold`. Returns (gap, "yes"|"no", ask) or None."""
    fee_mult = Decimal(str(fees_cfg.fee_multiplier))
    base = Decimal(str(fees_cfg.base_rate))
    best: tuple[float, str, Decimal] | None = None
    for side, fair, ask in (("yes", fair_yes, yes_ask), ("no", 1.0 - fair_yes, no_ask)):
        if not (Decimal("0") < ask < Decimal("1")):
            continue
        fee = taker_fee(Decimal(1), ask, fee_mult, base)
        gap = fair - float(ask + fee)
        if gap >= threshold and (best is None or gap > best[0]):
            best = (gap, side, ask)
    return best


def hold_to_settlement(
    rows: Sequence[StudyRow],
    fees_cfg: FeesConfig,
    threshold: float,
    n_boot: int = 2000,
    seed: int = 0,
) -> HoldResult:
    """One trade per window at most: the first minute where the best side's
    gap = fair - ask - entry_fee clears `threshold`; buy at the real ask,
    hold to Kalshi's actual settlement (no exit fee, no exit-timing risk)."""
    fee_mult = Decimal(str(fees_cfg.fee_multiplier))
    base = Decimal(str(fees_cfg.base_rate))
    pnls: dict[str, list[float]] = {}
    wins = 0
    for r in sorted(rows, key=lambda x: (x.ticker, x.minute)):
        if r.ticker in pnls:
            continue
        best = best_entry(r.fair_yes, r.quote.yes_ask, r.quote.no_ask, fees_cfg, threshold)
        if best is None:
            continue
        _, side, ask = best
        won = (side == "yes") == (r.outcome == 1.0)
        fee = taker_fee(Decimal(1), ask, fee_mult, base)
        pnl = (1.0 if won else 0.0) - float(ask + fee)
        wins += 1 if won else 0
        pnls[r.ticker] = [pnl]
    if not pnls:
        return HoldResult(threshold, 0, 0.0, 0.0, 0.0, 0.0)
    point, lo, hi = _cluster_bootstrap(pnls, n_boot, random.Random(seed))
    return HoldResult(threshold, len(pnls), wins / len(pnls), point, lo, hi)


def load_gld_cache(path: Path) -> list[PriceBar]:
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [PriceBar(datetime.fromisoformat(t), c) for t, c in raw]


def save_gld_cache(path: Path, bars: Sequence[PriceBar]) -> None:
    merged = {b.timestamp: b.close for b in load_gld_cache(path)}
    merged.update({b.timestamp: b.close for b in bars})
    path.write_text(
        json.dumps([[t.astimezone(UTC).isoformat(), c] for t, c in sorted(merged.items())]),
        encoding="utf-8",
    )


def format_study(
    label: str,
    compared: int,
    agree: int,
    brier: BrierComparison | None,
    holds: Sequence[HoldResult],
) -> str:
    lines = [f"=== {label} ==="]
    if compared:
        lines.append(
            f"GLD-vs-Kalshi outcome agreement: {agree}/{compared} windows "
            f"({agree / compared:.1%}) -- the ceiling on how far GLD can stand in for spot"
        )
    if brier is None:
        lines.append("no usable rows (need GLD minute bars overlapping settled windows)")
        return "\n".join(lines)
    verdict = (
        "model beat the market"
        if brier.ci_low > 0
        else "market beat the model"
        if brier.ci_high < 0
        else "no reliable difference"
    )
    lines += [
        f"rows={brier.n_rows}  windows={brier.n_windows} (independent outcomes = windows)",
        f"Brier  model={brier.model_brier:.4f}  market={brier.market_brier:.4f}  "
        f"market-minus-model={brier.diff:+.4f}  95% CI [{brier.ci_low:+.4f}, {brier.ci_high:+.4f}]"
        f"  -> {verdict}",
        "hold-to-settlement at the real ask, after entry fee (all thresholds shown, none picked):",
    ]
    for h in holds:
        if h.n_trades == 0:
            lines.append(f"  gap>={h.threshold:.2f}: no trades")
        else:
            lines.append(
                f"  gap>={h.threshold:.2f}: trades={h.n_trades}  win_rate={h.win_rate:.1%}  "
                f"mean_net_pnl/contract=${h.mean_pnl:+.3f}  "
                f"95% CI [${h.ci_low:+.3f}, ${h.ci_high:+.3f}]"
            )
    return "\n".join(lines)
