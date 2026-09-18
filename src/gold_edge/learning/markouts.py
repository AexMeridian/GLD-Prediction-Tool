"""Markouts: for a signal, what did the market and model look like at
+5/+15/+30/+60s afterward (config: `learning.markout_horizons_s`)?

Per CLAUDE.md: "Positive markout = the market moved toward our fair value
[after the signal] -> edge was real." This is the foundation of decision
quality grading (learning/grader.py) — it judges whether the read was right
at the moment of the signal, independent of what we actually did about it
(execution quality, fees, or how the trade was exited are separate
questions). Applies uniformly to every signal, taken or not: CLAUDE.md is
explicit that markouts are computed "for every signal (taken, skipped, or
missed)."

`edge_markout` uses `signal.market_price` (the ask for a BUY, the bid for a
SELL, at signal time) as the reference and flips sign for SELL, so a
positive value always means "the market moved to validate this signal's
action" regardless of side: for a BUY that's the price rising past what we
paid; for a SELL (an exit, often a stop) that's the price continuing to
fall past what we sold at.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

from gold_edge.backtest.fills import BookHistory
from gold_edge.config import ModelConfig, VolatilityConfig
from gold_edge.model.fair_value import compute_fair_value
from gold_edge.model.volatility import VolatilityTracker
from gold_edge.models import Action, Side, Signal, Tick, Window


@dataclass(frozen=True)
class Markout:
    horizon_s: float
    at: datetime
    pyth_price: float | None
    market_mid: float | None
    market_bid: float | None
    market_ask: float | None
    fair: float | None
    edge_markout: float | None


def compute_markouts(
    signal: Signal,
    window: Window,
    ticks: list[Tick],
    books: BookHistory,
    horizons_s: Sequence[float],
    model_cfg: ModelConfig,
    vol_cfg: VolatilityConfig,
) -> list[Markout]:
    """One Markout per entry in `horizons_s`, in the same order given. The
    book/price feed is replayed with a fresh VolatilityTracker seeded from
    `ticks` up through each horizon instant — markouts are a post-hoc
    diagnostic, not the live trading decision, so this doesn't need to
    match whatever tracker state happened to be live at the time."""
    sorted_ticks = sorted(ticks, key=lambda t: t.receive_time)
    vol_tracker = VolatilityTracker(
        half_life_s=vol_cfg.ewma_half_life_s,
        short_horizon_s=vol_cfg.short_horizon_s,
        min_sigma_per_minute=vol_cfg.min_sigma_per_minute,
    )
    sign = 1.0 if signal.action is Action.BUY else -1.0
    reference_price = float(signal.market_price)

    computed: dict[float, Markout] = {}
    tick_idx = 0
    latest_price: float | None = None

    for horizon_s in sorted(set(horizons_s)):
        at = signal.created_at + timedelta(seconds=horizon_s)

        while tick_idx < len(sorted_ticks) and sorted_ticks[tick_idx].receive_time <= at:
            t = sorted_ticks[tick_idx]
            vol_tracker.update(t.price, t.publish_time)
            latest_price = t.price
            tick_idx += 1

        # No market exists to reference once the window has closed and
        # settled — Kalshi stops trading it. The underlying reference price
        # and model fair value are still meaningful after close, though.
        book = None if at >= window.close_time else books.at_or_before(at)
        market_bid = market_ask = market_mid = None
        if book is not None:
            bid_dec = book.bid(signal.side)
            ask_dec = book.ask(signal.side)
            market_bid = float(bid_dec)
            market_ask = float(ask_dec)
            market_mid = float((bid_dec + ask_dec) / 2)

        fair_value = None
        if latest_price is not None and window.s0 is not None:
            tau_minutes = window.seconds_left(at) / 60.0
            fv = compute_fair_value(
                latest_price,
                float(window.s0),
                vol_tracker.sigma_per_minute,
                tau_minutes,
                model_cfg.min_fair_value,
                model_cfg.max_fair_value,
            )
            fair_value = fv.yes if signal.side is Side.YES else fv.no

        edge_markout = sign * (market_mid - reference_price) if market_mid is not None else None

        computed[horizon_s] = Markout(
            horizon_s=horizon_s,
            at=at,
            pyth_price=latest_price,
            market_mid=market_mid,
            market_bid=market_bid,
            market_ask=market_ask,
            fair=fair_value,
            edge_markout=edge_markout,
        )

    return [computed[h] for h in horizons_s]
