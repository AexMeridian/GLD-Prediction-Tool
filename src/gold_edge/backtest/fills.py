"""Human-delay fill simulator for backtesting.

CLAUDE.md: "a signal fills only if, at signal_time + delay (sample delay
uniformly from 1.0-2.0s, configurable), the book still offers the limit
price or better. Otherwise it's MISSED."

Reuses the exact same "has the book moved past this limit" predicate the
live engine uses to decide a pending signal has gone stale
(`engine.signals.price_moved_past_limit`), and scans every book snapshot
between signal creation and the fill instant — not just the one at the fill
instant — so a signal that was briefly invalidated and then recovered still
counts as MISSED. That mirrors the live state machine's "don't chase": once
a real book update would have caused `step()` to expire the signal, no
later reversion un-misses it.
"""

from __future__ import annotations

import random
from bisect import bisect_right
from datetime import datetime, timedelta

from gold_edge.config import BacktestConfig
from gold_edge.engine.signals import is_expired, price_moved_past_limit
from gold_edge.models import BookSnapshot, Fill, Signal


def sample_delay_s(cfg: BacktestConfig, rng: random.Random) -> float:
    return rng.uniform(cfg.human_delay_min_s, cfg.human_delay_max_s)


class BookHistory:
    """Book snapshots for one window, queried by time. All queries only
    look at snapshots that have already happened by time `t` — a backtest
    must never let a decision see the future."""

    def __init__(self, snapshots: list[BookSnapshot]) -> None:
        self._snapshots = sorted(snapshots, key=lambda b: b.receive_time)
        self._times = [b.receive_time for b in self._snapshots]

    def at_or_before(self, t: datetime) -> BookSnapshot | None:
        idx = bisect_right(self._times, t) - 1
        if idx < 0:
            return None
        return self._snapshots[idx]

    def between(self, start: datetime, end: datetime) -> list[BookSnapshot]:
        """Snapshots strictly after `start`, up to and including `end`."""
        lo = bisect_right(self._times, start)
        hi = bisect_right(self._times, end)
        return self._snapshots[lo:hi]


def simulate_fill(signal: Signal, book_history: BookHistory, delay_s: float) -> Fill | None:
    """Returns the Fill the human would have logged had they acted exactly
    `delay_s` after the signal was created, or None if it would have been
    MISSED (ttl expired first, or any intervening book update moved the
    price past the signal's limit)."""
    fill_time = signal.created_at + timedelta(seconds=delay_s)
    if is_expired(signal, fill_time):
        return None

    for book in book_history.between(signal.created_at, fill_time):
        if price_moved_past_limit(signal, book):
            return None

    if book_history.at_or_before(fill_time) is None:
        # No book data at all to confirm the price ever held — can't claim a fill.
        return None

    return Fill(
        signal_id=signal.id,
        window_ticker=signal.window_ticker,
        side=signal.side,
        action=signal.action,
        price=signal.limit_price,
        size=signal.size,
        logged_at=fill_time,
        is_skip=False,
    )
