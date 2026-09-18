"""Builds Signal objects with limits + expiry, and the pure predicates the
state machine uses to decide when a pending signal is stale ("don't chase").
"""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal

from gold_edge.models import Action, BookSnapshot, Side, Signal


def build_signal(
    action: Action,
    side: Side,
    window_ticker: str,
    book: BookSnapshot,
    fair: float,
    size: Decimal,
    edge_after_costs: float,
    reason: str,
    now: datetime,
    ttl_s: float,
) -> Signal:
    """BUY limit = ask at signal time; SELL limit = bid at signal time."""
    limit_price = book.ask(side) if action is Action.BUY else book.bid(side)
    return Signal(
        window_ticker=window_ticker,
        action=action,
        side=side,
        limit_price=limit_price,
        size=size,
        fair=fair,
        market_price=limit_price,
        edge_after_costs=edge_after_costs,
        reason=reason,
        created_at=now,
        expires_at=now + timedelta(seconds=ttl_s),
    )


def is_expired(signal: Signal, now: datetime) -> bool:
    return now >= signal.expires_at


def price_moved_past_limit(signal: Signal, book: BookSnapshot) -> bool:
    """True once the market no longer offers the signal's limit price or
    better — a BUY has gotten more expensive, or a SELL's bid has dropped.
    Equal-or-better prices don't count, so a signal isn't invalidated by
    the book standing still or improving."""
    if signal.action is Action.BUY:
        return book.ask(signal.side) > signal.limit_price
    if signal.action is Action.SELL:
        return book.bid(signal.side) < signal.limit_price
    return False
