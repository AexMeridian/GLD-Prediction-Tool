"""Core data models shared across feeds, engine, recorder, and backtest.

Prices and sizes use Decimal (Kalshi quotes contracts down to fractional
sizes and cent/sub-cent prices); floats are reserved for the underlying
Pyth price and model math (fair value, volatility), per CLAUDE.md.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, Field


class Side(StrEnum):
    YES = "YES"
    NO = "NO"


class Action(StrEnum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class PositionState(StrEnum):
    FLAT = "FLAT"
    LONG_YES = "LONG_YES"
    LONG_NO = "LONG_NO"


class SignalStatus(StrEnum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    SKIPPED = "SKIPPED"
    MISSED = "MISSED"
    EXPIRED = "EXPIRED"


class Tick(BaseModel):
    """One Pyth price update."""

    symbol: str
    price: float
    conf: float
    expo: int
    publish_time: datetime
    receive_time: datetime


class BookSnapshot(BaseModel):
    """Best bid/ask for both sides of a Kalshi market at a point in time."""

    window_ticker: str
    yes_bid: Decimal
    yes_ask: Decimal
    yes_bid_size: Decimal
    yes_ask_size: Decimal
    no_bid: Decimal
    no_ask: Decimal
    no_bid_size: Decimal
    no_ask_size: Decimal
    receive_time: datetime

    def spread(self, side: Side) -> Decimal:
        if side is Side.YES:
            return self.yes_ask - self.yes_bid
        return self.no_ask - self.no_bid

    def bid(self, side: Side) -> Decimal:
        return self.yes_bid if side is Side.YES else self.no_bid

    def ask(self, side: Side) -> Decimal:
        return self.yes_ask if side is Side.YES else self.no_ask


class Window(BaseModel):
    """One KXGOLD15M market (a single 15-minute window)."""

    ticker: str
    event_ticker: str
    series_ticker: str
    open_time: datetime
    close_time: datetime
    s0: Decimal | None = None
    status: str

    def seconds_left(self, now: datetime) -> float:
        return max(0.0, (self.close_time - now).total_seconds())


class Signal(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    window_ticker: str
    action: Action
    side: Side
    limit_price: Decimal
    size: Decimal
    fair: float
    market_price: Decimal
    edge_after_costs: float
    reason: str
    created_at: datetime
    expires_at: datetime
    status: SignalStatus = SignalStatus.PENDING


class Position(BaseModel):
    window_ticker: str
    side: Side
    size: Decimal
    entry_price: Decimal
    entered_at: datetime
    state: PositionState


class Fill(BaseModel):
    signal_id: str | None
    window_ticker: str
    side: Side
    action: Action
    price: Decimal
    size: Decimal
    logged_at: datetime
    is_skip: bool = False
