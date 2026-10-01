"""A second, parallel shadow-mode track built to answer one question: "how
much more trade volume would the REAL entry/exit/flip/cooldown state machine
(`engine/state_machine.py`) generate per calendar day, compared to
`blend_shadow.py`'s simplified one-trade-per-window rule?" CLAUDE.md:
"Multiple round trips per window are expected," and both backtesting and
shadow testing are supposed to "replay... through the exact same engine code
(no duplicated strategy logic)" -- `blend_shadow.py` predates that principle
being applied to live forward collection; this module applies it.

Writes to its OWN sqlite file (`data/shadow_full.sqlite`) -- `blend_shadow.py`
and its already-running `data/shadow.sqlite` are never touched, imported, or
depended on by anything here. This is purely additive data collection.

Known, deliberate simplifications vs. the live product (documented, not
hidden, and distinct from `blend_shadow.py`'s own caveats):
- Minute resolution only: this polls Kalshi's public market listing and a
  free gold-proxy price once a minute (same cadence as `blend_shadow.py`),
  so `step()` is only EVALUATED once a minute, not continuously the way the
  live Pyth-tick-driven engine is. A real exit could fire a few seconds
  earlier or later than this will ever observe.
- Every BUY/SELL signal is synthetically "filled" ~1.5s after being issued,
  at its own limit price -- there is no human here to click, and no learned
  delay distribution is applied. This measures how much volume the real
  state machine would produce, not realistic human-paced P&L -- its P&L
  numbers are NOT directly comparable to `blend_shadow.py`'s or to live
  performance without accounting for this.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path

from gold_edge.config import EngineConfig, FeesConfig, ModelConfig
from gold_edge.engine.state_machine import EngineState, MarketSnapshot, apply_fill, step
from gold_edge.learning.blend import blend_probability
from gold_edge.learning.blend_shadow import BlendArtifact, PriceState
from gold_edge.model.fair_value import FairValue, compute_fair_value
from gold_edge.model.fees import settle_position_pnl
from gold_edge.models import Action, BookSnapshot, Fill, Window

WINDOW_MINUTES = 15
MAX_SPREAD = Decimal("0.10")
FILL_DELAY_S = 1.5  # matches market_study.py's/blend_shadow.py's fetch-delay convention

SCHEMA = """
CREATE TABLE IF NOT EXISTS shadow_full_round_trips (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    side TEXT NOT NULL,
    entry_minute INTEGER NOT NULL,
    exit_minute INTEGER,
    entry_price REAL NOT NULL,
    exit_price REAL,
    entry_ts TEXT NOT NULL,
    exit_ts TEXT,
    exit_reason TEXT,
    pnl REAL,
    model_hash TEXT
);
CREATE INDEX IF NOT EXISTS idx_shadow_full_ticker ON shadow_full_round_trips(ticker);
"""


def _dec(market: dict, key: str) -> Decimal:
    try:
        return Decimal(str(market[key]))
    except (KeyError, InvalidOperation, TypeError):
        return Decimal("0")


def market_to_book_snapshot(market: dict, now: datetime) -> BookSnapshot | None:
    """Kalshi's public single-market payload -> a full `BookSnapshot` (the
    real engine's input type), including top-of-book sizes when present.
    None when the book is empty/crossed/too wide to trade, same filter
    `blend_shadow.parse_market_quote` uses."""
    try:
        yb = Decimal(str(market["yes_bid_dollars"]))
        ya = Decimal(str(market["yes_ask_dollars"]))
    except (KeyError, InvalidOperation, TypeError):
        return None
    if not (Decimal("0") < yb < ya < Decimal("1")) or ya - yb > MAX_SPREAD:
        return None
    yes_bid_size, yes_ask_size = _dec(market, "yes_bid_size_fp"), _dec(market, "yes_ask_size_fp")
    return BookSnapshot(
        window_ticker=market.get("ticker", ""),
        yes_bid=yb,
        yes_ask=ya,
        yes_bid_size=yes_bid_size,
        yes_ask_size=yes_ask_size,
        no_bid=Decimal(1) - ya,
        no_ask=Decimal(1) - yb,
        # Kalshi's public listing only reports YES-side sizes; buying NO at
        # price p is the same resting liquidity as selling YES at 1-p, so
        # the complementary sizes mirror.
        no_bid_size=yes_ask_size,
        no_ask_size=yes_bid_size,
        receive_time=now,
    )


@dataclass
class FullShadowWindow:
    ticker: str
    open_time: datetime
    close_time: datetime
    s0_proxy: float | None = None


class FullShadowStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.executescript(SCHEMA)

    def open_round_trip(
        self, ticker: str, side: str, entry_minute: int, entry_price: float,
        entry_ts: str, model_hash: str | None,
    ) -> int:
        cur = self.conn.execute(
            "INSERT INTO shadow_full_round_trips "
            "(ticker, side, entry_minute, entry_price, entry_ts, model_hash) "
            "VALUES (?,?,?,?,?,?)",
            (ticker, side, entry_minute, entry_price, entry_ts, model_hash),
        )
        self.conn.commit()
        return cur.lastrowid

    def close_round_trip(
        self, row_id: int, exit_minute: int | None, exit_price: float | None,
        exit_ts: str | None, exit_reason: str, pnl: float,
    ) -> None:
        self.conn.execute(
            "UPDATE shadow_full_round_trips SET exit_minute=?, exit_price=?, exit_ts=?, "
            "exit_reason=?, pnl=? WHERE id=?",
            (exit_minute, exit_price, exit_ts, exit_reason, pnl, row_id),
        )
        self.conn.commit()

    def open_row_ids(self) -> dict[str, int]:
        """ticker -> row id, for round trips still missing a pnl (either
        still open or awaiting window settlement)."""
        return dict(
            self.conn.execute(
                "SELECT ticker, id FROM shadow_full_round_trips WHERE pnl IS NULL"
            ).fetchall()
        )


@dataclass
class FullShadowEngine:
    """Owns one `engine.state_machine.EngineState` PER TICKER (a window can
    close with a new one opening seconds later; each gets its own state,
    exactly like the live server resets per-window state at rollover)."""

    artifact: BlendArtifact
    model_cfg: ModelConfig
    fees_cfg: FeesConfig
    engine_cfg: EngineConfig
    vol_spike_limit: float
    prices: PriceState
    store: FullShadowStore
    states: dict[str, EngineState] = field(default_factory=dict)
    open_rows: dict[str, int] = field(default_factory=dict)  # ticker -> open round-trip row id

    def on_boundary(
        self, w: FullShadowWindow, boundary: datetime, quote: BookSnapshot | None
    ) -> list[str]:
        """Handle one minute boundary of one window; may return several log
        lines (e.g. an exit immediately followed by a flip-entry). The
        caller must already have called `prices.advance(boundary)`."""
        msgs: list[str] = []
        price = self.prices.price_at(boundary)
        if w.s0_proxy is None:
            w.s0_proxy = self.prices.price_at(w.open_time)
        minute = int((boundary - w.open_time).total_seconds() // 60)
        if minute == 0 or not (1 <= minute < WINDOW_MINUTES):
            return msgs
        sigma = self.prices.sigma
        if w.s0_proxy is None or price is None or sigma is None or quote is None:
            return msgs

        raw_fair = compute_fair_value(
            price, w.s0_proxy, sigma, float(WINDOW_MINUTES - minute),
            self.model_cfg.min_fair_value, self.model_cfg.max_fair_value,
        )
        market_mid = float((quote.yes_bid + quote.yes_ask) / 2)
        p = blend_probability(self.artifact.weights, market_mid, raw_fair.yes)
        if self.artifact.calibrator is not None:
            p = self.artifact.calibrator.predict(p)
        fair = FairValue(yes=p, no=1.0 - p)

        window_model = Window(
            ticker=w.ticker, event_ticker=w.ticker, series_ticker="KXGOLD15M",
            open_time=w.open_time, close_time=w.close_time, status="open",
        )
        snapshot = MarketSnapshot(
            now=boundary, window=window_model, book=quote, fair=fair,
            pyth_age_s=0.0, kalshi_age_s=0.0,
            short_horizon_sigma_per_minute=self.prices.short_horizon_sigma,
            underlying_price=price,
        )
        state = self.states.get(w.ticker, EngineState())
        result = step(state, snapshot, self.engine_cfg, self.fees_cfg, self.vol_spike_limit)
        state = result.state

        if result.signal is not None:
            sig = result.signal
            fill = Fill(
                signal_id=sig.id, window_ticker=sig.window_ticker, side=sig.side,
                action=sig.action, price=sig.limit_price, size=sig.size,
                logged_at=boundary + timedelta(seconds=FILL_DELAY_S),
            )
            if sig.action is Action.BUY:
                row_id = self.store.open_round_trip(
                    w.ticker, sig.side.value, minute, float(sig.limit_price),
                    boundary.isoformat(), self.artifact.version_hash,
                )
                self.open_rows[w.ticker] = row_id
                state = apply_fill(state, fill, self.engine_cfg, self.fees_cfg)
                msgs.append(
                    f"FULL-SHADOW ENTER {sig.side.value} {w.ticker} min {minute} "
                    f"@ {sig.limit_price} ({sig.reason})"
                )
            else:
                before_pnl = state.realized_pnl_today
                state = apply_fill(state, fill, self.engine_cfg, self.fees_cfg)
                round_trip_pnl = float(state.realized_pnl_today - before_pnl)
                row_id = self.open_rows.pop(w.ticker, None)
                if row_id is not None:
                    self.store.close_round_trip(
                        row_id, minute, float(sig.limit_price), boundary.isoformat(),
                        sig.reason, round_trip_pnl,
                    )
                msgs.append(
                    f"FULL-SHADOW EXIT {sig.side.value} {w.ticker} min {minute} "
                    f"@ {sig.limit_price} ({sig.reason}) pnl={round_trip_pnl:+.3f}"
                )
        self.states[w.ticker] = state
        return msgs

    def settle_if_still_open(self, ticker: str, result: str) -> str | None:
        """Mirrors `server.py::_handle_window_closed`'s reconciliation: a
        position still open when the window closed settles at Kalshi's
        actual result instead of being silently dropped. Returns a log line,
        or None if there was nothing open for this ticker."""
        state = self.states.get(ticker)
        row_id = self.open_rows.pop(ticker, None)
        if state is None or state.position is None or row_id is None:
            return None
        position = state.position
        pnl = settle_position_pnl(position, result, self.fees_cfg)
        self.store.close_round_trip(
            row_id, None, None, None, f"settled:{result}", float(pnl)
        )
        self.states[ticker] = EngineState(realized_pnl_today=state.realized_pnl_today + pnl)
        return (
            f"FULL-SHADOW SETTLE {position.side.value} {ticker} result={result} "
            f"pnl={float(pnl):+.3f}"
        )
