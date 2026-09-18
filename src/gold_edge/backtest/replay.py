"""Replays recorded ticks + book snapshots through the exact same engine
code the live server runs (`engine.state_machine.step` / `apply_fill`), per
CLAUDE.md: "no duplicated strategy logic." The only things this module adds
on top of that shared engine are the pieces that only make sense offline:
- deciding fill vs. miss for each issued signal via the mandatory human-delay
  simulator (`backtest.fills`), and applying that fill at the right time;
- settling a position still open at window close using the recorded result
  (`server.settle_position_pnl`, also shared rather than reimplemented);
- carrying `realized_pnl_today` and one continuous `VolatilityTracker`
  across window boundaries, mirroring how `server.AppState` runs a single
  session rather than resetting per window.
"""

from __future__ import annotations

import itertools
import random
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from gold_edge.backtest.fills import BookHistory, sample_delay_s, simulate_fill
from gold_edge.config import BacktestConfig, EngineConfig, FeesConfig, ModelConfig, VolatilityConfig
from gold_edge.engine.state_machine import EngineState, MarketSnapshot, apply_fill, step
from gold_edge.model.fair_value import compute_fair_value
from gold_edge.model.fees import settle_position_pnl
from gold_edge.model.volatility import VolatilityTracker
from gold_edge.models import Action, BookSnapshot, Fill, Position, Signal, Tick, Window


@dataclass(frozen=True)
class RecordedWindow:
    """One window's recorded data: its own book snapshots and the eventual
    settlement result (`"yes"` / `"no"` / `None` if unknown)."""

    window: Window
    books: list[BookSnapshot]
    settlement_result: str | None


@dataclass(frozen=True)
class SignalRecord:
    signal: Signal
    filled: bool
    fill: Fill | None


@dataclass(frozen=True)
class RoundTripRecord:
    window_ticker: str
    side: str
    size: Decimal
    entry_price: Decimal
    entry_time: datetime
    exit_price: Decimal | None
    exit_time: datetime
    exit_reason: str
    pnl: Decimal
    fees_paid: Decimal
    # The original BUY/SELL Signal objects behind this round trip, when
    # available — the grader needs these for markouts (entry_signal is
    # always set for a filled entry; exit_signal is None for a position
    # held to settlement, since no exit signal was ever issued).
    entry_signal: Signal | None = None
    exit_signal: Signal | None = None


@dataclass(frozen=True)
class WindowReplayResult:
    ticker: str
    signal_records: list[SignalRecord]
    round_trips: list[RoundTripRecord]
    settlement_result: str | None
    ending_realized_pnl_today: Decimal
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class BacktestResult:
    windows: list[WindowReplayResult]


def _entry_fee(position: Position, fees_cfg: FeesConfig) -> Decimal:
    from gold_edge.model import fees as fees_mod

    return fees_mod.taker_fee(
        position.size,
        position.entry_price,
        Decimal(str(fees_cfg.fee_multiplier)),
        Decimal(str(fees_cfg.base_rate)),
    )


def _exit_fee(fill: Fill, fees_cfg: FeesConfig) -> Decimal:
    from gold_edge.model import fees as fees_mod

    return fees_mod.taker_fee(
        fill.size,
        fill.price,
        Decimal(str(fees_cfg.fee_multiplier)),
        Decimal(str(fees_cfg.base_rate)),
    )


def _replay_window(
    rw: RecordedWindow,
    ticks_iter: list[Tick],
    tick_start_idx: int,
    engine_state: EngineState,
    vol_tracker: VolatilityTracker,
    engine_cfg: EngineConfig,
    fees_cfg: FeesConfig,
    vol_cfg: VolatilityConfig,
    model_cfg: ModelConfig,
    backtest_cfg: BacktestConfig,
    rng: random.Random,
    on_step: Callable[[EngineState, MarketSnapshot], None] | None = None,
) -> tuple[WindowReplayResult, EngineState, int]:
    window = rw.window
    book_history = BookHistory(rw.books)
    notes: list[str] = []
    signal_records: list[SignalRecord] = []
    round_trips: list[RoundTripRecord] = []

    latest_price: float | None = None
    latest_tick_receive_time: datetime | None = None
    latest_book: BookSnapshot | None = None
    latest_book_receive_time: datetime | None = None
    pending_fill: Fill | None = None
    pending_fill_signal: Signal | None = None
    open_position_entry_signal: Signal | None = None

    events: list[tuple[datetime, str, Tick | BookSnapshot]] = []
    idx = tick_start_idx
    while idx < len(ticks_iter) and ticks_iter[idx].receive_time < window.close_time:
        events.append((ticks_iter[idx].receive_time, "tick", ticks_iter[idx]))
        idx += 1
    for b in rw.books:
        events.append((b.receive_time, "book", b))
    events.sort(key=lambda e: e[0])

    def maybe_apply_pending(now: datetime) -> None:
        nonlocal pending_fill, pending_fill_signal, engine_state
        if pending_fill is not None and pending_fill.logged_at <= now:
            _apply_and_record(pending_fill, pending_fill_signal)
            pending_fill = None
            pending_fill_signal = None

    def _apply_and_record(fill: Fill, signal: Signal | None) -> None:
        nonlocal engine_state, open_position_entry_signal
        position_before = engine_state.position
        entry_signal_before = open_position_entry_signal
        pnl_before = engine_state.realized_pnl_today
        engine_state = apply_fill(engine_state, fill, engine_cfg, fees_cfg)
        if fill.action is Action.BUY:
            open_position_entry_signal = signal
        elif fill.action is Action.SELL and position_before is not None:
            pnl = engine_state.realized_pnl_today - pnl_before
            fees_paid = _entry_fee(position_before, fees_cfg) + _exit_fee(fill, fees_cfg)
            round_trips.append(
                RoundTripRecord(
                    window_ticker=window.ticker,
                    side=position_before.side.value,
                    size=position_before.size,
                    entry_price=position_before.entry_price,
                    entry_time=position_before.entered_at,
                    exit_price=fill.price,
                    exit_time=fill.logged_at,
                    exit_reason="sold",
                    pnl=pnl,
                    fees_paid=fees_paid,
                    entry_signal=entry_signal_before,
                    exit_signal=signal,
                )
            )
            open_position_entry_signal = None

    for now, kind, payload in events:
        maybe_apply_pending(now)

        if kind == "tick":
            t: Tick = payload  # type: ignore[assignment]
            vol_tracker.update(t.price, t.publish_time)
            latest_price = t.price
            latest_tick_receive_time = t.receive_time
        else:
            b: BookSnapshot = payload  # type: ignore[assignment]
            latest_book = b
            latest_book_receive_time = b.receive_time

        if now < window.open_time:
            continue
        if window.s0 is None or latest_price is None or latest_book is None:
            continue

        tau_minutes = window.seconds_left(now) / 60.0
        fair = compute_fair_value(
            latest_price,
            float(window.s0),
            vol_tracker.sigma_per_minute,
            tau_minutes,
            model_cfg.min_fair_value,
            model_cfg.max_fair_value,
        )
        pyth_age_s = (
            (now - latest_tick_receive_time).total_seconds() if latest_tick_receive_time else 999.0
        )
        kalshi_age_s = (
            (now - latest_book_receive_time).total_seconds() if latest_book_receive_time else 999.0
        )
        snapshot = MarketSnapshot(
            now=now,
            window=window,
            book=latest_book,
            fair=fair,
            pyth_age_s=pyth_age_s,
            kalshi_age_s=kalshi_age_s,
            short_horizon_sigma_per_minute=vol_tracker.short_horizon_sigma_per_minute,
            underlying_price=latest_price,
        )
        if on_step is not None:
            on_step(engine_state, snapshot)
        result = step(engine_state, snapshot, engine_cfg, fees_cfg, vol_cfg.vol_spike_limit)
        engine_state = result.state

        if result.signal is not None:
            delay = sample_delay_s(backtest_cfg, rng)
            fill = simulate_fill(result.signal, book_history, delay)
            signal_records.append(
                SignalRecord(signal=result.signal, filled=fill is not None, fill=fill)
            )
            if fill is not None:
                pending_fill = fill
                pending_fill_signal = result.signal

    maybe_apply_pending(window.close_time)

    if engine_state.position is not None:
        position = engine_state.position
        if rw.settlement_result is None:
            notes.append(
                f"{window.ticker}: position still open at close with no recorded settlement "
                "result — excluded from P&L."
            )
        else:
            pnl = settle_position_pnl(position, rw.settlement_result, fees_cfg)
            won = rw.settlement_result == position.side.value.lower()
            round_trips.append(
                RoundTripRecord(
                    window_ticker=window.ticker,
                    side=position.side.value,
                    size=position.size,
                    entry_price=position.entry_price,
                    entry_time=position.entered_at,
                    exit_price=Decimal("1.00") if won else Decimal("0.00"),
                    exit_time=window.close_time,
                    exit_reason="settled",
                    pnl=pnl,
                    fees_paid=_entry_fee(position, fees_cfg),
                    entry_signal=open_position_entry_signal,
                )
            )
            new_pnl_today = engine_state.realized_pnl_today + pnl
            engine_state = replace(engine_state, realized_pnl_today=new_pnl_today)

    ending_pnl = engine_state.realized_pnl_today
    next_state = EngineState(realized_pnl_today=ending_pnl)

    return (
        WindowReplayResult(
            ticker=window.ticker,
            signal_records=signal_records,
            round_trips=round_trips,
            settlement_result=rw.settlement_result,
            ending_realized_pnl_today=ending_pnl,
            notes=notes,
        ),
        next_state,
        idx,
    )


def replay_all(
    ticks: list[Tick],
    windows: list[RecordedWindow],
    engine_cfg: EngineConfig,
    fees_cfg: FeesConfig,
    vol_cfg: VolatilityConfig,
    model_cfg: ModelConfig,
    backtest_cfg: BacktestConfig,
    rng: random.Random | None = None,
    on_step: Callable[[EngineState, MarketSnapshot], None] | None = None,
) -> BacktestResult:
    """Replays a sorted sequence of windows against one continuous tick
    stream and one continuous VolatilityTracker (matching how `server.py`
    runs a single live session across many windows), resetting engine
    position state at each window boundary but always carrying
    `realized_pnl_today` forward.

    `on_step`, if given, is called with the engine's state and the market
    snapshot right before every `step()` call — i.e. exactly what the
    engine saw and knew when it made (or didn't make) each decision. This
    is how learning/opportunities.py rides along on a real replay to ask
    "why didn't the engine take this?" without re-deriving the engine's own
    trajectory a second time."""
    rng = rng or random.Random()
    sorted_ticks = sorted(ticks, key=lambda t: t.receive_time)
    sorted_windows = sorted(windows, key=lambda rw: rw.window.open_time)

    vol_tracker = VolatilityTracker(
        half_life_s=vol_cfg.ewma_half_life_s,
        short_horizon_s=vol_cfg.short_horizon_s,
        min_sigma_per_minute=vol_cfg.min_sigma_per_minute,
    )
    engine_state = EngineState()
    tick_idx = 0
    results: list[WindowReplayResult] = []
    pnl_date: date | None = None

    for rw in sorted_windows:
        # Mirrors server.py's live daily reset: realized_pnl_today carries
        # across windows within the same UTC day, but a new calendar day
        # starts fresh — otherwise DAILY_LOSS_STOP would gate on a running
        # total that's never actually "today's" losses in a multi-day
        # backtest.
        window_date = rw.window.open_time.date()
        if pnl_date is not None and window_date != pnl_date:
            engine_state = replace(engine_state, realized_pnl_today=Decimal("0"))
        pnl_date = window_date

        window_result, engine_state, tick_idx = _replay_window(
            rw,
            sorted_ticks,
            tick_idx,
            engine_state,
            vol_tracker,
            engine_cfg,
            fees_cfg,
            vol_cfg,
            model_cfg,
            backtest_cfg,
            rng,
            on_step,
        )
        results.append(window_result)

    return BacktestResult(windows=results)


def load_recorded_data(
    sqlite_path: Path, start: datetime | None = None, end: datetime | None = None
) -> tuple[list[Tick], list[RecordedWindow]]:
    """Loads everything `replay_all` needs from the recorder's SQLite file:
    every window with a known S0 (unresolved/void windows can't be replayed
    since fair value has no reference price), its book snapshots, its
    settlement result if known, and the full tick stream over the same
    range (ticks aren't tagged by window — they're one continuous feed)."""
    conn = sqlite3.connect(sqlite_path)
    conn.row_factory = sqlite3.Row
    try:
        window_query = "SELECT * FROM windows WHERE s0 IS NOT NULL"
        window_params: list[Any] = []
        if start is not None:
            window_query += " AND open_time >= ?"
            window_params.append(start.isoformat())
        if end is not None:
            window_query += " AND close_time <= ?"
            window_params.append(end.isoformat())
        window_query += " ORDER BY open_time"

        recorded_windows: list[RecordedWindow] = []
        for row in conn.execute(window_query, window_params).fetchall():
            window = Window(
                ticker=row["ticker"],
                event_ticker=row["event_ticker"],
                series_ticker=row["series_ticker"],
                open_time=datetime.fromisoformat(row["open_time"]),
                close_time=datetime.fromisoformat(row["close_time"]),
                s0=Decimal(row["s0"]),
                status=row["status"],
            )
            book_rows = conn.execute(
                "SELECT * FROM book_snapshots WHERE window_ticker = ? ORDER BY receive_time",
                (window.ticker,),
            ).fetchall()
            books = [
                BookSnapshot(
                    window_ticker=b["window_ticker"],
                    yes_bid=Decimal(b["yes_bid"]),
                    yes_ask=Decimal(b["yes_ask"]),
                    yes_bid_size=Decimal(b["yes_bid_size"]),
                    yes_ask_size=Decimal(b["yes_ask_size"]),
                    no_bid=Decimal(b["no_bid"]),
                    no_ask=Decimal(b["no_ask"]),
                    no_bid_size=Decimal(b["no_bid_size"]),
                    no_ask_size=Decimal(b["no_ask_size"]),
                    receive_time=datetime.fromisoformat(b["receive_time"]),
                )
                for b in book_rows
            ]
            settlement_row = conn.execute(
                "SELECT result FROM settlements WHERE ticker = ?", (window.ticker,)
            ).fetchone()
            settlement_result = settlement_row["result"] if settlement_row else None
            recorded_windows.append(
                RecordedWindow(window=window, books=books, settlement_result=settlement_result)
            )

        tick_query = "SELECT * FROM ticks"
        tick_params: list[Any] = []
        conditions = []
        if start is not None:
            conditions.append("receive_time >= ?")
            tick_params.append(start.isoformat())
        if end is not None:
            conditions.append("receive_time <= ?")
            tick_params.append(end.isoformat())
        if conditions:
            tick_query += " WHERE " + " AND ".join(conditions)
        tick_query += " ORDER BY receive_time"
        ticks = [
            Tick(
                symbol=t["symbol"],
                price=t["price"],
                conf=t["conf"],
                expo=t["expo"],
                publish_time=datetime.fromisoformat(t["publish_time"]),
                receive_time=datetime.fromisoformat(t["receive_time"]),
            )
            for t in conn.execute(tick_query, tick_params).fetchall()
        ]
        return ticks, recorded_windows
    finally:
        conn.close()


def _net_pnl(result: BacktestResult) -> Decimal:
    return sum((rt.pnl for wr in result.windows for rt in wr.round_trips), Decimal("0"))


def _round_trip_count(result: BacktestResult) -> int:
    return sum(len(wr.round_trips) for wr in result.windows)


def _max_drawdown(result: BacktestResult) -> Decimal:
    all_round_trips = [rt for wr in result.windows for rt in wr.round_trips]
    peak = cum = max_dd = Decimal("0")
    for rt in sorted(all_round_trips, key=lambda r: r.exit_time):
        cum += rt.pnl
        peak = max(peak, cum)
        max_dd = max(max_dd, peak - cum)
    return max_dd


def run_sweep(
    train_ticks: list[Tick],
    train_windows: list[RecordedWindow],
    test_ticks: list[Tick],
    test_windows: list[RecordedWindow],
    base_engine_cfg: EngineConfig,
    fees_cfg: FeesConfig,
    vol_cfg: VolatilityConfig,
    model_cfg: ModelConfig,
    backtest_cfg: BacktestConfig,
    param_grid: dict[str, list[float]],
    seed: int | None = None,
    max_combinations: int = 200,
) -> list[dict[str, Any]]:
    """Bounded grid search over EngineConfig fields, per CLAUDE.md: 'keep
    search grids small and bounded to sane ranges' and 'tuned on training
    days and reported on held-out days only.' Every candidate is evaluated
    on both splits; the caller sees both, never just the train-winner's
    held-out number cherry-picked after the fact — the whole point of a
    held-out set is that overfitting to train doesn't hide here."""
    keys = list(param_grid)
    combos = list(itertools.product(*(param_grid[k] for k in keys)))
    if len(combos) > max_combinations:
        raise ValueError(
            f"sweep grid too large: {len(combos)} combinations exceeds max_combinations="
            f"{max_combinations}. Narrow the grid."
        )

    candidates: list[dict[str, Any]] = []
    for combo in combos:
        overrides = dict(zip(keys, combo, strict=True))
        candidate_cfg = base_engine_cfg.model_copy(update=overrides)

        train_result = replay_all(
            train_ticks,
            train_windows,
            candidate_cfg,
            fees_cfg,
            vol_cfg,
            model_cfg,
            backtest_cfg,
            random.Random(seed),
        )
        test_result = replay_all(
            test_ticks,
            test_windows,
            candidate_cfg,
            fees_cfg,
            vol_cfg,
            model_cfg,
            backtest_cfg,
            random.Random(seed),
        )
        candidates.append(
            {
                "params": overrides,
                "train_net_pnl": _net_pnl(train_result),
                "train_round_trips": _round_trip_count(train_result),
                "test_net_pnl": _net_pnl(test_result),
                "test_round_trips": _round_trip_count(test_result),
                "test_max_drawdown": _max_drawdown(test_result),
            }
        )

    candidates.sort(key=lambda c: c["train_net_pnl"], reverse=True)
    return candidates
