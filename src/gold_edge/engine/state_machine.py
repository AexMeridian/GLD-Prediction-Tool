"""FLAT / LONG_YES / LONG_NO state machine.

Pure and deterministic: `step()` takes an explicit `EngineState` and a
`MarketSnapshot` and returns a new `EngineState` plus at most one new
`Signal` — no hidden globals, no I/O, no wall-clock reads (the caller
supplies `now`). This is what makes it unit-testable and replayable.

Position state changes ONLY through `apply_fill` — issuing a BUY/SELL
signal never moves `position_state` on its own, per CLAUDE.md ("Engine
position state changes only when the user logs a fill... Until confirmed,
show 'awaiting fill.'"). While a signal is pending (unconfirmed), `step()`
does not issue another one; it only watches for that signal to expire.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from decimal import Decimal

from gold_edge.config import EngineConfig, FeesConfig
from gold_edge.engine import risk
from gold_edge.engine.signals import build_signal, is_expired, price_moved_past_limit
from gold_edge.model import fees as fees_mod
from gold_edge.model.fair_value import FairValue
from gold_edge.models import (
    Action,
    BookSnapshot,
    Fill,
    Position,
    PositionState,
    Side,
    Signal,
    SignalStatus,
    Window,
)


@dataclass(frozen=True)
class EngineState:
    position_state: PositionState = PositionState.FLAT
    position: Position | None = None
    pending_signal: Signal | None = None
    cooldown_until: datetime | None = None
    round_trips: int = 0
    realized_pnl_today: Decimal = Decimal("0")
    gap_exceeded_since: dict[Side, datetime | None] = field(
        default_factory=lambda: {Side.YES: None, Side.NO: None}
    )
    # Set on an exit signal that qualifies as a flip; consumed by apply_fill
    # to skip cooldown and pre-seed persistence for the opposite side.
    flip_hint_side: Side | None = None


@dataclass(frozen=True)
class MarketSnapshot:
    now: datetime
    window: Window
    book: BookSnapshot
    fair: FairValue
    pyth_age_s: float
    kalshi_age_s: float
    short_horizon_sigma_per_minute: float | None = None
    # The raw Pyth price S behind `fair` -- optional (defaults to None so
    # existing callers/tests are unaffected) because inverting fair value
    # back to S is unreliable once it's clamped at [min,max]. Populated by
    # both live (server.py) and replay (backtest/replay.py) callers, who
    # already have it on hand; used by learning/patterns.py for the
    # |S-S0|-in-sigmas feature.
    underlying_price: float | None = None


@dataclass(frozen=True)
class StepResult:
    state: EngineState
    signal: Signal | None = None
    finalized_signals: list[Signal] = field(default_factory=list)
    note: str | None = None


def fee_for(price: Decimal, fees_cfg: FeesConfig, contracts: Decimal = Decimal(1)) -> Decimal:
    return fees_mod.taker_fee(
        contracts,
        price,
        Decimal(str(fees_cfg.fee_multiplier)),
        Decimal(str(fees_cfg.base_rate)),
    )


def round_trip_cost(book: BookSnapshot, side: Side, fees_cfg: FeesConfig) -> Decimal:
    """fee(ask) + fee_est(exit) + spread_est, all per contract. Exported (not
    just used internally) so offline analysis — learning/opportunities.py's
    hindsight scanner — can compute exactly the same gap-after-costs the
    live engine uses, instead of re-deriving the formula."""
    entry_fee = fee_for(book.ask(side), fees_cfg)
    exit_fee_est = fee_for(book.bid(side), fees_cfg)
    return entry_fee + exit_fee_est + book.spread(side)


def gap_for_side(fair_side: float, book: BookSnapshot, side: Side, fees_cfg: FeesConfig) -> float:
    return fair_side - float(book.ask(side)) - float(round_trip_cost(book, side, fees_cfg))


def exit_value_for_side(book: BookSnapshot, side: Side, fees_cfg: FeesConfig) -> float:
    return float(book.bid(side) - fee_for(book.bid(side), fees_cfg))


def step(
    state: EngineState,
    snapshot: MarketSnapshot,
    engine_cfg: EngineConfig,
    fees_cfg: FeesConfig,
    vol_spike_limit: float,
) -> StepResult:
    finalized: list[Signal] = []

    if state.pending_signal is not None:
        pending = state.pending_signal
        if is_expired(pending, snapshot.now) or price_moved_past_limit(pending, snapshot.book):
            finalized.append(pending.model_copy(update={"status": SignalStatus.EXPIRED}))
            new_gap_since = dict(state.gap_exceeded_since)
            if pending.action is Action.BUY:
                # "Don't chase": require a fresh persistence window before
                # this side can signal again.
                new_gap_since[pending.side] = None
            state = replace(state, pending_signal=None, gap_exceeded_since=new_gap_since)
        else:
            # Still awaiting a fill or an expiry; no new decision this step.
            return StepResult(state=state, finalized_signals=finalized)

    stale = risk.is_stale(snapshot.pyth_age_s, snapshot.kalshi_age_s, engine_cfg.stale_s)

    if state.position_state is PositionState.FLAT:
        return _step_flat(state, snapshot, engine_cfg, fees_cfg, vol_spike_limit, stale, finalized)
    return _step_in_position(state, snapshot, engine_cfg, fees_cfg, stale, finalized)


def _step_flat(
    state: EngineState,
    snap: MarketSnapshot,
    engine_cfg: EngineConfig,
    fees_cfg: FeesConfig,
    vol_spike_limit: float,
    stale: bool,
    finalized: list[Signal],
) -> StepResult:
    if stale:
        # Freeze persistence timers rather than reset them; a brief data
        # gap shouldn't erase progress toward an otherwise-real edge.
        return StepResult(state=state, finalized_signals=finalized)

    fair_by_side = {Side.YES: snap.fair.yes, Side.NO: snap.fair.no}
    gaps = {
        side: gap_for_side(fair_by_side[side], snap.book, side, fees_cfg)
        for side in (Side.YES, Side.NO)
    }

    new_gap_since = dict(state.gap_exceeded_since)
    for side in (Side.YES, Side.NO):
        if gaps[side] > engine_cfg.enter_edge:
            if new_gap_since[side] is None:
                new_gap_since[side] = snap.now
        else:
            new_gap_since[side] = None
    state = replace(state, gap_exceeded_since=new_gap_since)

    time_left_s = snap.window.seconds_left(snap.now)
    max_spread = Decimal(str(engine_cfg.max_spread))
    daily_loss_stop = Decimal(str(engine_cfg.daily_loss_stop))

    eligible: list[Side] = []
    for side in (Side.YES, Side.NO):
        since = new_gap_since[side]
        persisted = since is not None and (snap.now - since).total_seconds() >= engine_cfg.persist_s
        if not persisted:
            continue
        if risk.cooldown_active(snap.now, state.cooldown_until):
            continue
        if not risk.entry_cutoff_ok(time_left_s, engine_cfg.entry_cutoff_s):
            continue
        if not risk.spread_ok(snap.book.spread(side), max_spread):
            continue
        if not risk.vol_ok(snap.short_horizon_sigma_per_minute, vol_spike_limit):
            continue
        if not risk.round_trips_ok(state.round_trips, engine_cfg.max_round_trips):
            continue
        if not risk.daily_loss_ok(state.realized_pnl_today, daily_loss_stop):
            continue
        eligible.append(side)

    if not eligible:
        return StepResult(state=state, finalized_signals=finalized)
    if len(eligible) == 2:
        return StepResult(
            state=state, finalized_signals=finalized, note="both_sides_qualified_took_neither"
        )

    side = eligible[0]
    signal = build_signal(
        action=Action.BUY,
        side=side,
        window_ticker=snap.window.ticker,
        book=snap.book,
        fair=fair_by_side[side],
        size=Decimal(1),
        edge_after_costs=gaps[side],
        reason="enter_edge",
        now=snap.now,
        ttl_s=engine_cfg.signal_ttl_s,
    )
    state = replace(state, pending_signal=signal)
    return StepResult(state=state, signal=signal, finalized_signals=finalized)


def _step_in_position(
    state: EngineState,
    snap: MarketSnapshot,
    engine_cfg: EngineConfig,
    fees_cfg: FeesConfig,
    stale: bool,
    finalized: list[Signal],
) -> StepResult:
    side = Side.YES if state.position_state is PositionState.LONG_YES else Side.NO
    opposite = Side.NO if side is Side.YES else Side.YES
    position = state.position
    assert position is not None  # invariant: non-FLAT states always carry a position

    fair_side = snap.fair.yes if side is Side.YES else snap.fair.no
    exit_val = exit_value_for_side(snap.book, side, fees_cfg)

    reason: str | None = None
    if exit_val > fair_side:
        reason = "overshoot"
    elif exit_val >= fair_side - engine_cfg.converge_band:
        reason = "converged"
    elif float(snap.book.bid(side)) <= float(position.entry_price) - engine_cfg.stop:
        reason = "stop"
    else:
        time_left_s = snap.window.seconds_left(snap.now)
        if risk.exit_cutoff_reached(time_left_s, engine_cfg.exit_cutoff_s):
            hold = (
                engine_cfg.hold_to_settlement_when_itm
                and fair_side >= engine_cfg.hold_to_settlement_fair_threshold
            )
            if not hold:
                reason = "time_cutoff"
    if reason is None and stale:
        reason = "stale_data"

    if reason is None:
        return StepResult(state=state, finalized_signals=finalized)

    flip_hint: Side | None = None
    # Never flip straight out of a stop: getting stopped out is evidence the
    # model's read was wrong a moment ago, which is the worst time to skip
    # the cooldown and re-enter with fresh risk. Stale data shouldn't open
    # new risk either, for the same "don't trust what you can't verify"
    # reason entries require fresh data at all.
    if reason not in ("stale_data", "stop"):
        fair_opp = snap.fair.no if side is Side.YES else snap.fair.yes
        gap_opp = gap_for_side(fair_opp, snap.book, opposite, fees_cfg)
        exit_fee = fee_for(snap.book.bid(side), fees_cfg)
        if gap_opp > engine_cfg.enter_edge + float(exit_fee):
            flip_hint = opposite

    signal = build_signal(
        action=Action.SELL,
        side=side,
        window_ticker=snap.window.ticker,
        book=snap.book,
        fair=fair_side,
        size=position.size,
        edge_after_costs=fair_side - exit_val,
        reason=f"{reason}:flip_{flip_hint.value}" if flip_hint is not None else reason,
        now=snap.now,
        ttl_s=engine_cfg.signal_ttl_s,
    )
    state = replace(state, pending_signal=signal, flip_hint_side=flip_hint)
    return StepResult(state=state, signal=signal, finalized_signals=finalized)


def apply_fill(
    state: EngineState, fill: Fill, engine_cfg: EngineConfig, fees_cfg: FeesConfig
) -> EngineState:
    """Confirms a signal was acted on. This is the ONLY place position_state
    changes — issuing a signal never does."""
    if fill.is_skip:
        return apply_skip(state)

    if fill.action is Action.BUY:
        new_position_state = (
            PositionState.LONG_YES if fill.side is Side.YES else PositionState.LONG_NO
        )
        position = Position(
            window_ticker=fill.window_ticker,
            side=fill.side,
            size=fill.size,
            entry_price=fill.price,
            entered_at=fill.logged_at,
            state=new_position_state,
        )
        return replace(
            state,
            position=position,
            position_state=new_position_state,
            pending_signal=None,
        )

    # SELL fill: closes the open position and realizes P&L for the round trip.
    position = state.position
    if position is None:
        raise ValueError("Received a SELL fill with no open position to close")

    entry_fee = fee_for(position.entry_price, fees_cfg, contracts=position.size)
    exit_fee = fee_for(fill.price, fees_cfg, contracts=fill.size)
    pnl = (fill.price - position.entry_price) * fill.size - entry_fee - exit_fee

    if state.flip_hint_side is not None:
        # A flip skips the cooldown and treats persistence for the target
        # side as already satisfied — it was validated at the moment the
        # exit signal was issued, not re-discovered from scratch.
        new_gap_since = dict(state.gap_exceeded_since)
        preseeded = fill.logged_at - timedelta(seconds=engine_cfg.persist_s)
        new_gap_since[state.flip_hint_side] = preseeded
        new_cooldown_until = None
    else:
        new_gap_since = {Side.YES: None, Side.NO: None}
        new_cooldown_until = fill.logged_at + timedelta(seconds=engine_cfg.cooldown_s)

    return replace(
        state,
        position=None,
        position_state=PositionState.FLAT,
        pending_signal=None,
        round_trips=state.round_trips + 1,
        realized_pnl_today=state.realized_pnl_today + pnl,
        cooldown_until=new_cooldown_until,
        gap_exceeded_since=new_gap_since,
        flip_hint_side=None,
    )


def apply_skip(state: EngineState) -> EngineState:
    """The user declined the pending signal. Position state is untouched;
    a skipped entry resets that side's persistence timer (same "don't
    chase" reasoning as expiry), a skipped exit simply clears the signal
    so the exit rules re-evaluate fresh next step."""
    pending = state.pending_signal
    if pending is None:
        return state
    new_gap_since = dict(state.gap_exceeded_since)
    if pending.action is Action.BUY:
        new_gap_since[pending.side] = None
    return replace(state, pending_signal=None, gap_exceeded_since=new_gap_since)
