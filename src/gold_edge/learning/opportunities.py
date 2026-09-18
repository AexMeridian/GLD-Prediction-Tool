"""Hindsight opportunity scanner, per CLAUDE.md: after each window, find
candidate entries the engine's filters blocked and check whether they would
have been profitable anyway, and separately score each filter by how much
money it saved vs. cost.

Design: this rides along on a real `backtest.replay.replay_all` pass via its
`on_step` hook rather than re-deriving the engine's trajectory. That hook
hands over the exact `(EngineState, MarketSnapshot)` pair the live engine
saw right before it decided (or declined) to act — so "why didn't the
engine take this?" is answered by literally re-running the same per-side
gates (`engine.risk`, `state_machine.gap_for_side`) the engine itself just
used, not a parallel reimplementation that could quietly drift from it.

A "candidate" is deliberately looser than the engine's own entry condition:
any second where a side's raw fair value exceeds its ask (`CLAUDE.md`:
"ask was below fair value"), before round-trip costs. Most candidates will
therefore be `below_enter_edge` — the mispricing was real but too small
once costs are subtracted — with the other reasons covering the engine's
remaining entry gates in the same precedence order `_step_flat` itself
checks them.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from gold_edge.backtest.fills import BookHistory, sample_delay_s, simulate_fill
from gold_edge.config import BacktestConfig, EngineConfig, FeesConfig
from gold_edge.engine import risk
from gold_edge.engine.signals import build_signal
from gold_edge.engine.state_machine import (
    EngineState,
    MarketSnapshot,
    gap_for_side,
    round_trip_cost,
    step,
)
from gold_edge.learning.calibrator import IsotonicCalibrator
from gold_edge.model import fees as fees_mod
from gold_edge.model.fees import settle_position_pnl
from gold_edge.models import Action, Position, PositionState, Side

# Every reason `classify_candidates` can return, in the precedence order the
# checks below apply them. `daily_loss_stop` and `fair_value_disagreed`
# extend CLAUDE.md's listed set with the two remaining real "why not"
# branches `_step_flat` actually has (the engine's daily kill switch, and
# its "both sides qualified, take neither" rule) — omitting them would
# force a misleading fallback for genuinely-explainable blocks.
REASONS = (
    "already_in_position",
    "stale_data",
    "below_enter_edge",
    "persistence",
    "cooldown",
    "entry_cutoff",
    "spread_filter",
    "vol_spike",
    "max_round_trips",
    "daily_loss_stop",
    "fair_value_disagreed",
)


@dataclass(frozen=True)
class Opportunity:
    window_ticker: str
    side: Side
    at: datetime
    reason: str
    realistic_net_pnl: Decimal
    oracle_net_pnl: Decimal


@dataclass(frozen=True)
class FilterEvent:
    """One blocked candidate, logged regardless of whether it cleared
    MIN_OPPORTUNITY — the raw material for `filter_scorecard`."""

    window_ticker: str
    side: Side
    at: datetime
    reason: str
    realistic_net_pnl: Decimal


@dataclass(frozen=True)
class FilterStats:
    reason: str
    n: int
    losses_avoided: Decimal
    profits_missed: Decimal

    @property
    def value(self) -> Decimal:
        return self.losses_avoided - self.profits_missed


def _own_chain_reason(
    side: Side,
    fair_side: float,
    state: EngineState,
    snap: MarketSnapshot,
    engine_cfg: EngineConfig,
    fees_cfg: FeesConfig,
    vol_spike_limit: float,
) -> str | None:
    """None means this side's own gates all pass — the engine would (or
    did) take it, ignoring the separate both-sides-qualify exclusion."""
    gap_after_costs = gap_for_side(fair_side, snap.book, side, fees_cfg)
    if gap_after_costs <= engine_cfg.enter_edge:
        return "below_enter_edge"

    since = state.gap_exceeded_since.get(side)
    persisted = since is not None and (snap.now - since).total_seconds() >= engine_cfg.persist_s
    if not persisted:
        return "persistence"
    if risk.cooldown_active(snap.now, state.cooldown_until):
        return "cooldown"
    time_left_s = snap.window.seconds_left(snap.now)
    if not risk.entry_cutoff_ok(time_left_s, engine_cfg.entry_cutoff_s):
        return "entry_cutoff"
    if not risk.spread_ok(snap.book.spread(side), Decimal(str(engine_cfg.max_spread))):
        return "spread_filter"
    if not risk.vol_ok(snap.short_horizon_sigma_per_minute, vol_spike_limit):
        return "vol_spike"
    if not risk.round_trips_ok(state.round_trips, engine_cfg.max_round_trips):
        return "max_round_trips"
    if not risk.daily_loss_ok(state.realized_pnl_today, Decimal(str(engine_cfg.daily_loss_stop))):
        return "daily_loss_stop"
    return None


def classify_candidates(
    state_before: EngineState,
    snap: MarketSnapshot,
    engine_cfg: EngineConfig,
    fees_cfg: FeesConfig,
    vol_spike_limit: float,
) -> dict[Side, str | None]:
    """For every side whose ask was below fair value at this instant, why
    didn't the engine take it? `None` means it would have (and, in a real
    trajectory, did). Mirrors `step()`'s own dispatch order exactly: a
    pending signal or an open position means entry logic never even runs."""
    if state_before.pending_signal is not None:
        return {}

    fair_by_side = {Side.YES: snap.fair.yes, Side.NO: snap.fair.no}
    candidate_sides = [s for s in (Side.YES, Side.NO) if fair_by_side[s] > float(snap.book.ask(s))]
    if not candidate_sides:
        return {}

    if state_before.position_state is not PositionState.FLAT:
        return {s: "already_in_position" for s in candidate_sides}

    if risk.is_stale(snap.pyth_age_s, snap.kalshi_age_s, engine_cfg.stale_s):
        return {s: "stale_data" for s in candidate_sides}

    own_reason = {
        s: _own_chain_reason(
            s, fair_by_side[s], state_before, snap, engine_cfg, fees_cfg, vol_spike_limit
        )
        for s in (Side.YES, Side.NO)
        if fair_by_side[s] > float(snap.book.ask(s))
    }
    would_qualify = {s for s, r in own_reason.items() if r is None}

    result: dict[Side, str | None] = {}
    for s in candidate_sides:
        if own_reason[s] is not None:
            result[s] = own_reason[s]
        elif len(would_qualify) == 2:
            result[s] = "fair_value_disagreed"
        else:
            result[s] = None
    return result


def _round_trip_fee(price: Decimal, size: Decimal, fees_cfg: FeesConfig) -> Decimal:
    return fees_mod.taker_fee(
        size, price, Decimal(str(fees_cfg.fee_multiplier)), Decimal(str(fees_cfg.base_rate))
    )


def simulate_from_entry(
    side: Side,
    entry_price: Decimal,
    entry_time: datetime,
    entry_index: int,
    trace: Sequence[tuple[EngineState, MarketSnapshot]],
    book_history: BookHistory,
    engine_cfg: EngineConfig,
    fees_cfg: FeesConfig,
    vol_spike_limit: float,
    backtest_cfg: BacktestConfig,
    settlement_result: str | None,
    rng: random.Random,
) -> tuple[Decimal, Decimal] | None:
    """Given a position already opened at `entry_price`/`entry_time`, what
    happens next? Returns `(realistic_net_pnl, oracle_net_pnl)`, or None if
    the outcome can't be determined (no exit fired and no settlement
    result is known). Shared by `simulate_hypothetical_trade` (entry comes
    from a simulated fill) and grading a real missed/skipped signal (entry
    is assumed to have succeeded at its original limit price) — the "what
    happens after we're in" logic is identical either way.
    """
    _, entry_snap = trace[entry_index]
    window = entry_snap.window
    size = Decimal(1)
    entry_fee = _round_trip_fee(entry_price, size, fees_cfg)

    position = Position(
        window_ticker=window.ticker,
        side=side,
        size=size,
        entry_price=entry_price,
        entered_at=entry_time,
        state=PositionState.LONG_YES if side is Side.YES else PositionState.LONG_NO,
    )

    # Oracle ceiling = the best of every way the position could have been
    # closed out: selling at the best bid seen before close, OR just
    # holding to settlement (always an available "exit," and often the
    # better one — e.g. a book that never reprices before a decisive
    # settlement). Comparing only against book prices would understate the
    # ceiling whenever settlement beats anything the book ever offered.
    at_entry_book = book_history.at_or_before(entry_time)
    later_books = book_history.between(entry_time, window.close_time)
    oracle_books = ([at_entry_book] if at_entry_book is not None else []) + later_books
    if oracle_books:
        best_bid = max(b.bid(side) for b in oracle_books)
        best_exit_fee = _round_trip_fee(best_bid, size, fees_cfg)
        oracle_sell_pnl = (best_bid - entry_price) * size - entry_fee - best_exit_fee
    else:
        oracle_sell_pnl = -entry_fee
    oracle_pnl = oracle_sell_pnl
    if settlement_result is not None:
        oracle_settle_pnl = settle_position_pnl(position, settlement_result, fees_cfg)
        oracle_pnl = max(oracle_pnl, oracle_settle_pnl)
    hypo_state = EngineState(position_state=position.state, position=position)

    realistic_pnl: Decimal | None = None
    for j in range(entry_index + 1, len(trace)):
        _, snap_j = trace[j]
        if snap_j.now < entry_time:
            continue
        result = step(hypo_state, snap_j, engine_cfg, fees_cfg, vol_spike_limit)
        hypo_state = result.state
        if result.signal is not None and result.signal.action is Action.SELL:
            exit_delay = sample_delay_s(backtest_cfg, rng)
            exit_fill = simulate_fill(result.signal, book_history, exit_delay)
            if exit_fill is not None:
                exit_fee = _round_trip_fee(exit_fill.price, exit_fill.size, fees_cfg)
                realistic_pnl = (exit_fill.price - entry_price) * size - entry_fee - exit_fee
                break

    if realistic_pnl is None:
        if settlement_result is None:
            return None
        realistic_pnl = settle_position_pnl(position, settlement_result, fees_cfg)

    return realistic_pnl, oracle_pnl


def simulate_hypothetical_trade(
    side: Side,
    entry_index: int,
    trace: Sequence[tuple[EngineState, MarketSnapshot]],
    book_history: BookHistory,
    engine_cfg: EngineConfig,
    fees_cfg: FeesConfig,
    vol_spike_limit: float,
    backtest_cfg: BacktestConfig,
    settlement_result: str | None,
    rng: random.Random,
) -> tuple[Decimal, Decimal] | None:
    """What if the user had bought `side` at `trace[entry_index]`? Returns
    `(realistic_net_pnl, oracle_net_pnl)`, or None if the hypothetical
    entry itself would never have filled. The realistic leg runs the exact
    same exit rules (`state_machine.step`) forward against the real
    subsequent snapshots — not reimplemented — settling at the window's
    recorded result if no exit rule fires first. The oracle leg is the best
    exit price actually available afterward, the theoretical ceiling.
    """
    _, entry_snap = trace[entry_index]
    fair_for_side = entry_snap.fair.yes if side is Side.YES else entry_snap.fair.no
    entry_signal = build_signal(
        action=Action.BUY,
        side=side,
        window_ticker=entry_snap.window.ticker,
        book=entry_snap.book,
        fair=fair_for_side,
        size=Decimal(1),
        edge_after_costs=0.0,
        reason="hypothetical",
        now=entry_snap.now,
        ttl_s=engine_cfg.signal_ttl_s,
    )
    delay = sample_delay_s(backtest_cfg, rng)
    fill = simulate_fill(entry_signal, book_history, delay)
    if fill is None:
        return None

    return simulate_from_entry(
        side,
        fill.price,
        fill.logged_at,
        entry_index,
        trace,
        book_history,
        engine_cfg,
        fees_cfg,
        vol_spike_limit,
        backtest_cfg,
        settlement_result,
        rng,
    )


def scan_for_opportunities(
    window_ticker: str,
    trace: Sequence[tuple[EngineState, MarketSnapshot]],
    book_history: BookHistory,
    settlement_result: str | None,
    engine_cfg: EngineConfig,
    fees_cfg: FeesConfig,
    vol_spike_limit: float,
    backtest_cfg: BacktestConfig,
    min_opportunity: Decimal,
    rng: random.Random,
) -> tuple[list[Opportunity], list[FilterEvent]]:
    opportunities: list[Opportunity] = []
    filter_events: list[FilterEvent] = []

    # A single blocked mispricing typically persists for many consecutive
    # seconds (e.g. below_enter_edge for the whole time a small gap sits
    # there). Simulating and reporting every one of those seconds would
    # flood the results with near-duplicates of the same episode, so only
    # the instant a side's reason first appears (or changes) is reported;
    # an unbroken run under the same reason is one opportunity, not one per
    # second it persisted.
    active_reason: dict[Side, str | None] = {Side.YES: None, Side.NO: None}

    for i, (state_before, snap) in enumerate(trace):
        reasons = classify_candidates(state_before, snap, engine_cfg, fees_cfg, vol_spike_limit)
        for side in (Side.YES, Side.NO):
            reason = reasons.get(side)
            if reason == active_reason[side]:
                continue  # same ongoing episode (or still None) -- already handled
            active_reason[side] = reason
            if reason is None:
                continue  # the engine actually took this -- not a miss
            outcome = simulate_hypothetical_trade(
                side,
                i,
                trace,
                book_history,
                engine_cfg,
                fees_cfg,
                vol_spike_limit,
                backtest_cfg,
                settlement_result,
                rng,
            )
            if outcome is None:
                continue  # wouldn't even have filled -- not actionable
            realistic_pnl, oracle_pnl = outcome
            filter_events.append(FilterEvent(window_ticker, side, snap.now, reason, realistic_pnl))
            if realistic_pnl >= min_opportunity:
                opportunities.append(
                    Opportunity(window_ticker, side, snap.now, reason, realistic_pnl, oracle_pnl)
                )

    return opportunities, filter_events


def scan_for_no_signal_opportunities(
    window_ticker: str,
    trace: Sequence[tuple[EngineState, MarketSnapshot]],
    book_history: BookHistory,
    settlement_result: str | None,
    calibrator: IsotonicCalibrator,
    engine_cfg: EngineConfig,
    fees_cfg: FeesConfig,
    vol_spike_limit: float,
    backtest_cfg: BacktestConfig,
    min_opportunity: Decimal,
    rng: random.Random,
) -> list[Opportunity]:
    """CLAUDE.md's deferred `MISSED_NO_SIGNAL`: a hindsight-profitable trade
    the raw model never flagged as a candidate at all (ask was never below
    raw fair value on either side), so `scan_for_opportunities` never even
    considered it -- there's no filter block to grade, only a blind spot in
    the model itself. Recalibrating fair value (learning/calibrator.py) can
    surface these: if the CALIBRATED fair value clears the ask by more than
    round-trip costs even though the raw one never did, that's exactly the
    edge the raw model was blind to. Reported with reason "no_signal"."""
    opportunities: list[Opportunity] = []
    active: dict[Side, bool] = {Side.YES: False, Side.NO: False}

    for i, (state_before, snap) in enumerate(trace):
        if state_before.pending_signal is not None:
            continue
        if state_before.position_state is not PositionState.FLAT:
            continue

        for side in (Side.YES, Side.NO):
            raw_fair = snap.fair.yes if side is Side.YES else snap.fair.no
            ask = float(snap.book.ask(side))
            had_raw_candidate = raw_fair > ask
            calibrated_fair = calibrator.predict(raw_fair)
            cost = float(round_trip_cost(snap.book, side, fees_cfg))
            calibrated_gap = calibrated_fair - ask - cost
            is_hit = (not had_raw_candidate) and calibrated_gap > engine_cfg.enter_edge

            if is_hit == active[side]:
                continue  # same ongoing episode (or still not a hit)
            active[side] = is_hit
            if not is_hit:
                continue

            outcome = simulate_hypothetical_trade(
                side,
                i,
                trace,
                book_history,
                engine_cfg,
                fees_cfg,
                vol_spike_limit,
                backtest_cfg,
                settlement_result,
                rng,
            )
            if outcome is None:
                continue
            realistic_pnl, oracle_pnl = outcome
            if realistic_pnl >= min_opportunity:
                opportunities.append(
                    Opportunity(
                        window_ticker, side, snap.now, "no_signal", realistic_pnl, oracle_pnl
                    )
                )

    return opportunities


def filter_scorecard(filter_events: Sequence[FilterEvent]) -> list[FilterStats]:
    """A filter's value = losses avoided − profits missed, per CLAUDE.md."""
    by_reason: dict[str, list[FilterEvent]] = {}
    for fe in filter_events:
        by_reason.setdefault(fe.reason, []).append(fe)

    stats = []
    for reason, events in by_reason.items():
        losses_avoided = sum(
            (-e.realistic_net_pnl for e in events if e.realistic_net_pnl <= 0), Decimal("0")
        )
        profits_missed = sum(
            (e.realistic_net_pnl for e in events if e.realistic_net_pnl > 0), Decimal("0")
        )
        stats.append(
            FilterStats(
                reason=reason,
                n=len(events),
                losses_avoided=losses_avoided,
                profits_missed=profits_missed,
            )
        )
    return stats
