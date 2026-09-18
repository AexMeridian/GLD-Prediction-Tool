"""Grades decision quality, not just outcome, per CLAUDE.md: "a trade can
lose money and still be a good decision... or make money and be a bad
decision." Every filled round trip gets exactly one primary grade plus
optional tags, built from its markouts (learning/markouts.py) rather than
raw P&L alone. Missed/skipped signals and engine-filter blocks get their
own grades too, using learning/opportunities.py's hypothetical-trade
simulation to answer "would it have profited?"

`MISSED_NO_SIGNAL` (a hindsight opportunity the raw model never flagged as
mispriced at all) is graded straight from
`opportunities.scan_for_no_signal_opportunities`'s output, once a
calibrator exists to define "mispriced" beyond the raw model's own blind
spot -- see `grade_no_signal_opportunity` below.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import StrEnum

from gold_edge.backtest.fills import BookHistory
from gold_edge.backtest.replay import RoundTripRecord, SignalRecord
from gold_edge.config import BacktestConfig, EngineConfig, FeesConfig, LearningConfig
from gold_edge.engine.state_machine import EngineState, MarketSnapshot, fee_for
from gold_edge.learning.markouts import Markout
from gold_edge.learning.opportunities import FilterEvent, Opportunity, simulate_from_entry
from gold_edge.models import Side


class Grade(StrEnum):
    GOOD_CALL = "GOOD_CALL"
    GOOD_BUT_UNLUCKY = "GOOD_BUT_UNLUCKY"
    LUCKY = "LUCKY"
    BAD_MODEL = "BAD_MODEL"
    BAD_EXECUTION = "BAD_EXECUTION"
    COSTS_ATE_EDGE = "COSTS_ATE_EDGE"
    STOPPED_CORRECTLY = "STOPPED_CORRECTLY"
    STOPPED_WRONGLY = "STOPPED_WRONGLY"
    EXIT_TOO_EARLY = "EXIT_TOO_EARLY"
    EXIT_TOO_LATE = "EXIT_TOO_LATE"
    MISSED_BY_USER = "MISSED_BY_USER"
    SKIP_WAS_RIGHT = "SKIP_WAS_RIGHT"
    MISSED_BY_FILTER = "MISSED_BY_FILTER"
    FILTER_SAVED_US = "FILTER_SAVED_US"
    MISSED_NO_SIGNAL = "MISSED_NO_SIGNAL"


@dataclass(frozen=True)
class GradeResult:
    primary: Grade
    tags: list[Grade] = field(default_factory=list)


def _reference_markout(markouts: Sequence[Markout] | None) -> Markout | None:
    """The longest-horizon markout with real data — the most-confirmed read
    of whether the market ultimately moved toward or away from the call."""
    if not markouts:
        return None
    valid = [m for m in markouts if m.edge_markout is not None]
    if not valid:
        return None
    return max(valid, key=lambda m: m.horizon_s)


def _closest_markout(markouts: Sequence[Markout] | None, target_horizon_s: float) -> Markout | None:
    if not markouts:
        return None
    valid = [m for m in markouts if m.edge_markout is not None]
    if not valid:
        return None
    return min(valid, key=lambda m: abs(m.horizon_s - target_horizon_s))


def _oracle_exit_value_between(
    book_history: BookHistory, side: Side, start: datetime, end: datetime, fees_cfg: FeesConfig
) -> Decimal | None:
    at_start = book_history.at_or_before(start)
    later = book_history.between(start, end)
    candidates = ([at_start] if at_start is not None else []) + later
    if not candidates:
        return None
    best_bid = max(b.bid(side) for b in candidates)
    return best_bid - fee_for(best_bid, fees_cfg)


def grade_round_trip(
    round_trip: RoundTripRecord,
    entry_markouts: Sequence[Markout],
    exit_markouts: Sequence[Markout] | None,
    book_history: BookHistory | None,
    fees_cfg: FeesConfig,
    learning_cfg: LearningConfig,
) -> GradeResult | None:
    """`entry_markouts` must be `compute_markouts` run on `round_trip.entry_signal`;
    `exit_markouts` the same for `round_trip.exit_signal` (pass None when
    there wasn't one — a position held to settlement). `book_history` is
    only needed for the EXIT_TOO_LATE tag (pass None to skip it). Returns
    None if there's no usable markout data yet (e.g. graded too soon after
    the window closed for any horizon to have elapsed)."""
    exit_signal = round_trip.exit_signal
    is_stop = exit_signal is not None and exit_signal.reason.startswith("stop")

    if is_stop:
        ref = _reference_markout(exit_markouts)
        if ref is None or ref.edge_markout is None:
            return None
        primary = Grade.STOPPED_CORRECTLY if ref.edge_markout > 0 else Grade.STOPPED_WRONGLY
        return GradeResult(primary=primary)

    entry_ref = _reference_markout(entry_markouts)
    if entry_ref is None or entry_ref.edge_markout is None:
        return None
    decision_positive = entry_ref.edge_markout > 0

    gross_pnl = round_trip.pnl + round_trip.fees_paid
    net_pnl = round_trip.pnl

    # Was the edge already visibly eroding by the moment the fill actually
    # happened (not just at signal creation)? Uses the entry markout whose
    # horizon is closest to the real fill delay — a big enough drop by then
    # points at execution timing rather than the model or the market being
    # wrong, so it's checked before the outcome-based grades below.
    bad_execution = False
    if round_trip.entry_signal is not None:
        delay_s = (round_trip.entry_time - round_trip.entry_signal.created_at).total_seconds()
        at_fill = _closest_markout(entry_markouts, delay_s)
        if at_fill is not None and at_fill.edge_markout is not None:
            bad_execution = at_fill.edge_markout < -learning_cfg.exit_regret

    if bad_execution:
        primary = Grade.BAD_EXECUTION
    elif gross_pnl > 0 and net_pnl <= 0:
        primary = Grade.COSTS_ATE_EDGE
    elif decision_positive and net_pnl > 0:
        primary = Grade.GOOD_CALL
    elif decision_positive and net_pnl <= 0:
        primary = Grade.GOOD_BUT_UNLUCKY
    elif not decision_positive and net_pnl > 0:
        primary = Grade.LUCKY
    else:
        primary = Grade.BAD_MODEL

    tags: list[Grade] = []
    if exit_signal is not None:
        exit_ref = _reference_markout(exit_markouts)
        if (
            exit_ref is not None
            and exit_ref.edge_markout is not None
            and exit_ref.edge_markout > learning_cfg.exit_regret
        ):
            tags.append(Grade.EXIT_TOO_EARLY)

        if book_history is not None and round_trip.exit_price is not None:
            side = Side.YES if round_trip.side == Side.YES.value else Side.NO
            best_exit_value = _oracle_exit_value_between(
                book_history, side, round_trip.entry_time, round_trip.exit_time, fees_cfg
            )
            actual_exit_value = round_trip.exit_price - fee_for(round_trip.exit_price, fees_cfg)
            if best_exit_value is not None and (best_exit_value - actual_exit_value) > Decimal(
                str(learning_cfg.exit_regret)
            ):
                tags.append(Grade.EXIT_TOO_LATE)

    return GradeResult(primary=primary, tags=tags)


def grade_missed_signal(
    signal_record: SignalRecord,
    entry_index: int,
    trace: Sequence[tuple[EngineState, MarketSnapshot]],
    book_history: BookHistory,
    engine_cfg: EngineConfig,
    fees_cfg: FeesConfig,
    vol_spike_limit: float,
    backtest_cfg: BacktestConfig,
    settlement_result: str | None,
    rng: random.Random,
    learning_cfg: LearningConfig,
) -> Grade | None:
    """Grades a signal that was issued but never filled (expired, missed,
    or explicitly skipped): CLAUDE.md's MISSED_BY_USER ("would have
    profited after delay+fees") vs SKIP_WAS_RIGHT. `entry_index` must be
    the `trace` position matching `signal_record.signal.created_at` — the
    entry is assumed to have succeeded at the signal's own limit price
    (`opportunities.simulate_from_entry`), since grading "should you have
    caught this" requires assuming you did, not re-rolling whether it
    would have filled (that's what made it a miss in the first place)."""
    signal = signal_record.signal
    outcome = simulate_from_entry(
        signal.side,
        signal.limit_price,
        signal.created_at,
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
    if outcome is None:
        return None
    realistic_pnl, _oracle_pnl = outcome
    return (
        Grade.MISSED_BY_USER
        if realistic_pnl >= learning_cfg.min_opportunity
        else Grade.SKIP_WAS_RIGHT
    )


def grade_filter_event(filter_event: FilterEvent) -> Grade:
    """CLAUDE.md's filter-block grades: symmetric to MISSED_BY_USER /
    SKIP_WAS_RIGHT, but for entries an engine filter (not the user) blocked.
    Uses the same realistic_net_pnl sign `filter_scorecard` aggregates on,
    so a window's grade counts and its filter scorecard always agree."""
    return Grade.MISSED_BY_FILTER if filter_event.realistic_net_pnl > 0 else Grade.FILTER_SAVED_US


def grade_no_signal_opportunity(opportunity: Opportunity) -> Grade:
    """Every `Opportunity` `scan_for_no_signal_opportunities` reports has
    already cleared MIN_OPPORTUNITY net of delay and fees (that's what makes
    it an Opportunity, unlike a `FilterEvent`, which is logged regardless of
    profitability) -- so unlike filter blocks there's no "the blind spot
    saved us" counterpart here to weigh against. It's always a miss."""
    return Grade.MISSED_NO_SIGNAL
