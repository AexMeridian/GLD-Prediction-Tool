"""Dollar-impact attribution, per CLAUDE.md: "For each losing or missed
trade, assign the dominant cause with an estimated dollar impact: model
error, volatility misestimate, user delay, spread, fees, exit rule, filter,
stale data, or news/vol shock. Session reports roll these up: 'Of -$4.20
today, $2.60 was user delay, $1.10 fees, $0.50 model.'"

This module does not re-derive decision quality -- it reads the grade
`learning/grader.py` already produced (and, for filter blocks, the reason
`learning/opportunities.py` already recorded) and maps that onto exactly one
cause with a dollar figure. A round trip's grade and its attribution can
therefore never disagree about what happened; this module only answers
"which line item does the loss belong to."

Only losing or missed value gets attributed (a winning trade, or a filter
block that correctly saved money, has nothing to explain) -- callers get
`None` back for those and should simply not add them to a rollup.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum

from gold_edge.backtest.replay import RoundTripRecord, SignalRecord
from gold_edge.config import LearningConfig
from gold_edge.learning.grader import Grade, GradeResult
from gold_edge.learning.markouts import Markout
from gold_edge.learning.opportunities import FilterEvent, Opportunity


class Cause(StrEnum):
    MODEL_ERROR = "model_error"
    VOLATILITY = "volatility"
    USER_DELAY = "user_delay"
    SPREAD = "spread"
    FEES = "fees"
    EXIT_RULE = "exit_rule"
    FILTER = "filter"
    STALE_DATA = "stale_data"
    NEWS_VOL_SHOCK = "news_vol_shock"


@dataclass(frozen=True)
class AttributionResult:
    window_ticker: str
    cause: Cause
    dollar_impact: Decimal
    detail: str


# A filter's own reason (learning/opportunities.py's REASONS) is often
# already a precise cause in CLAUDE.md's list -- reusing it beats guessing a
# generic "filter" for a block that was obviously about spread or vol.
_REASON_CAUSE: dict[str, Cause] = {
    "spread_filter": Cause.SPREAD,
    "vol_spike": Cause.VOLATILITY,
    "stale_data": Cause.STALE_DATA,
}


def _looks_like_shock(markouts: Sequence[Markout], threshold: float) -> bool:
    """A GOOD_BUT_UNLUCKY round trip means the *longest*-horizon markout
    confirmed the decision but the trade still lost money -- so somewhere
    between the shortest and longest horizon the market round-tripped. A
    large swing between those two readings is a proxy for "a fast, shock-like
    reversal" rather than "a slow grind the wrong way"; it's a heuristic, not
    a real event-detection signal, and is reported as such."""
    valid = [m for m in markouts if m.edge_markout is not None]
    if len(valid) < 2:
        return False
    shortest = min(valid, key=lambda m: m.horizon_s)
    longest = max(valid, key=lambda m: m.horizon_s)
    swing = longest.edge_markout - shortest.edge_markout  # type: ignore[operator]
    return abs(swing) >= threshold


def attribute_round_trip(
    round_trip: RoundTripRecord,
    grade: GradeResult,
    entry_markouts: Sequence[Markout],
    learning_cfg: LearningConfig,
) -> AttributionResult | None:
    """`entry_markouts` must be the same markouts `grade_round_trip` was
    given for this round trip's entry signal. Returns None for a round trip
    that made money -- COSTS_ATE_EDGE is the one exception, since a
    break-even-or-worse outcome despite a real gross edge is exactly the
    "costs ate it" story CLAUDE.md wants attributed."""
    net_pnl = round_trip.pnl
    primary = grade.primary
    if net_pnl >= 0 and primary is not Grade.COSTS_ATE_EDGE:
        return None

    if primary is Grade.BAD_EXECUTION:
        return AttributionResult(
            round_trip.window_ticker,
            Cause.USER_DELAY,
            abs(net_pnl),
            "edge had already eroded by the time the fill actually happened",
        )
    if primary is Grade.COSTS_ATE_EDGE:
        return AttributionResult(
            round_trip.window_ticker,
            Cause.FEES,
            round_trip.fees_paid,
            "gross edge was real but fees consumed it",
        )
    if primary is Grade.STOPPED_WRONGLY:
        return AttributionResult(
            round_trip.window_ticker,
            Cause.EXIT_RULE,
            abs(net_pnl),
            "stop fired and price reverted past entry afterward",
        )
    if primary is Grade.STOPPED_CORRECTLY:
        return AttributionResult(
            round_trip.window_ticker,
            Cause.MODEL_ERROR,
            abs(net_pnl),
            "entry read didn't hold up; the stop correctly limited the damage",
        )
    if primary is Grade.BAD_MODEL:
        return AttributionResult(
            round_trip.window_ticker,
            Cause.MODEL_ERROR,
            abs(net_pnl),
            "market moved against the call -- fair value was wrong",
        )
    if primary is Grade.GOOD_BUT_UNLUCKY:
        shock = _looks_like_shock(entry_markouts, learning_cfg.news_shock_threshold)
        cause = Cause.NEWS_VOL_SHOCK if shock else Cause.VOLATILITY
        detail = (
            "a sudden move reversed a real edge"
            if shock
            else "edge was real but realized volatility exceeded the model's estimate"
        )
        return AttributionResult(round_trip.window_ticker, cause, abs(net_pnl), detail)

    return None  # GOOD_CALL / LUCKY: not a loss to explain


def attribute_missed_signal(
    signal_record: SignalRecord, grade: Grade, realistic_pnl: Decimal
) -> AttributionResult | None:
    """Symmetric to `attribute_round_trip`, for a signal that was issued but
    never filled. SKIP_WAS_RIGHT means there's nothing to attribute."""
    if grade is not Grade.MISSED_BY_USER:
        return None
    return AttributionResult(
        signal_record.signal.window_ticker,
        Cause.USER_DELAY,
        realistic_pnl,
        "signal expired or was skipped and would have profited after delay and fees",
    )


def attribute_filter_event(filter_event: FilterEvent, grade: Grade) -> AttributionResult | None:
    """FILTER_SAVED_US means the filter did its job -- nothing to attribute.
    MISSED_BY_FILTER maps to the specific cause its blocking reason already
    names when one applies (spread/vol/stale), falling back to the generic
    FILTER cause for the engine's other gates (cooldown, entry cutoff, etc.)."""
    if grade is not Grade.MISSED_BY_FILTER:
        return None
    cause = _REASON_CAUSE.get(filter_event.reason, Cause.FILTER)
    return AttributionResult(
        filter_event.window_ticker,
        cause,
        filter_event.realistic_net_pnl,
        f"blocked by '{filter_event.reason}' but would have profited",
    )


def attribute_no_signal_opportunity(opportunity: Opportunity) -> AttributionResult:
    """A MISSED_NO_SIGNAL opportunity is, by construction, a blind spot in
    the raw fair-value model (opportunities.scan_for_no_signal_opportunities
    only reports it once the raw model failed to see a real, cost-clearing
    edge) -- always model_error, always attributed."""
    return AttributionResult(
        opportunity.window_ticker,
        Cause.MODEL_ERROR,
        opportunity.realistic_net_pnl,
        "raw model never flagged this as a candidate at all; calibration would have",
    )


def summarize_attribution(results: Sequence[AttributionResult]) -> dict[Cause, Decimal]:
    """Session/weekly rollup: total dollar impact per cause."""
    totals: dict[Cause, Decimal] = {}
    for r in results:
        totals[r.cause] = totals.get(r.cause, Decimal("0")) + r.dollar_impact
    return totals


def _fmt_dollars(amount: Decimal) -> str:
    sign = "-" if amount < 0 else ""
    return f"{sign}${abs(amount):.2f}"


def format_attribution_report(total_net_pnl: Decimal, by_cause: dict[Cause, Decimal]) -> str:
    """Deterministic, numbers-first text matching CLAUDE.md's example
    verbatim shape: "Of -$4.20 today, $2.60 was user delay, $1.10 fees,
    $0.50 model." Causes with zero (or negative -- shouldn't happen, but
    never silently drop the number) impact are omitted rather than padded
    in with a $0.00 line nobody asked for."""
    nonzero = sorted(
        ((c, a) for c, a in by_cause.items() if a > 0), key=lambda kv: kv[1], reverse=True
    )
    if not nonzero:
        return f"Net P&L: {_fmt_dollars(total_net_pnl)}. No attributable losses or misses."
    parts = [
        f"{_fmt_dollars(amount)} was {cause.value.replace('_', ' ')}" for cause, amount in nonzero
    ]
    return f"Of {_fmt_dollars(total_net_pnl)} today, " + ", ".join(parts) + "."
