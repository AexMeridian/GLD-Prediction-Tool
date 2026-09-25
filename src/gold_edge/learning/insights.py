"""Insights reports, per CLAUDE.md: "Deterministic, template-generated,
numbers-first text (no LLM required)... Every claim shows its sample size.
Small samples are labeled."

Two reports:
- `session_report_card`: trades, grades breakdown, net P&L, fees, best/worst
  call, top missed opportunities, attribution rollup, filter scorecard.
- `weekly_report`: significant patterns (with n and CI), delay profile
  trend, pending proposals, drift status.

Pure string formatting over data these other learning modules already
computed -- nothing here recomputes a grade, a markout, or a pattern.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal

from gold_edge.learning.attribution import AttributionResult, format_attribution_report
from gold_edge.learning.delay_profile import DelayProfile
from gold_edge.learning.drift import DriftEvent
from gold_edge.learning.grader import Grade
from gold_edge.learning.opportunities import FilterStats, Opportunity
from gold_edge.learning.patterns import BucketStat
from gold_edge.learning.proposer import Proposal


@dataclass(frozen=True)
class GradedTrade:
    window_ticker: str
    grade: Grade
    net_pnl: Decimal
    fees_paid: Decimal


def _fmt(amount: Decimal) -> str:
    sign = "-" if amount < 0 else ""
    return f"{sign}${abs(amount):.2f}"


def session_report_card(
    trades: Sequence[GradedTrade],
    attribution: Sequence[AttributionResult],
    opportunities: Sequence[Opportunity],
    filter_stats: Sequence[FilterStats],
    top_n_opportunities: int = 3,
) -> str:
    lines: list[str] = ["=== Session Report Card ===", ""]

    n = len(trades)
    net_pnl = sum((t.net_pnl for t in trades), Decimal("0"))
    total_fees = sum((t.fees_paid for t in trades), Decimal("0"))
    lines.append(f"Trades: {n}  |  Net P&L: {_fmt(net_pnl)}  |  Fees paid: {_fmt(total_fees)}")
    lines.append("")

    if n == 0:
        lines.append("No graded trades this session.")
    else:
        lines.append("Grade breakdown:")
        counts: dict[Grade, int] = {}
        for t in trades:
            counts[t.grade] = counts.get(t.grade, 0) + 1
        for grade, count in sorted(counts.items(), key=lambda kv: kv[1], reverse=True):
            lines.append(f"  {grade.value}: {count}")
        lines.append("")

        best = max(trades, key=lambda t: t.net_pnl)
        worst = min(trades, key=lambda t: t.net_pnl)
        lines.append(f"Best call: {best.window_ticker} ({best.grade.value}, {_fmt(best.net_pnl)})")
        lines.append(
            f"Worst call: {worst.window_ticker} ({worst.grade.value}, {_fmt(worst.net_pnl)})"
        )
        lines.append("")

    if attribution:
        by_cause: dict = {}
        for a in attribution:
            by_cause[a.cause] = by_cause.get(a.cause, Decimal("0")) + a.dollar_impact
        lines.append(format_attribution_report(net_pnl, by_cause))
        lines.append("")

    top_opps = sorted(opportunities, key=lambda o: o.realistic_net_pnl, reverse=True)[
        :top_n_opportunities
    ]
    if top_opps:
        lines.append(f"Top {len(top_opps)} missed opportunities:")
        for o in top_opps:
            lines.append(
                f"  {o.window_ticker} {o.side.value} at {o.at.isoformat()}: "
                f"{_fmt(o.realistic_net_pnl)} missed (reason: {o.reason})"
            )
        lines.append("")

    if filter_stats:
        lines.append("Filter scorecard (losses avoided - profits missed):")
        for fs in sorted(filter_stats, key=lambda s: s.value, reverse=True):
            lines.append(
                f"  {fs.reason} (n={fs.n}): saved {_fmt(fs.losses_avoided)}, "
                f"missed {_fmt(fs.profits_missed)}, net {_fmt(fs.value)}"
            )

    return "\n".join(lines)


def weekly_report(
    pattern_stats: Sequence[BucketStat],
    delay_profile: DelayProfile,
    pending_proposals: Sequence[Proposal],
    drift_event: DriftEvent | None,
    min_bucket_n: int,
) -> str:
    lines: list[str] = ["=== Weekly Report ===", ""]

    significant = [s for s in pattern_stats if s.significant]
    lines.append(f"Significant patterns (n >= {min_bucket_n}, BH-corrected):")
    if not significant:
        lines.append("  None found this week.")
    else:
        for s in significant:
            lines.append(
                f"  {s.dimension}={s.bucket} (n={s.n}): mean P&L {_fmt(s.mean_pnl)} "
                f"[95% CI {_fmt(s.ci_low)}, {_fmt(s.ci_high)}], p={s.p_value:.4f}"
            )
    insufficient = len(pattern_stats) - len(significant)
    if insufficient:
        lines.append(f"  ({insufficient} other bucket(s) had data but did not clear significance)")
    lines.append("")

    lines.append("Delay profile:")
    if delay_profile.is_learned:
        lines.append(
            f"  Learned from {delay_profile.n} fills: mean {delay_profile.mean_delay_s():.2f}s, "
            f"median {delay_profile.percentile_delay_s(50):.2f}s, "
            f"p90 {delay_profile.percentile_delay_s(90):.2f}s"
        )
    else:
        lines.append(
            f"  Not yet learned ({delay_profile.n} fills recorded; needs 50) -- "
            "using the configured default delay range."
        )
    if delay_profile.miss_rate_by_reason:
        lines.append("  Miss rate by signal type (expired or explicitly skipped):")
        for reason, rate in sorted(
            delay_profile.miss_rate_by_reason.items(), key=lambda kv: kv[1], reverse=True
        ):
            lines.append(f"    {reason}: {rate:.0%}")
    lines.append("")

    lines.append(f"Pending proposals: {len(pending_proposals)}")
    for p in pending_proposals:
        lines.append(f"  {p.rationale}")
    lines.append("")

    lines.append("Drift status:")
    if drift_event is None:
        lines.append("  No drift check has run yet.")
    elif not drift_event.has_drift:
        lines.append("  No drift detected.")
    else:
        for m in drift_event.degraded_metrics:
            lines.append(f"  DEGRADED: {m}")
        lines.append(f"  Suggestion: {drift_event.suggestion}")

    return "\n".join(lines)
