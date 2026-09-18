"""Turns a BacktestResult into the metrics CLAUDE.md's Backtesting section
requires: per-window and overall signals/fills/misses/round-trips/fees,
gross and net P&L, max drawdown, P&L per round trip, a comparison to a
do-nothing baseline (always $0 — no trades, no fees), and a calibration
report bucketing predicted fair value against actual settlement outcomes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal

from gold_edge.backtest.replay import BacktestResult, WindowReplayResult
from gold_edge.models import Action

DO_NOTHING_BASELINE_PNL = Decimal("0")


@dataclass(frozen=True)
class Summary:
    window_count: int
    signals_issued: int
    signals_filled: int
    signals_missed: int
    round_trip_count: int
    total_fees: Decimal
    net_pnl: Decimal
    gross_pnl: Decimal
    max_drawdown: Decimal
    baseline_pnl: Decimal = DO_NOTHING_BASELINE_PNL

    @property
    def pnl_per_round_trip(self) -> Decimal:
        if self.round_trip_count == 0:
            return Decimal("0")
        return self.net_pnl / self.round_trip_count

    @property
    def fill_rate(self) -> float:
        if self.signals_issued == 0:
            return 0.0
        return self.signals_filled / self.signals_issued


def build_summary(result: BacktestResult) -> Summary:
    signals_issued = 0
    signals_filled = 0
    all_round_trips = []
    for wr in result.windows:
        signals_issued += len(wr.signal_records)
        signals_filled += sum(1 for sr in wr.signal_records if sr.filled)
        all_round_trips.extend(wr.round_trips)

    total_fees = sum((rt.fees_paid for rt in all_round_trips), Decimal("0"))
    net_pnl = sum((rt.pnl for rt in all_round_trips), Decimal("0"))
    gross_pnl = net_pnl + total_fees

    max_drawdown = Decimal("0")
    peak = Decimal("0")
    cum = Decimal("0")
    for rt in sorted(all_round_trips, key=lambda r: r.exit_time):
        cum += rt.pnl
        peak = max(peak, cum)
        max_drawdown = max(max_drawdown, peak - cum)

    return Summary(
        window_count=len(result.windows),
        signals_issued=signals_issued,
        signals_filled=signals_filled,
        signals_missed=signals_issued - signals_filled,
        round_trip_count=len(all_round_trips),
        total_fees=total_fees,
        net_pnl=net_pnl,
        gross_pnl=gross_pnl,
        max_drawdown=max_drawdown,
    )


def calibration_table(result: BacktestResult, bucket_width: float = 0.1) -> list[dict]:
    """Buckets every BUY signal's fair value (for the side it bought) against
    whether that side actually won at settlement — a model-quality check
    independent of whether the trade was ever filled. Windows with unknown
    settlement are excluded (there's no ground truth to compare against)."""
    buckets: dict[int, list[tuple[float, bool]]] = {}
    for wr in result.windows:
        if wr.settlement_result is None:
            continue
        for sr in wr.signal_records:
            signal = sr.signal
            if signal.action is not Action.BUY:
                continue
            won = wr.settlement_result == signal.side.value.lower()
            idx = math.floor(signal.fair / bucket_width + 1e-9)
            buckets.setdefault(idx, []).append((signal.fair, won))

    table = []
    for idx in sorted(buckets):
        entries = buckets[idx]
        n = len(entries)
        mean_predicted = sum(p for p, _ in entries) / n
        actual_win_rate = sum(1 for _, w in entries if w) / n
        table.append(
            {
                "bucket_start": round(idx * bucket_width, 10),
                "bucket_end": round((idx + 1) * bucket_width, 10),
                "n": n,
                "mean_predicted": mean_predicted,
                "actual_win_rate": actual_win_rate,
            }
        )
    return table


def _fmt_window(wr: WindowReplayResult) -> str:
    filled = sum(1 for sr in wr.signal_records if sr.filled)
    missed = len(wr.signal_records) - filled
    pnl = sum((rt.pnl for rt in wr.round_trips), Decimal("0"))
    fees = sum((rt.fees_paid for rt in wr.round_trips), Decimal("0"))
    lines = [
        f"  {wr.ticker}  settlement={wr.settlement_result or 'unknown'}  "
        f"signals={len(wr.signal_records)} (filled={filled}, missed={missed})  "
        f"round_trips={len(wr.round_trips)}  net_pnl=${pnl}  fees=${fees}"
    ]
    for note in wr.notes:
        lines.append(f"    ! {note}")
    return "\n".join(lines)


def format_report(result: BacktestResult, bucket_width: float = 0.1) -> str:
    summary = build_summary(result)
    lines = ["=== Per-window ===", ""]
    for wr in result.windows:
        lines.append(_fmt_window(wr))
    lines += [
        "",
        "=== Overall ===",
        f"windows: {summary.window_count}",
        f"signals issued: {summary.signals_issued}  filled: {summary.signals_filled}  "
        f"missed: {summary.signals_missed}  (fill rate {summary.fill_rate:.1%})",
        f"round trips: {summary.round_trip_count}",
        f"gross P&L: ${summary.gross_pnl}",
        f"fees paid: ${summary.total_fees}",
        f"net P&L: ${summary.net_pnl}  (do-nothing baseline: ${summary.baseline_pnl})",
        f"P&L per round trip: ${summary.pnl_per_round_trip}",
        f"max drawdown: ${summary.max_drawdown}",
    ]

    calibration = calibration_table(result, bucket_width)
    lines += ["", "=== Calibration (predicted fair value vs. actual win rate) ==="]
    if not calibration:
        lines.append("  no settled signals to calibrate against yet")
    for bucket in calibration:
        lines.append(
            f"  [{bucket['bucket_start']:.2f}, {bucket['bucket_end']:.2f})  "
            f"n={bucket['n']}  mean_predicted={bucket['mean_predicted']:.3f}  "
            f"actual_win_rate={bucket['actual_win_rate']:.3f}"
        )
    return "\n".join(lines)


def format_sweep_report(candidates: list[dict]) -> str:
    """CLAUDE.md: parameter sweeps are 'tuned on training days and reported
    on held-out days only.' Every candidate's held-out result is shown next
    to its training result, ranked by training performance — deliberately
    not just the train-set winner's held-out number, so a candidate that
    only looked good on train (overfit) is visible as such, not hidden."""
    lines = ["=== Parameter sweep (ranked by TRAIN net P&L; held-out results shown alongside) ==="]
    for c in candidates:
        params_str = ", ".join(f"{k}={v}" for k, v in c["params"].items())
        lines.append(f"  {params_str}")
        lines.append(
            f"    train: net_pnl=${c['train_net_pnl']}  round_trips={c['train_round_trips']}"
        )
        lines.append(
            f"    test:  net_pnl=${c['test_net_pnl']}  round_trips={c['test_round_trips']}  "
            f"max_drawdown=${c['test_max_drawdown']}"
        )
    return "\n".join(lines)
