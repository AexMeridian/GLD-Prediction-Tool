"""Pattern mining, per CLAUDE.md: "Aggregate grades and net P&L per round
trip across dimensions: minute-of-window, time of day / session, gap size
bucket, side, spread, realized vol regime, distance |S-S0| in sigmas, time
since last trade, proximity to scheduled macro releases, and day of week."
Report only buckets with n >= MIN_BUCKET_N, with bootstrap confidence
intervals, and apply a Benjamini-Hochberg correction before calling anything
significant.

This module only aggregates and tests; it doesn't decide what to do about a
pattern (that's proposer.py's job, reading these `BucketStat`s as evidence).
"""

from __future__ import annotations

import json
import random
from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import numpy as np

from gold_edge.backtest.replay import RoundTripRecord
from gold_edge.config import FeesConfig
from gold_edge.engine.state_machine import EngineState, MarketSnapshot, gap_for_side
from gold_edge.models import Side

DIMENSIONS = (
    "minute_of_window",
    "session",
    "gap_bucket",
    "side",
    "spread_bucket",
    "vol_regime",
    "sigma_distance_bucket",
    "time_since_last_trade_bucket",
    "day_of_week",
    "macro_proximity_bucket",
)


@dataclass(frozen=True)
class RoundTripFeatures:
    minute_of_window: int
    session: str
    gap_bucket: str
    side: str
    spread_bucket: str
    vol_regime: str
    sigma_distance_bucket: str
    time_since_last_trade_bucket: str
    day_of_week: str
    macro_proximity_bucket: str

    def value(self, dimension: str) -> str:
        return str(getattr(self, dimension))


@dataclass(frozen=True)
class BucketStat:
    dimension: str
    bucket: str
    n: int
    mean_pnl: Decimal
    ci_low: Decimal
    ci_high: Decimal
    p_value: float
    significant: bool = False


@dataclass(frozen=True)
class MacroEvent:
    name: str
    at: datetime


def load_macro_events(path: Path) -> list[MacroEvent]:
    """CLAUDE.md: "maintain a simple events calendar file." Format: a JSON
    list of {"name": str, "at": ISO datetime}. Missing/empty file is not an
    error -- every trade just falls into the "none_nearby" bucket, which is
    the honest answer when there's no calendar data yet."""
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [MacroEvent(name=e["name"], at=datetime.fromisoformat(e["at"])) for e in raw]


def _session_for_hour(hour_utc: int) -> str:
    if 0 <= hour_utc < 8:
        return "asia"
    if 8 <= hour_utc < 13:
        return "london"
    if 13 <= hour_utc < 21:
        return "ny"
    return "other"


def _bucket_edges(value: float, edges: Sequence[float], labels: Sequence[str]) -> str:
    for edge, label in zip(edges, labels[:-1], strict=False):
        if value < edge:
            return label
    return labels[-1]


def _bucket_gap(gap: float) -> str:
    return _bucket_edges(gap, [0.03, 0.05, 0.08], ["<0.03", "0.03-0.05", "0.05-0.08", "0.08+"])


def _bucket_spread(spread: Decimal) -> str:
    return _bucket_edges(
        float(spread), [0.01, 0.02, 0.04], ["<0.01", "0.01-0.02", "0.02-0.04", "0.04+"]
    )


def _bucket_vol_regime(sigma: float | None, vol_spike_limit: float) -> str:
    if sigma is None:
        return "unknown"
    if sigma < 0.5 * vol_spike_limit:
        return "low"
    if sigma < vol_spike_limit:
        return "normal"
    return "high"


def _bucket_sigma_distance(
    underlying_price: float | None,
    s0: Decimal | None,
    sigma_per_minute: float | None,
    tau_minutes: float,
) -> str:
    if underlying_price is None or s0 is None or sigma_per_minute is None or sigma_per_minute <= 0:
        return "unknown"
    denom = sigma_per_minute * max(tau_minutes, 1e-6) ** 0.5
    if denom <= 0:
        return "unknown"
    distance = abs(underlying_price - float(s0)) / denom
    return _bucket_edges(distance, [0.5, 1.0, 2.0], ["<0.5σ", "0.5-1σ", "1-2σ", "2σ+"])


def _bucket_time_since_last(seconds: float | None) -> str:
    if seconds is None:
        return "first_trade"
    return _bucket_edges(seconds, [60.0, 300.0], ["<60s", "60-300s", "300s+"])


def _bucket_macro_proximity(at: datetime, events: Sequence[MacroEvent]) -> str:
    if not events:
        return "none_nearby"
    nearest_s = min(abs((e.at - at).total_seconds()) for e in events)
    return _bucket_edges(
        nearest_s / 60.0, [5.0, 15.0, 60.0], ["<5m", "5-15m", "15-60m", "none_nearby"]
    )


def _snapshot_at_or_before(
    times: list[datetime], trace: Sequence[tuple[EngineState, MarketSnapshot]], at: datetime
) -> MarketSnapshot | None:
    idx = bisect_right(times, at) - 1
    if idx < 0:
        return None
    return trace[idx][1]


def build_round_trip_features(
    round_trip: RoundTripRecord,
    times: list[datetime],
    trace: Sequence[tuple[EngineState, MarketSnapshot]],
    fees_cfg: FeesConfig,
    vol_spike_limit: float,
    prior_exit_time: datetime | None,
    macro_events: Sequence[MacroEvent] = (),
) -> RoundTripFeatures | None:
    """None if no snapshot exists at/before the entry time (shouldn't happen
    for a real round trip, but the trace could be incomplete for
    hand-built test fixtures)."""
    snap = _snapshot_at_or_before(times, trace, round_trip.entry_time)
    if snap is None:
        return None

    side = Side.YES if round_trip.side == Side.YES.value else Side.NO
    fair_side = snap.fair.yes if side is Side.YES else snap.fair.no
    gap = gap_for_side(fair_side, snap.book, side, fees_cfg)
    minute_of_window = int((round_trip.entry_time - snap.window.open_time).total_seconds() // 60)
    tau_minutes = snap.window.seconds_left(round_trip.entry_time) / 60.0
    time_since_last = (
        None
        if prior_exit_time is None
        else (round_trip.entry_time - prior_exit_time).total_seconds()
    )

    return RoundTripFeatures(
        minute_of_window=minute_of_window,
        session=_session_for_hour(round_trip.entry_time.hour),
        gap_bucket=_bucket_gap(gap),
        side=round_trip.side,
        spread_bucket=_bucket_spread(snap.book.spread(side)),
        vol_regime=_bucket_vol_regime(snap.short_horizon_sigma_per_minute, vol_spike_limit),
        sigma_distance_bucket=_bucket_sigma_distance(
            snap.underlying_price,
            snap.window.s0,
            snap.short_horizon_sigma_per_minute,
            tau_minutes,
        ),
        time_since_last_trade_bucket=_bucket_time_since_last(time_since_last),
        day_of_week=round_trip.entry_time.strftime("%A"),
        macro_proximity_bucket=_bucket_macro_proximity(round_trip.entry_time, macro_events),
    )


def _bootstrap_mean_ci_pvalue(
    values: Sequence[Decimal], n_boot: int, rng: random.Random
) -> tuple[Decimal, Decimal, Decimal, float]:
    arr = np.array([float(v) for v in values])
    n = len(arr)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        boot_means[i] = arr[idx].mean()
    mean = float(arr.mean())
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
    p_value = min(1.0, 2 * min(float(np.mean(boot_means <= 0)), float(np.mean(boot_means >= 0))))
    return (
        Decimal(str(round(mean, 4))),
        Decimal(str(round(float(ci_low), 4))),
        Decimal(str(round(float(ci_high), 4))),
        p_value,
    )


def compute_bucket_stats(
    dimension: str,
    pnls_by_bucket: dict[str, list[Decimal]],
    min_bucket_n: int,
    n_boot: int = 2000,
    rng: random.Random | None = None,
) -> list[BucketStat]:
    """One BucketStat per bucket with n >= min_bucket_n; buckets below that
    are silently dropped here -- CLAUDE.md's "insufficient data" label is a
    presentation concern (insights.py), not a statistic to compute on too
    few samples."""
    rng = rng or random.Random()
    stats = []
    for bucket, pnls in pnls_by_bucket.items():
        if len(pnls) < min_bucket_n:
            continue
        mean, ci_low, ci_high, p_value = _bootstrap_mean_ci_pvalue(pnls, n_boot, rng)
        stats.append(
            BucketStat(
                dimension=dimension,
                bucket=bucket,
                n=len(pnls),
                mean_pnl=mean,
                ci_low=ci_low,
                ci_high=ci_high,
                p_value=p_value,
            )
        )
    return stats


def apply_benjamini_hochberg(stats: Sequence[BucketStat], alpha: float = 0.05) -> list[BucketStat]:
    """Standard BH step-up procedure across ALL bucket p-values passed in
    together -- CLAUDE.md: "before calling any pattern significant," which
    reads as one multiple-comparisons family across the whole pattern report,
    not one family per dimension (that would understate the correction)."""
    m = len(stats)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: stats[i].p_value)
    threshold_rank = 0
    for rank, idx in enumerate(order, start=1):
        if stats[idx].p_value <= (rank / m) * alpha:
            threshold_rank = rank
    significant_idx = {order[i] for i in range(threshold_rank)}
    return [replace(s, significant=(i in significant_idx)) for i, s in enumerate(stats)]


def aggregate_patterns(
    features_and_pnl: Sequence[tuple[RoundTripFeatures, Decimal]],
    min_bucket_n: int,
    alpha: float = 0.05,
    n_boot: int = 2000,
    rng: random.Random | None = None,
) -> list[BucketStat]:
    """The full pipeline: bucket by every dimension, compute bootstrap stats
    for buckets with enough data, then apply one BH correction across all of
    them. Returns only buckets that cleared MIN_BUCKET_N; `significant`
    marks which of those also survive the multiple-comparisons correction."""
    rng = rng or random.Random()
    all_stats: list[BucketStat] = []
    for dimension in DIMENSIONS:
        by_bucket: dict[str, list[Decimal]] = {}
        for features, pnl in features_and_pnl:
            by_bucket.setdefault(features.value(dimension), []).append(pnl)
        all_stats.extend(compute_bucket_stats(dimension, by_bucket, min_bucket_n, n_boot, rng))
    return apply_benjamini_hochberg(all_stats, alpha)
