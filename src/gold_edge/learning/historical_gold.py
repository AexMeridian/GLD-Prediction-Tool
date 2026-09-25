"""Offline "instinct" check for the fair-value model against years of real,
free, keyless GLD (Pax gold ETF) price history from Yahoo Finance's public
chart endpoint.

This is deliberately NOT a trading backtest. There is no historical Kalshi
orderbook, spread, or fee data for past dates -- only live recording (see
`record`/`backtest`) can produce that, and CLAUDE.md's promotion gates
require real recorded round trips, not synthetic ones. What real historical
spot-gold prices CAN do honestly is check whether the fair-value formula
itself (`model/fair_value.py`, imported unmodified -- never re-derived here)
is well-calibrated, and whether `config.yaml`'s volatility defaults are in
the right ballpark, over far more observations (thousands of days) than a
few weeks of live recording can offer. See docs/historical_calibration.md
for the full explanation and its limitations, especially:

- GLD is an ETF, not spot gold. It only trades NYSE hours (roughly 9:30am-
  4:00pm ET) and can gap on open relative to overnight spot moves, unlike
  the 24/5+ price Kalshi's contracts actually settle on. The intraday check
  below only ever pairs bars within a short elapsed time (`max_gap_minutes`)
  specifically to avoid stretching a high-frequency vol estimate across an
  overnight/weekend gap it was never fit on.
- Yahoo's free chart endpoint silently downsamples `interval=1d` once the
  requested `range` gets very long (e.g. `range=max` returns ~monthly bars
  for a 20-year span, not daily) -- verified against the live endpoint
  before writing this. Bounded ranges (`5y`, `10y`, `20y`) return true daily
  bars. Keep `daily_range` at or under ~20y.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

import httpx

from gold_edge.config import ModelConfig, VolatilityConfig
from gold_edge.learning.calibrator import (
    VolMultiplierTable,
    brier_score,
    fit_isotonic_calibrator,
    fit_vol_multipliers,
    log_loss,
    should_promote_calibrator,
)
from gold_edge.learning.patterns import _bucket_vol_regime, _session_for_hour
from gold_edge.model.fair_value import compute_fair_value
from gold_edge.model.volatility import VolatilityTracker

YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/"
_YAHOO_HEADERS = {"User-Agent": "Mozilla/5.0"}
_WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


@dataclass(frozen=True)
class PriceBar:
    timestamp: datetime
    close: float


def parse_yahoo_chart_json(payload: dict) -> list[PriceBar]:
    """Pure parser for Yahoo's chart response shape:
    `chart.result[0].timestamp` (unix seconds) paired with
    `chart.result[0].indicators.quote[0].close` (nullable during gaps,
    e.g. a halted or not-yet-traded bar -- those are dropped)."""
    try:
        result = payload["chart"]["result"][0]
        timestamps = result["timestamp"]
        closes = result["indicators"]["quote"][0]["close"]
    except (KeyError, IndexError, TypeError):
        return []
    bars = []
    for ts, close in zip(timestamps, closes, strict=False):
        if close is None:
            continue
        bars.append(PriceBar(timestamp=datetime.fromtimestamp(ts, tz=UTC), close=float(close)))
    return bars


async def fetch_yahoo_chart(
    symbol: str,
    interval: str,
    range_: str,
    base_url: str = YAHOO_CHART_URL,
) -> list[PriceBar]:
    """Free, keyless historical bars for `symbol` from Yahoo Finance. Used
    for both the long daily history (`interval="1d"`, bounded `range_` such
    as "20y") and the short recent intraday history (`interval="5m"`,
    `range_="60d"`) -- see module docstring for the downsampling and NYSE-
    hours caveats of each."""
    async with httpx.AsyncClient(timeout=15.0, headers=_YAHOO_HEADERS) as client:
        resp = await client.get(
            f"{base_url}{symbol}", params={"interval": interval, "range": range_}
        )
        resp.raise_for_status()
    return parse_yahoo_chart_json(resp.json())


def _stdev(values: Sequence[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    return math.sqrt(variance)


@dataclass(frozen=True)
class RealizedVolSummary:
    n_bars: int
    n_returns: int
    overall_sigma: float
    by_weekday: dict[str, float]


def summarize_realized_vol(bars: Sequence[PriceBar], min_bucket_n: int = 30) -> RealizedVolSummary:
    """Log-return stdev per bar-to-bar step (a daily sigma for daily bars, a
    per-bar sigma for intraday bars), overall and by weekday. Weekday
    buckets below `min_bucket_n` are omitted rather than shown on too little
    data, matching CLAUDE.md's pattern-mining rule."""
    if len(bars) < 2:
        return RealizedVolSummary(len(bars), 0, 0.0, {})
    returns: list[float] = []
    by_weekday: dict[str, list[float]] = {}
    for prev, cur in zip(bars, bars[1:], strict=False):
        if prev.close <= 0 or cur.close <= 0:
            continue
        r = math.log(cur.close / prev.close)
        returns.append(r)
        by_weekday.setdefault(_WEEKDAYS[cur.timestamp.weekday()], []).append(r)
    overall = _stdev(returns) if returns else 0.0
    weekday_sigma = {wd: _stdev(rs) for wd, rs in by_weekday.items() if len(rs) >= min_bucket_n}
    return RealizedVolSummary(len(bars), len(returns), overall, weekday_sigma)


def _round_label(ts: datetime, round_by: str) -> str:
    if round_by == "year":
        return str(ts.year)
    if round_by == "day":
        return ts.strftime("%Y-%m-%d")
    raise ValueError(f"unknown round_by: {round_by!r}")


@dataclass(frozen=True)
class CalibrationPoint:
    predicted_yes: float
    outcome: float
    tau_minutes: float
    sigma_per_minute: float
    z: float
    session: str
    round_label: str


def build_calibration_points(
    bars: Sequence[PriceBar],
    vol_cfg: VolatilityConfig,
    model_cfg: ModelConfig,
    window_bars: int,
    warmup_bars: int = 20,
    max_window_minutes: float | None = None,
    round_by: str | None = None,
) -> list[CalibrationPoint]:
    """Treats every `window_bars`-wide slice `[i, i+window_bars]` as one
    Kalshi-style window: `bars[i]` is the fixed open (S0), `bars[i+window_bars]`
    is the close used only to determine the outcome, and the model is
    evaluated once per INTERIOR bar `k` in `(i, i+window_bars)` using
    `bars[k].close` as the live "current" price S at that moment -- exactly
    how the live engine recomputes fair value continuously as tau shrinks
    during a real window. `VolatilityTracker`/`compute_fair_value` are
    imported unmodified so this can never silently drift from what the live
    engine actually computes, and sigma at bar k only ever uses data up to
    and including bar k (no lookahead).

    The window's own close bar is deliberately never used as "current price"
    -- doing so would feed the future settlement price into the model as if
    it were live, which makes S and the outcome the same observation and
    produces a tautological, not a calibration, result (this was a real bug
    caught by inspecting a first draft's output: every bucket showed exactly
    0% or 100% actual win rate). `warmup_bars` skips the tracker's cold
    start. `max_window_minutes` drops windows whose open-to-close elapsed
    time is implausible for the horizon being studied (see module
    docstring: an intraday NYSE-hours-only series has a large overnight/
    weekend gap between every session's last and next bar). `round_by`
    ("year" or "day", or None to leave it blank) tags each point with which
    chronological round it belongs to (by the window's open time), for
    `run_walkforward_rounds` below -- "multiple rounds of analysis" over
    multiple past sessions, not one single train/test split."""
    tracker = VolatilityTracker(
        vol_cfg.ewma_half_life_s, vol_cfg.short_horizon_s, vol_cfg.min_sigma_per_minute
    )
    sigmas: list[float] = []
    for bar in bars:
        tracker.update(bar.close, bar.timestamp)
        sigmas.append(tracker.sigma_per_minute)

    points: list[CalibrationPoint] = []
    for i in range(len(bars)):
        close_idx = i + window_bars
        if i < warmup_bars or close_idx >= len(bars):
            continue
        s0_bar, close_bar = bars[i], bars[close_idx]
        window_minutes = (close_bar.timestamp - s0_bar.timestamp).total_seconds() / 60.0
        if window_minutes <= 0:
            continue
        if max_window_minutes is not None and window_minutes > max_window_minutes:
            continue
        outcome = 1.0 if close_bar.close >= s0_bar.close else 0.0
        label = _round_label(s0_bar.timestamp, round_by) if round_by else ""
        for k in range(i + 1, close_idx):
            cur_bar = bars[k]
            tau_minutes = (close_bar.timestamp - cur_bar.timestamp).total_seconds() / 60.0
            if tau_minutes <= 0:
                continue
            sigma = sigmas[k]
            fv = compute_fair_value(
                s=cur_bar.close,
                s0=s0_bar.close,
                sigma_per_minute=sigma,
                tau_minutes=tau_minutes,
                min_fair_value=model_cfg.min_fair_value,
                max_fair_value=model_cfg.max_fair_value,
            )
            # sigma is floored > 0 by VolatilityTracker, so this never
            # divides by zero the way fair_value.py itself has to guard for.
            z = math.log(cur_bar.close / s0_bar.close) / (sigma * math.sqrt(tau_minutes))
            points.append(
                CalibrationPoint(
                    predicted_yes=fv.yes,
                    outcome=outcome,
                    tau_minutes=tau_minutes,
                    sigma_per_minute=sigma,
                    z=z,
                    session=_session_for_hour(cur_bar.timestamp.hour),
                    round_label=label,
                )
            )
    return points


def _bucket_table(points: Sequence[CalibrationPoint], bucket_width: float) -> list[dict]:
    buckets: dict[int, list[CalibrationPoint]] = {}
    for p in points:
        idx = math.floor(p.predicted_yes / bucket_width + 1e-9)
        buckets.setdefault(idx, []).append(p)
    table = []
    for idx in sorted(buckets):
        entries = buckets[idx]
        n = len(entries)
        table.append(
            {
                "bucket_start": round(idx * bucket_width, 10),
                "bucket_end": round((idx + 1) * bucket_width, 10),
                "n": n,
                "mean_predicted": sum(p.predicted_yes for p in entries) / n,
                "actual_win_rate": sum(p.outcome for p in entries) / n,
            }
        )
    return table


@dataclass(frozen=True)
class HistoricalCalibrationResult:
    n_pairs: int
    raw_brier: float
    raw_log_loss: float
    holdout_n: int
    holdout_raw_brier: float | None
    holdout_raw_log_loss: float | None
    holdout_calibrated_brier: float | None
    holdout_calibrated_log_loss: float | None
    calibrator_would_help: bool | None
    bucket_table: list[dict]


def evaluate_historical_calibration(
    points: Sequence[CalibrationPoint],
    bucket_width: float = 0.1,
    train_fraction: float = 0.8,
    min_holdout_n: int = 30,
) -> HistoricalCalibrationResult:
    """Full-sample Brier/log-loss for the raw (untrained) formula, plus -- if
    there's enough data on both sides of a chronological split -- a
    held-out comparison against an isotonic recalibration fit only on the
    training slice, exactly mirroring `calibrator.py`'s own train/held-out
    contract (`should_promote_calibrator`) so "would this help" means the
    same thing here as it does for the live promotion path."""
    if not points:
        return HistoricalCalibrationResult(0, 0.0, 0.0, 0, None, None, None, None, None, [])

    predicted = [p.predicted_yes for p in points]
    outcomes = [p.outcome for p in points]
    raw_brier = brier_score(predicted, outcomes)
    raw_ll = log_loss(predicted, outcomes)

    split = int(len(points) * train_fraction)
    train, test = points[:split], points[split:]

    holdout_n = 0
    holdout_raw_brier = holdout_raw_ll = None
    holdout_cal_brier = holdout_cal_ll = None
    would_help = None
    if len(train) >= min_holdout_n and len(test) >= min_holdout_n:
        calibrator = fit_isotonic_calibrator(
            [p.predicted_yes for p in train], [p.outcome for p in train]
        )
        test_pred = [p.predicted_yes for p in test]
        test_outcomes = [p.outcome for p in test]
        test_calibrated = calibrator.predict_many(test_pred)

        holdout_n = len(test)
        holdout_raw_brier = brier_score(test_pred, test_outcomes)
        holdout_raw_ll = log_loss(test_pred, test_outcomes)
        holdout_cal_brier = brier_score(test_calibrated, test_outcomes)
        holdout_cal_ll = log_loss(test_calibrated, test_outcomes)
        would_help = should_promote_calibrator(test_pred, test_calibrated, test_outcomes)

    return HistoricalCalibrationResult(
        n_pairs=len(points),
        raw_brier=raw_brier,
        raw_log_loss=raw_ll,
        holdout_n=holdout_n,
        holdout_raw_brier=holdout_raw_brier,
        holdout_raw_log_loss=holdout_raw_ll,
        holdout_calibrated_brier=holdout_cal_brier,
        holdout_calibrated_log_loss=holdout_cal_ll,
        calibrator_would_help=would_help,
        bucket_table=_bucket_table(points, bucket_width),
    )


def format_historical_report(
    label: str, vol_summary: RealizedVolSummary, calibration: HistoricalCalibrationResult
) -> str:
    lines = [f"=== {label} ===", f"bars: {vol_summary.n_bars}  returns: {vol_summary.n_returns}"]
    lines.append(f"realized sigma per step (log-return stdev): {vol_summary.overall_sigma:.5f}")
    if vol_summary.by_weekday:
        wd_str = "  ".join(f"{wd}={s:.5f}" for wd, s in vol_summary.by_weekday.items())
        lines.append(f"by weekday (n >= min_bucket_n only): {wd_str}")

    lines.append("")
    lines.append(
        f"calibration pairs (fair-value formula vs. actual outcome): {calibration.n_pairs}"
    )
    if calibration.n_pairs == 0:
        lines.append("  not enough data")
        return "\n".join(lines)

    lines.append(
        f"  full-sample: Brier={calibration.raw_brier:.4f}  log-loss={calibration.raw_log_loss:.4f}"
    )
    if calibration.calibrator_would_help is None:
        lines.append(
            "  (not enough data for a held-out recalibration check; need "
            ">= min_holdout_n pairs on each side of the train/test split)"
        )
    else:
        lines.append(
            f"  held-out (n={calibration.holdout_n}): "
            f"raw Brier={calibration.holdout_raw_brier:.4f} "
            f"log-loss={calibration.holdout_raw_log_loss:.4f}"
            f"  |  isotonic-recalibrated Brier={calibration.holdout_calibrated_brier:.4f} "
            f"log-loss={calibration.holdout_calibrated_log_loss:.4f}"
        )
        verdict = "WOULD improve" if calibration.calibrator_would_help else "would NOT improve"
        lines.append(f"  -> recalibration {verdict} held-out calibration on this sample")

    lines.append("")
    lines.append("bucket: mean predicted fair_yes vs. actual win rate")
    for b in calibration.bucket_table:
        lines.append(
            f"  [{b['bucket_start']:.2f}, {b['bucket_end']:.2f})  n={b['n']}  "
            f"mean_predicted={b['mean_predicted']:.3f}  actual_win_rate={b['actual_win_rate']:.3f}"
        )
    return "\n".join(lines)


@dataclass(frozen=True)
class RoundResult:
    label: str
    n_train: int
    n_test: int
    raw_brier: float
    raw_log_loss: float
    calibrated_brier: float | None
    calibrated_log_loss: float | None
    calibrator_would_help: bool | None


def run_walkforward_rounds(
    points: Sequence[CalibrationPoint], min_test_n: int = 30, min_train_n: int = 100
) -> list[RoundResult]:
    """"Multiple rounds of analysis" over multiple past sessions, per
    CLAUDE.md's walk-forward principle ("train on days 1..k, test on day
    k+1, roll forward. Report only out-of-sample results"): groups `points`
    by `round_label` in the chronological order they were built in (one
    round per day or per year, depending on what `build_calibration_points`
    was called with), then for each round with enough data, fits an
    isotonic calibrator on every STRICTLY EARLIER round pooled together
    (expanding window) and evaluates only on the current round -- a round is
    never in both its own training pool and its own test set. Rounds with
    too few points to be a meaningful test are folded into the training
    pool without being scored on their own (too little data to trust a
    per-round number, but still real data worth training on)."""
    order: list[str] = []
    by_round: dict[str, list[CalibrationPoint]] = {}
    for p in points:
        if p.round_label not in by_round:
            order.append(p.round_label)
            by_round[p.round_label] = []
        by_round[p.round_label].append(p)

    results: list[RoundResult] = []
    train_pool: list[CalibrationPoint] = []
    for label in order:
        test_points = by_round[label]
        if len(test_points) < min_test_n:
            train_pool.extend(test_points)
            continue

        test_pred = [p.predicted_yes for p in test_points]
        test_outcomes = [p.outcome for p in test_points]
        raw_brier = brier_score(test_pred, test_outcomes)
        raw_ll = log_loss(test_pred, test_outcomes)

        cal_brier = cal_ll = None
        would_help = None
        if train_pool and len(train_pool) >= min_train_n:
            calibrator = fit_isotonic_calibrator(
                [p.predicted_yes for p in train_pool], [p.outcome for p in train_pool]
            )
            calibrated = calibrator.predict_many(test_pred)
            cal_brier = brier_score(calibrated, test_outcomes)
            cal_ll = log_loss(calibrated, test_outcomes)
            would_help = should_promote_calibrator(test_pred, calibrated, test_outcomes)

        results.append(
            RoundResult(
                label=label,
                n_train=len(train_pool),
                n_test=len(test_points),
                raw_brier=raw_brier,
                raw_log_loss=raw_ll,
                calibrated_brier=cal_brier,
                calibrated_log_loss=cal_ll,
                calibrator_would_help=would_help,
            )
        )
        train_pool.extend(test_points)
    return results


def format_walkforward_report(label: str, rounds: Sequence[RoundResult]) -> str:
    lines = [f"=== {label}: walk-forward, round by round ===", f"rounds scored: {len(rounds)}"]
    if not rounds:
        lines.append("  not enough data for even one scored round")
        return "\n".join(lines)
    help_count = sum(1 for r in rounds if r.calibrator_would_help)
    scored_for_help = sum(1 for r in rounds if r.calibrator_would_help is not None)
    for r in rounds:
        base = (
            f"  {r.label}  train_n={r.n_train:<6d} test_n={r.n_test:<5d}  "
            f"raw Brier={r.raw_brier:.4f} log-loss={r.raw_log_loss:.4f}"
        )
        if r.calibrator_would_help is None:
            lines.append(base + "  (train pool too small to recalibrate yet)")
        else:
            verdict = "helped" if r.calibrator_would_help else "did not help"
            lines.append(
                base + f"  |  recalibrated Brier={r.calibrated_brier:.4f} "
                f"log-loss={r.calibrated_log_loss:.4f}  -> {verdict}"
            )
    if scored_for_help:
        lines.append(
            f"recalibration helped in {help_count}/{scored_for_help} rounds where it could be "
            "tested -- a low or inconsistent count means the earlier single-split result may "
            "have been a lucky draw, not a real, repeatable improvement."
        )
    return "\n".join(lines)


def fit_gld_session_vol_multipliers(
    points: Sequence[CalibrationPoint], vol_spike_limit: float, min_bucket_n: int
) -> VolMultiplierTable:
    """"Learn behaviors of GLD": a sigma multiplier by (vol_regime, session)
    fit on real historical price action, reusing `calibrator.py`'s exact
    `fit_vol_multipliers` (learned component #2 in CLAUDE.md) and the same
    vol-regime/session bucket definitions `patterns.py` uses for live round
    trips, so a bucket label here means the same thing it would in a live
    pattern report. Session bucketing needs genuine time-of-day variation in
    the input bars -- meaningful for intraday data, degenerate for daily
    bars (every daily close falls in the same NYSE-close session)."""
    bucketed: dict[tuple[str, str], list[tuple[float, float, float]]] = {}
    for p in points:
        regime = _bucket_vol_regime(p.sigma_per_minute, vol_spike_limit)
        bucketed.setdefault((regime, p.session), []).append((p.sigma_per_minute, p.z, p.outcome))
    return fit_vol_multipliers(bucketed, min_bucket_n)


def format_vol_multiplier_report(table: VolMultiplierTable, min_bucket_n: int) -> str:
    lines = [
        "=== Learned GLD volatility multiplier by (vol regime, session) ===",
        f"(buckets need >= {min_bucket_n} observations to get a fitted multiplier; "
        "others default to 1.0x -- not shown)",
    ]
    if not table.multipliers:
        lines.append("  not enough data in any bucket yet")
        return "\n".join(lines)
    for (regime, session), factor in sorted(table.multipliers.items()):
        lines.append(f"  regime={regime:<7s} session={session:<8s} multiplier={factor:.2f}x")
    return "\n".join(lines)
