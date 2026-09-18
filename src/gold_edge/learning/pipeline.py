"""Orchestrates one `learn` run, per CLAUDE.md's "Learning pipeline (runs
after sessions, never during a live window)":

1. Compute markouts, grades, opportunities, attribution for new data.
2. Refit learned components on a rolling training window.
3. Evaluate walk-forward: train on days 1..k, test on day k+1.
4. Emit Proposal objects with evidence.

This module is pure/testable without touching SQLite: `run_learning_pipeline`
takes already-loaded ticks/windows/fill-latency samples and returns a
`LearningPipelineResult`; the `learn` CLI command (cli.py) loads that data
from the recorder and persists the result, keeping this module reusable
from tests or a notebook without a live database.

Design note: like `backtest.replay`, this reuses `replay_all` as the ONE
way this codebase reconstructs "what happened" from recorded ticks/books --
there's no separate reconstruction from the raw `signals`/`fills` tables.
Missed-opportunity scanning against a freshly-fit calibrator
(`MISSED_NO_SIGNAL`) is left as a follow-up manual/scheduled step using
`opportunities.scan_for_no_signal_opportunities` directly, once a promoted
calibrator exists -- running it automatically inside every `learn` call
would require a second full replay pass per session, which isn't justified
until a calibrator has actually been promoted.
"""

from __future__ import annotations

import random
from bisect import bisect_left
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal

from gold_edge.backtest.fills import BookHistory
from gold_edge.backtest.replay import RecordedWindow, replay_all
from gold_edge.config import (
    BacktestConfig,
    EngineConfig,
    FeesConfig,
    LearningConfig,
    ModelConfig,
    VolatilityConfig,
)
from gold_edge.engine.state_machine import EngineState, MarketSnapshot
from gold_edge.learning.attribution import (
    AttributionResult,
    attribute_filter_event,
    attribute_missed_signal,
    attribute_round_trip,
)
from gold_edge.learning.calibrator import (
    IsotonicCalibrator,
    fit_isotonic_calibrator,
    should_promote_calibrator,
)
from gold_edge.learning.delay_profile import DelayProfile, FillLatencySample, build_delay_profile
from gold_edge.learning.grader import Grade, GradeResult, grade_filter_event, grade_round_trip
from gold_edge.learning.insights import GradedTrade
from gold_edge.learning.markouts import Markout, compute_markouts
from gold_edge.learning.opportunities import (
    FilterEvent,
    Opportunity,
    scan_for_opportunities,
    simulate_from_entry,
)
from gold_edge.learning.patterns import (
    BucketStat,
    MacroEvent,
    RoundTripFeatures,
    aggregate_patterns,
    build_round_trip_features,
)
from gold_edge.learning.proposer import Proposal, propose_threshold_changes
from gold_edge.models import Tick

# Bounded, per CLAUDE.md's "keep search grids small and bounded to sane
# ranges" -- a fixed default grid over the params CLAUDE.md names.
DEFAULT_PROPOSER_GRID: dict[str, list[float]] = {
    "enter_edge": [0.02, 0.03, 0.04, 0.05],
    "persist_s": [1.0, 1.5, 2.0],
    "stop": [0.06, 0.08, 0.10],
}

MIN_CALIBRATOR_TRAIN_PAIRS = 20


@dataclass(frozen=True)
class MarkoutRecord:
    signal_id: str
    markout: Markout


@dataclass(frozen=True)
class GradedRoundTrip:
    window_ticker: str
    entry_signal_id: str | None
    exit_signal_id: str | None
    grade: GradeResult
    net_pnl: Decimal


@dataclass(frozen=True)
class GradedFilterEvent:
    filter_event: FilterEvent
    grade: Grade


@dataclass(frozen=True)
class LearningPipelineResult:
    markouts: list[MarkoutRecord] = field(default_factory=list)
    graded_round_trips: list[GradedRoundTrip] = field(default_factory=list)
    graded_trades: list[GradedTrade] = field(default_factory=list)  # for insights reports
    attribution_results: list[AttributionResult] = field(default_factory=list)
    opportunities: list[Opportunity] = field(default_factory=list)
    graded_filter_events: list[GradedFilterEvent] = field(default_factory=list)
    pattern_stats: list[BucketStat] = field(default_factory=list)
    calibrator: IsotonicCalibrator | None = None
    calibrator_promoted: bool = False
    delay_profile: DelayProfile = field(default_factory=DelayProfile)
    proposals: list[Proposal] = field(default_factory=list)


def _window_trace(
    trace: list[tuple[EngineState, MarketSnapshot]], ticker: str
) -> list[tuple[EngineState, MarketSnapshot]]:
    return [(s, snap) for s, snap in trace if snap.window.ticker == ticker]


def _find_entry_index(times: list[datetime], at: datetime) -> int | None:
    idx = bisect_left(times, at)
    return idx if idx < len(times) else None


def _settlement_outcome(side_value: str, settlement_result: str | None) -> float | None:
    if settlement_result is None:
        return None
    return 1.0 if settlement_result == side_value.lower() else 0.0


def run_learning_pipeline(
    ticks: list[Tick],
    windows: list[RecordedWindow],
    engine_cfg: EngineConfig,
    fees_cfg: FeesConfig,
    vol_cfg: VolatilityConfig,
    model_cfg: ModelConfig,
    backtest_cfg: BacktestConfig,
    learning_cfg: LearningConfig,
    fill_latency_samples: list[FillLatencySample],
    macro_events: list[MacroEvent] | None = None,
    proposer_param_grid: dict[str, list[float]] | None = None,
    seed: int | None = None,
) -> LearningPipelineResult:
    macro_events = macro_events or []
    proposer_param_grid = proposer_param_grid or DEFAULT_PROPOSER_GRID
    rng = random.Random(seed)

    trace: list[tuple[EngineState, MarketSnapshot]] = []
    result = replay_all(
        ticks,
        windows,
        engine_cfg,
        fees_cfg,
        vol_cfg,
        model_cfg,
        backtest_cfg,
        random.Random(seed),
        on_step=lambda s, snap: trace.append((s, snap)),
    )
    windows_by_ticker = {rw.window.ticker: rw for rw in windows}

    markouts: list[MarkoutRecord] = []
    graded_round_trips: list[GradedRoundTrip] = []
    graded_trades: list[GradedTrade] = []
    attribution_results: list[AttributionResult] = []
    all_opportunities: list[Opportunity] = []
    all_graded_filter_events: list[GradedFilterEvent] = []
    features_and_pnl: list[tuple[RoundTripFeatures, Decimal]] = []
    calibrator_x: list[float] = []
    calibrator_y: list[float] = []

    for wr in result.windows:
        rw = windows_by_ticker[wr.ticker]
        book_history = BookHistory(rw.books)
        window_ticks = [
            t for t in ticks if rw.window.open_time <= t.receive_time <= rw.window.close_time
        ]
        w_trace = _window_trace(trace, wr.ticker)
        w_times = [snap.now for _, snap in w_trace]

        prior_exit_time: datetime | None = None
        for rt in wr.round_trips:
            entry_markouts = None
            exit_markouts = None
            if rt.entry_signal is not None:
                entry_markouts = compute_markouts(
                    rt.entry_signal,
                    rw.window,
                    window_ticks,
                    book_history,
                    learning_cfg.markout_horizons_s,
                    model_cfg,
                    vol_cfg,
                )
                markouts.extend(MarkoutRecord(rt.entry_signal.id, m) for m in entry_markouts)

                outcome = _settlement_outcome(rt.entry_signal.side.value, wr.settlement_result)
                if outcome is not None:
                    calibrator_x.append(rt.entry_signal.fair)
                    calibrator_y.append(outcome)

            if rt.exit_signal is not None:
                exit_markouts = compute_markouts(
                    rt.exit_signal,
                    rw.window,
                    window_ticks,
                    book_history,
                    learning_cfg.markout_horizons_s,
                    model_cfg,
                    vol_cfg,
                )
                markouts.extend(MarkoutRecord(rt.exit_signal.id, m) for m in exit_markouts)

            if entry_markouts is not None:
                grade = grade_round_trip(
                    rt, entry_markouts, exit_markouts, book_history, fees_cfg, learning_cfg
                )
                if grade is not None:
                    graded_round_trips.append(
                        GradedRoundTrip(
                            window_ticker=wr.ticker,
                            entry_signal_id=rt.entry_signal.id if rt.entry_signal else None,
                            exit_signal_id=rt.exit_signal.id if rt.exit_signal else None,
                            grade=grade,
                            net_pnl=rt.pnl,
                        )
                    )
                    graded_trades.append(
                        GradedTrade(wr.ticker, grade.primary, rt.pnl, rt.fees_paid)
                    )
                    attr = attribute_round_trip(rt, grade, entry_markouts, learning_cfg)
                    if attr is not None:
                        attribution_results.append(attr)

            features = build_round_trip_features(
                rt,
                w_times,
                w_trace,
                fees_cfg,
                vol_cfg.vol_spike_limit,
                prior_exit_time,
                macro_events,
            )
            if features is not None:
                features_and_pnl.append((features, rt.pnl))
            prior_exit_time = rt.exit_time

        for sr in wr.signal_records:
            if sr.filled:
                continue
            entry_index = _find_entry_index(w_times, sr.signal.created_at)
            if entry_index is None:
                continue
            # Computed directly via simulate_from_entry (rather than calling
            # grader.grade_missed_signal separately) so the SAME simulated
            # outcome backs both the grade and its attributed dollar amount
            # -- calling it twice would burn the RNG twice and could grade
            # and attribute two different hypothetical fills.
            outcome = simulate_from_entry(
                sr.signal.side,
                sr.signal.limit_price,
                sr.signal.created_at,
                entry_index,
                w_trace,
                book_history,
                engine_cfg,
                fees_cfg,
                vol_cfg.vol_spike_limit,
                backtest_cfg,
                wr.settlement_result,
                rng,
            )
            if outcome is None:
                continue
            realistic_pnl, _oracle_pnl = outcome
            missed_grade = (
                Grade.MISSED_BY_USER
                if realistic_pnl >= Decimal(str(learning_cfg.min_opportunity))
                else Grade.SKIP_WAS_RIGHT
            )
            attr = attribute_missed_signal(sr, missed_grade, realistic_pnl)
            if attr is not None:
                attribution_results.append(attr)

        opportunities, filter_events = scan_for_opportunities(
            wr.ticker,
            w_trace,
            book_history,
            wr.settlement_result,
            engine_cfg,
            fees_cfg,
            vol_cfg.vol_spike_limit,
            backtest_cfg,
            Decimal(str(learning_cfg.min_opportunity)),
            rng,
        )
        all_opportunities.extend(opportunities)
        for fe in filter_events:
            fe_grade = grade_filter_event(fe)
            all_graded_filter_events.append(GradedFilterEvent(fe, fe_grade))
            fe_attr = attribute_filter_event(fe, fe_grade)
            if fe_attr is not None:
                attribution_results.append(fe_attr)

    pattern_stats = aggregate_patterns(features_and_pnl, learning_cfg.min_bucket_n, rng=rng)

    calibrator: IsotonicCalibrator | None = None
    calibrator_promoted = False
    if len(calibrator_x) >= MIN_CALIBRATOR_TRAIN_PAIRS:
        split = int(len(calibrator_x) * 0.7)
        train_x, test_x = calibrator_x[:split], calibrator_x[split:]
        train_y, test_y = calibrator_y[:split], calibrator_y[split:]
        if test_x:
            calibrator = fit_isotonic_calibrator(train_x, train_y)
            calibrated_test = calibrator.predict_many(test_x)
            calibrator_promoted = should_promote_calibrator(test_x, calibrated_test, test_y)

    delay_profile = build_delay_profile(fill_latency_samples)

    proposals: list[Proposal] = []
    sorted_windows = sorted(windows, key=lambda rw: rw.window.open_time)
    if len(sorted_windows) >= 2:
        split = max(1, int(len(sorted_windows) * 0.7))
        train_windows, test_windows = sorted_windows[:split], sorted_windows[split:]
        if train_windows and test_windows:
            train_end = train_windows[-1].window.close_time
            train_ticks = [t for t in ticks if t.receive_time <= train_end]
            test_ticks = [t for t in ticks if t.receive_time > train_end]
            if test_ticks:
                proposals = propose_threshold_changes(
                    train_ticks,
                    train_windows,
                    test_ticks,
                    test_windows,
                    engine_cfg,
                    fees_cfg,
                    vol_cfg,
                    model_cfg,
                    backtest_cfg,
                    proposer_param_grid,
                    seed=seed,
                )

    return LearningPipelineResult(
        markouts=markouts,
        graded_round_trips=graded_round_trips,
        graded_trades=graded_trades,
        attribution_results=attribution_results,
        opportunities=all_opportunities,
        graded_filter_events=all_graded_filter_events,
        pattern_stats=pattern_stats,
        calibrator=calibrator,
        calibrator_promoted=calibrator_promoted,
        delay_profile=delay_profile,
        proposals=proposals,
    )
