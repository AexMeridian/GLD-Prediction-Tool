import random
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from gold_edge.backtest.fills import BookHistory
from gold_edge.backtest.replay import RecordedWindow, replay_all
from gold_edge.engine.state_machine import EngineState, MarketSnapshot
from gold_edge.learning.opportunities import (
    FilterEvent,
    classify_candidates,
    filter_scorecard,
    scan_for_opportunities,
    simulate_from_entry,
    simulate_hypothetical_trade,
)
from gold_edge.model.fair_value import FairValue
from gold_edge.models import Side, Tick
from tests.test_backtest_replay import bt_cfg, model_cfg, vol_cfg
from tests.test_state_machine import EDGE_BOOK, EDGE_FAIR_YES, book, engine_cfg, fees_cfg, window

T0 = datetime(2026, 1, 1, tzinfo=UTC)


def tick(price: float, t: datetime) -> Tick:
    return Tick(
        symbol="Metal.XAU/USD", price=price, conf=0.1, expo=-2, publish_time=t, receive_time=t
    )


def snap(now=T0, win=None, bk=None, fair=None, pyth_age_s=0.5, kalshi_age_s=0.5, sigma=None):
    return MarketSnapshot(
        now=now,
        window=win or window(),
        book=bk or book(),
        fair=fair or FairValue(yes=0.5, no=0.5),
        pyth_age_s=pyth_age_s,
        kalshi_age_s=kalshi_age_s,
        short_horizon_sigma_per_minute=sigma,
    )


def persisted_state(side: Side, now=T0, persist_s=1.5, **overrides) -> EngineState:
    gap_since = {Side.YES: None, Side.NO: None}
    gap_since[side] = now - timedelta(seconds=persist_s)
    defaults = dict(gap_exceeded_since=gap_since)
    defaults.update(overrides)
    return EngineState(**defaults)


class TestClassifyCandidates:
    def test_no_candidates_when_neither_side_mispriced(self):
        result = classify_candidates(
            EngineState(), snap(fair=FairValue(yes=0.5, no=0.5)), engine_cfg(), fees_cfg(), 0.5
        )
        assert result == {}

    def test_pending_signal_means_no_candidates_evaluated(self):
        from gold_edge.engine.signals import build_signal
        from gold_edge.models import Action

        sig = build_signal(
            action=Action.BUY,
            side=Side.YES,
            window_ticker="KXGOLD15M-TEST",
            book=EDGE_BOOK,
            fair=0.6,
            size=Decimal(1),
            edge_after_costs=0.04,
            reason="enter_edge",
            now=T0,
            ttl_s=6.0,
        )
        state = EngineState(pending_signal=sig)
        result = classify_candidates(
            state, snap(bk=EDGE_BOOK, fair=EDGE_FAIR_YES), engine_cfg(), fees_cfg(), 0.5
        )
        assert result == {}

    def test_already_in_position(self):
        from gold_edge.models import Position, PositionState

        position = Position(
            window_ticker="KXGOLD15M-TEST",
            side=Side.NO,
            size=Decimal(1),
            entry_price=Decimal("0.40"),
            entered_at=T0,
            state=PositionState.LONG_NO,
        )
        state = EngineState(position_state=PositionState.LONG_NO, position=position)
        result = classify_candidates(
            state, snap(bk=EDGE_BOOK, fair=EDGE_FAIR_YES), engine_cfg(), fees_cfg(), 0.5
        )
        assert result[Side.YES] == "already_in_position"

    def test_stale_data_blocks_all_candidates(self):
        result = classify_candidates(
            EngineState(),
            snap(bk=EDGE_BOOK, fair=EDGE_FAIR_YES, pyth_age_s=10.0),
            engine_cfg(),
            fees_cfg(),
            0.5,
        )
        assert result[Side.YES] == "stale_data"

    def test_below_enter_edge_when_gap_after_costs_too_small(self):
        # NO side of EDGE_BOOK/EDGE_FAIR_YES: raw candidate (0.40 fair > 0.32
        # ask) but gap-after-costs is only 0.02, below the 0.03 threshold.
        result = classify_candidates(
            EngineState(), snap(bk=EDGE_BOOK, fair=EDGE_FAIR_YES), engine_cfg(), fees_cfg(), 0.5
        )
        assert result[Side.NO] == "below_enter_edge"

    def test_persistence_when_not_yet_held_long_enough(self):
        state = EngineState(gap_exceeded_since={Side.YES: T0, Side.NO: None})
        result = classify_candidates(
            state,
            snap(now=T0 + timedelta(seconds=0.2), bk=EDGE_BOOK, fair=EDGE_FAIR_YES),
            engine_cfg(),
            fees_cfg(),
            0.5,
        )
        assert result[Side.YES] == "persistence"

    def test_engine_would_take_it_when_all_checks_pass(self):
        state = persisted_state(Side.YES, now=T0)
        result = classify_candidates(
            state, snap(now=T0, bk=EDGE_BOOK, fair=EDGE_FAIR_YES), engine_cfg(), fees_cfg(), 0.5
        )
        assert result[Side.YES] is None

    def test_cooldown_blocks_a_persisted_candidate(self):
        state = persisted_state(Side.YES, now=T0, cooldown_until=T0 + timedelta(seconds=5))
        result = classify_candidates(
            state, snap(now=T0, bk=EDGE_BOOK, fair=EDGE_FAIR_YES), engine_cfg(), fees_cfg(), 0.5
        )
        assert result[Side.YES] == "cooldown"

    def test_entry_cutoff_blocks_a_persisted_candidate(self):
        state = persisted_state(Side.YES, now=T0)
        near_close = window(close_in_s=30.0)  # < default entry_cutoff_s (90)
        result = classify_candidates(
            state,
            snap(now=T0, win=near_close, bk=EDGE_BOOK, fair=EDGE_FAIR_YES),
            engine_cfg(),
            fees_cfg(),
            0.5,
        )
        assert result[Side.YES] == "entry_cutoff"

    def test_spread_filter_blocks_a_persisted_candidate(self):
        # Spread (0.05) just over max_spread (0.04), but fair is high enough
        # (0.95) that gap-after-costs still clears enter_edge on its own —
        # isolating the spread check from below_enter_edge.
        wide_book = book(yes_bid="0.40", yes_ask="0.45", no_bid="0.05", no_ask="0.10")
        high_fair = FairValue(yes=0.95, no=0.05)
        state = persisted_state(Side.YES, now=T0)
        result = classify_candidates(
            state, snap(now=T0, bk=wide_book, fair=high_fair), engine_cfg(), fees_cfg(), 0.5
        )
        assert result[Side.YES] == "spread_filter"

    def test_vol_spike_blocks_a_persisted_candidate(self):
        state = persisted_state(Side.YES, now=T0)
        result = classify_candidates(
            state,
            snap(now=T0, bk=EDGE_BOOK, fair=EDGE_FAIR_YES, sigma=1.0),
            engine_cfg(),
            fees_cfg(),
            0.5,
        )
        assert result[Side.YES] == "vol_spike"

    def test_max_round_trips_blocks_a_persisted_candidate(self):
        state = persisted_state(Side.YES, now=T0, round_trips=6)
        result = classify_candidates(
            state, snap(now=T0, bk=EDGE_BOOK, fair=EDGE_FAIR_YES), engine_cfg(), fees_cfg(), 0.5
        )
        assert result[Side.YES] == "max_round_trips"

    def test_daily_loss_stop_blocks_a_persisted_candidate(self):
        state = persisted_state(Side.YES, now=T0, realized_pnl_today=Decimal("-30"))
        result = classify_candidates(
            state, snap(now=T0, bk=EDGE_BOOK, fair=EDGE_FAIR_YES), engine_cfg(), fees_cfg(), 0.5
        )
        assert result[Side.YES] == "daily_loss_stop"

    def test_fair_value_disagreed_when_both_sides_would_qualify(self):
        symmetric_book = book(yes_bid="0.08", yes_ask="0.10", no_bid="0.08", no_ask="0.10")
        fair = FairValue(yes=0.60, no=0.60)  # deliberately inconsistent, for the test
        state = EngineState(
            gap_exceeded_since={
                Side.YES: T0 - timedelta(seconds=5),
                Side.NO: T0 - timedelta(seconds=5),
            }
        )
        result = classify_candidates(
            state, snap(now=T0, bk=symmetric_book, fair=fair), engine_cfg(), fees_cfg(), 0.5
        )
        assert result[Side.YES] == "fair_value_disagreed"
        assert result[Side.NO] == "fair_value_disagreed"


class TestSimulateHypotheticalTrade:
    def test_returns_none_when_entry_would_not_fill(self):
        win = window(close_in_s=60.0)
        entry_snap = snap(now=T0, win=win, bk=EDGE_BOOK, fair=EDGE_FAIR_YES)
        trace = [(EngineState(), entry_snap)]
        # Book immediately moves the ask up past 0.50 before any delay elapses.
        moved_book = book(yes_ask="0.90", now=T0 + timedelta(seconds=0.1))
        book_history = BookHistory([EDGE_BOOK, moved_book])
        result = simulate_hypothetical_trade(
            Side.YES,
            0,
            trace,
            book_history,
            engine_cfg(),
            fees_cfg(),
            0.5,
            bt_cfg(human_delay_min_s=1.0, human_delay_max_s=1.0),
            "yes",
            random.Random(0),
        )
        assert result is None

    def test_settles_at_close_when_no_exit_fires(self):
        # fair >= hold_to_settlement_fair_threshold (0.90) throughout, so
        # the exit-cutoff rule holds rather than force-selling, and the
        # book never triggers overshoot/converged/stop -> rides to close.
        win = window(close_in_s=10.0)
        high_fair = FairValue(yes=0.95, no=0.05)
        entry_snap = snap(now=T0, win=win, bk=EDGE_BOOK, fair=high_fair)
        later_snap = snap(now=T0 + timedelta(seconds=9), win=win, bk=EDGE_BOOK, fair=high_fair)
        trace = [(EngineState(), entry_snap), (EngineState(), later_snap)]
        steady_book = EDGE_BOOK.model_copy(update={"receive_time": T0 + timedelta(seconds=9)})
        book_history = BookHistory([EDGE_BOOK, steady_book])
        result = simulate_hypothetical_trade(
            Side.YES,
            0,
            trace,
            book_history,
            engine_cfg(),
            fees_cfg(),
            0.5,
            bt_cfg(human_delay_min_s=1.0, human_delay_max_s=1.0),
            "yes",
            random.Random(0),
        )
        assert result is not None
        realistic_pnl, oracle_pnl = result
        # Won at settlement: entry 0.50 -> payoff 1.00, minus entry fee only.
        assert realistic_pnl > Decimal("0.30")


class TestSimulateFromEntry:
    def test_matches_simulate_hypothetical_trade_when_entry_assumed_filled(self):
        """`simulate_hypothetical_trade` is just `simulate_from_entry` given
        a fill that already succeeded -- forcing the same entry price/time
        directly should reproduce the identical outcome."""
        win = window(close_in_s=600.0)
        entry_snap = snap(now=T0, win=win, bk=EDGE_BOOK, fair=FairValue(yes=0.95, no=0.05))
        trace = [(EngineState(), entry_snap)]
        book_history = BookHistory([EDGE_BOOK])
        bcfg = bt_cfg(human_delay_min_s=1.0, human_delay_max_s=1.0)

        via_hypothetical = simulate_hypothetical_trade(
            Side.YES, 0, trace, book_history, engine_cfg(), fees_cfg(), 0.5, bcfg, "yes",
            random.Random(3),
        )
        via_forced_entry = simulate_from_entry(
            Side.YES, Decimal("0.50"), T0 + timedelta(seconds=1), 0, trace, book_history,
            engine_cfg(), fees_cfg(), 0.5, bcfg, "yes", random.Random(3),
        )
        assert via_hypothetical == via_forced_entry

    def test_returns_none_when_no_exit_and_no_settlement_known(self):
        win = window(close_in_s=600.0)
        entry_snap = snap(now=T0, win=win, bk=EDGE_BOOK, fair=FairValue(yes=0.95, no=0.05))
        trace = [(EngineState(), entry_snap)]
        book_history = BookHistory([EDGE_BOOK])
        result = simulate_from_entry(
            Side.YES, Decimal("0.50"), T0, 0, trace, book_history, engine_cfg(), fees_cfg(),
            0.5, bt_cfg(human_delay_min_s=1.0, human_delay_max_s=1.0), None, random.Random(0),
        )
        assert result is None


class TestFilterScorecard:
    def test_aggregates_losses_avoided_and_profits_missed(self):
        events = [
            FilterEvent("W", Side.YES, T0, "spread_filter", Decimal("-1.00")),
            FilterEvent("W", Side.YES, T0, "spread_filter", Decimal("0.50")),
            FilterEvent("W", Side.NO, T0, "cooldown", Decimal("-2.00")),
        ]
        stats = {s.reason: s for s in filter_scorecard(events)}
        assert stats["spread_filter"].n == 2
        assert stats["spread_filter"].losses_avoided == Decimal("1.00")
        assert stats["spread_filter"].profits_missed == Decimal("0.50")
        assert stats["spread_filter"].value == Decimal("0.50")
        assert stats["cooldown"].losses_avoided == Decimal("2.00")
        assert stats["cooldown"].value == Decimal("2.00")

    def test_empty_input_returns_empty_list(self):
        assert filter_scorecard([]) == []


class TestScanForOpportunitiesIntegration:
    def test_finds_opportunity_blocked_by_enter_edge_threshold(self):
        """A real replay where the NO side never clears ENTER_EDGE (gap
        stuck at ~0.02, per EDGE_BOOK/EDGE_FAIR_YES) but, because the win
        is later locked in by settlement, would have cleared MIN_OPPORTUNITY
        had the engine ignored its own threshold."""
        # A realistic ~15-minute window (not a short synthetic one): with
        # close_in_s=600 and the default open 300s before T0, time_left
        # never drops below entry/exit cutoff during this test's span, so
        # a hypothetical position rides to settlement deterministically
        # rather than getting cut off immediately (a too-short window was
        # the bug here originally — every hypothetical trade hit an
        # immediate time_cutoff exit no matter when it entered).
        win = window(close_in_s=600.0).model_copy(update={"ticker": "KXGOLD15M-OPP"})
        n = 200
        ticks = [tick(2000.0, T0 + timedelta(seconds=i)) for i in range(n)]
        books = [
            EDGE_BOOK.model_copy(
                update={"receive_time": T0 + timedelta(seconds=i), "window_ticker": win.ticker}
            )
            for i in range(n)
        ]
        rw = RecordedWindow(window=win, books=books, settlement_result="no")

        trace: list[tuple[EngineState, MarketSnapshot]] = []
        replay_all(
            ticks=ticks,
            windows=[rw],
            engine_cfg=engine_cfg(),
            fees_cfg=fees_cfg(),
            vol_cfg=vol_cfg(),
            model_cfg=model_cfg(),
            backtest_cfg=bt_cfg(human_delay_min_s=1.0, human_delay_max_s=1.0),
            rng=random.Random(0),
            on_step=lambda state, s: trace.append((state, s)),
        )
        # The recorded fair value is whatever the real vol/price path
        # produced; force it to EDGE_FAIR_YES for a controlled, known-answer
        # scan, since the point of this test is the opportunity-scan logic,
        # not re-deriving the model's fair value from noisy synthetic ticks.
        trace = [(s, replace(snap_, fair=EDGE_FAIR_YES)) for s, snap_ in trace]

        book_history = BookHistory(books)
        opportunities, filter_events = scan_for_opportunities(
            window_ticker=win.ticker,
            trace=trace,
            book_history=book_history,
            settlement_result="no",
            engine_cfg=engine_cfg(),
            fees_cfg=fees_cfg(),
            vol_spike_limit=0.5,
            backtest_cfg=bt_cfg(human_delay_min_s=1.0, human_delay_max_s=1.0),
            min_opportunity=Decimal("0.03"),
            rng=random.Random(1),
        )
        assert any(fe.reason == "below_enter_edge" and fe.side is Side.NO for fe in filter_events)
        assert any(o.side is Side.NO and o.reason == "below_enter_edge" for o in opportunities)


class TestScanForNoSignalOpportunities:
    def test_finds_a_calibrated_edge_the_raw_model_never_saw(self):
        from gold_edge.learning.calibrator import IsotonicCalibrator
        from gold_edge.learning.opportunities import scan_for_no_signal_opportunities

        # Neither side clears its own ask under the RAW fair value (0.50 ask
        # == 0.50 fair on both sides -- no candidate at all), but a
        # calibrator that's learned raw=0.50 actually means ~0.65 reveals a
        # real, cost-clearing edge the raw model was blind to.
        flat_book = book(yes_bid="0.48", yes_ask="0.50", no_bid="0.48", no_ask="0.50")
        win = window(close_in_s=600.0).model_copy(update={"ticker": "KXGOLD15M-NOSIG"})
        n = 50
        ticks = [tick(2000.0, T0 + timedelta(seconds=i)) for i in range(n)]
        books = [
            flat_book.model_copy(
                update={"receive_time": T0 + timedelta(seconds=i), "window_ticker": win.ticker}
            )
            for i in range(n)
        ]
        rw = RecordedWindow(window=win, books=books, settlement_result="yes")

        trace: list[tuple[EngineState, MarketSnapshot]] = []
        replay_all(
            ticks=ticks,
            windows=[rw],
            engine_cfg=engine_cfg(),
            fees_cfg=fees_cfg(),
            vol_cfg=vol_cfg(),
            model_cfg=model_cfg(),
            backtest_cfg=bt_cfg(human_delay_min_s=1.0, human_delay_max_s=1.0),
            rng=random.Random(0),
            on_step=lambda state, s: trace.append((state, s)),
        )
        flat_fair = FairValue(yes=0.50, no=0.50)
        trace = [(s, replace(snap_, fair=flat_fair)) for s, snap_ in trace]

        calibrator = IsotonicCalibrator(x_thresholds=[0.0, 0.5, 1.0], y_values=[0.0, 0.65, 1.0])
        book_history = BookHistory(books)
        opportunities = scan_for_no_signal_opportunities(
            window_ticker=win.ticker,
            trace=trace,
            book_history=book_history,
            settlement_result="yes",
            calibrator=calibrator,
            engine_cfg=engine_cfg(),
            fees_cfg=fees_cfg(),
            vol_spike_limit=0.5,
            backtest_cfg=bt_cfg(human_delay_min_s=1.0, human_delay_max_s=1.0),
            min_opportunity=Decimal("0.03"),
            rng=random.Random(1),
        )
        assert any(o.side is Side.YES and o.reason == "no_signal" for o in opportunities)

    def test_no_hits_when_calibrator_agrees_with_raw_model(self):
        from gold_edge.learning.calibrator import IsotonicCalibrator
        from gold_edge.learning.opportunities import scan_for_no_signal_opportunities

        flat_book = book(yes_bid="0.48", yes_ask="0.50", no_bid="0.48", no_ask="0.50")
        win = window(close_in_s=600.0).model_copy(update={"ticker": "KXGOLD15M-NOSIG2"})
        n = 20
        ticks = [tick(2000.0, T0 + timedelta(seconds=i)) for i in range(n)]
        books = [
            flat_book.model_copy(
                update={"receive_time": T0 + timedelta(seconds=i), "window_ticker": win.ticker}
            )
            for i in range(n)
        ]
        rw = RecordedWindow(window=win, books=books, settlement_result="yes")
        trace: list[tuple[EngineState, MarketSnapshot]] = []
        replay_all(
            ticks=ticks,
            windows=[rw],
            engine_cfg=engine_cfg(),
            fees_cfg=fees_cfg(),
            vol_cfg=vol_cfg(),
            model_cfg=model_cfg(),
            backtest_cfg=bt_cfg(human_delay_min_s=1.0, human_delay_max_s=1.0),
            rng=random.Random(0),
            on_step=lambda state, s: trace.append((state, s)),
        )
        flat_fair = FairValue(yes=0.50, no=0.50)
        trace = [(s, replace(snap_, fair=flat_fair)) for s, snap_ in trace]

        identity_calibrator = IsotonicCalibrator(x_thresholds=[0.0, 1.0], y_values=[0.0, 1.0])
        book_history = BookHistory(books)
        opportunities = scan_for_no_signal_opportunities(
            window_ticker=win.ticker,
            trace=trace,
            book_history=book_history,
            settlement_result="yes",
            calibrator=identity_calibrator,
            engine_cfg=engine_cfg(),
            fees_cfg=fees_cfg(),
            vol_spike_limit=0.5,
            backtest_cfg=bt_cfg(human_delay_min_s=1.0, human_delay_max_s=1.0),
            min_opportunity=Decimal("0.03"),
            rng=random.Random(1),
        )
        assert opportunities == []
