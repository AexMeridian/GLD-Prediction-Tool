import random
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from gold_edge.backtest.fills import BookHistory
from gold_edge.backtest.replay import RoundTripRecord, SignalRecord
from gold_edge.config import LearningConfig
from gold_edge.engine.signals import build_signal
from gold_edge.engine.state_machine import EngineState
from gold_edge.learning.grader import (
    Grade,
    grade_filter_event,
    grade_missed_signal,
    grade_round_trip,
)
from gold_edge.learning.markouts import Markout
from gold_edge.learning.opportunities import FilterEvent
from gold_edge.model.fair_value import FairValue
from gold_edge.models import Action, Side
from tests.test_backtest_replay import bt_cfg
from tests.test_state_machine import book, engine_cfg, fees_cfg, window

T0 = datetime(2026, 1, 1, tzinfo=UTC)


def learning_cfg(**overrides) -> LearningConfig:
    defaults = dict(
        min_opportunity=0.03,
        min_bucket_n=30,
        exit_regret=0.02,
        min_proposal_trades=200,
        promotion_margin=0.01,
        max_dd_worsen=0.10,
        shadow_sessions=10,
        filter_min_prob=0.55,
        markout_horizons_s=[5.0, 15.0, 30.0, 60.0],
        news_shock_threshold=0.08,
    )
    defaults.update(overrides)
    return LearningConfig(**defaults)


def markout(horizon_s: float, edge_markout: float | None) -> Markout:
    return Markout(
        horizon_s=horizon_s,
        at=T0 + timedelta(seconds=horizon_s),
        pyth_price=2000.0,
        market_mid=0.5,
        market_bid=0.49,
        market_ask=0.51,
        fair=0.5,
        edge_markout=edge_markout,
    )


def exit_signal(reason: str, now=T0) -> object:
    return build_signal(
        action=Action.SELL,
        side=Side.YES,
        window_ticker="KXGOLD15M-TEST",
        book=book(yes_bid="0.55", now=now),
        fair=0.55,
        size=Decimal(1),
        edge_after_costs=0.01,
        reason=reason,
        now=now,
        ttl_s=6.0,
    )


def entry_buy_signal(now=T0, limit="0.50") -> object:
    return build_signal(
        action=Action.BUY,
        side=Side.YES,
        window_ticker="KXGOLD15M-TEST",
        book=book(yes_ask=limit, now=now),
        fair=0.55,
        size=Decimal(1),
        edge_after_costs=0.04,
        reason="enter_edge",
        now=now,
        ttl_s=6.0,
    )


def rt(
    pnl: str,
    fees: str,
    exit_sig=None,
    entry_sig=None,
    entry_time=T0,
    exit_price="0.55",
    exit_time=None,
) -> RoundTripRecord:
    return RoundTripRecord(
        window_ticker="KXGOLD15M-TEST",
        side="YES",
        size=Decimal(1),
        entry_price=Decimal("0.50"),
        entry_time=entry_time,
        exit_price=Decimal(exit_price) if exit_price is not None else None,
        exit_time=exit_time or (entry_time + timedelta(seconds=30)),
        exit_reason="sold" if exit_sig is not None else "settled",
        pnl=Decimal(pnl),
        fees_paid=Decimal(fees),
        exit_signal=exit_sig,
        entry_signal=entry_sig,
    )


class TestGradeRoundTrip:
    def test_good_call_positive_markout_positive_net_pnl(self):
        rtr = rt(pnl="1.00", fees="0.10")
        result = grade_round_trip(
            rtr, [markout(30.0, 0.10)], None, None, fees_cfg(), learning_cfg()
        )
        assert result.primary == Grade.GOOD_CALL

    def test_good_but_unlucky_positive_markout_negative_net_pnl(self):
        rtr = rt(pnl="-0.50", fees="0.10")
        result = grade_round_trip(
            rtr, [markout(30.0, 0.10)], None, None, fees_cfg(), learning_cfg()
        )
        assert result.primary == Grade.GOOD_BUT_UNLUCKY

    def test_lucky_negative_markout_positive_net_pnl(self):
        rtr = rt(pnl="0.50", fees="0.10")
        result = grade_round_trip(
            rtr, [markout(30.0, -0.10)], None, None, fees_cfg(), learning_cfg()
        )
        assert result.primary == Grade.LUCKY

    def test_bad_model_negative_markout_negative_net_pnl(self):
        rtr = rt(pnl="-0.50", fees="0.10")
        result = grade_round_trip(
            rtr, [markout(30.0, -0.10)], None, None, fees_cfg(), learning_cfg()
        )
        assert result.primary == Grade.BAD_MODEL

    def test_costs_ate_edge_takes_precedence_over_good_but_unlucky(self):
        # gross = pnl + fees = -0.02 + 0.10 = 0.08 > 0, net = -0.02 <= 0.
        rtr = rt(pnl="-0.02", fees="0.10")
        result = grade_round_trip(
            rtr, [markout(30.0, 0.10)], None, None, fees_cfg(), learning_cfg()
        )
        assert result.primary == Grade.COSTS_ATE_EDGE

    def test_stopped_correctly_when_price_kept_moving_against_position(self):
        stop = exit_signal("stop")
        rtr = rt(pnl="-0.80", fees="0.10", exit_sig=stop)
        result = grade_round_trip(
            rtr, [markout(30.0, 0.10)], [markout(30.0, 0.15)], None, fees_cfg(), learning_cfg()
        )
        assert result.primary == Grade.STOPPED_CORRECTLY

    def test_stopped_wrongly_when_price_reverts_past_entry(self):
        stop = exit_signal("stop")
        rtr = rt(pnl="-0.80", fees="0.10", exit_sig=stop)
        result = grade_round_trip(
            rtr, [markout(30.0, 0.10)], [markout(30.0, -0.15)], None, fees_cfg(), learning_cfg()
        )
        assert result.primary == Grade.STOPPED_WRONGLY

    def test_exit_too_early_tag_on_non_stop_exit_beyond_regret_threshold(self):
        converged = exit_signal("converged")
        rtr = rt(pnl="0.50", fees="0.10", exit_sig=converged)
        result = grade_round_trip(
            rtr,
            [markout(30.0, 0.10)],
            [markout(30.0, 0.10)],
            None,
            fees_cfg(),
            learning_cfg(exit_regret=0.02),
        )
        assert Grade.EXIT_TOO_EARLY in result.tags

    def test_no_exit_too_early_tag_below_regret_threshold(self):
        converged = exit_signal("converged")
        rtr = rt(pnl="0.50", fees="0.10", exit_sig=converged)
        result = grade_round_trip(
            rtr,
            [markout(30.0, 0.10)],
            [markout(30.0, 0.01)],
            None,
            fees_cfg(),
            learning_cfg(exit_regret=0.02),
        )
        assert Grade.EXIT_TOO_EARLY not in result.tags

    def test_no_exit_too_early_tag_possible_on_stop_exit(self):
        stop = exit_signal("stop")
        rtr = rt(pnl="-0.80", fees="0.10", exit_sig=stop)
        result = grade_round_trip(
            rtr, [markout(30.0, 0.10)], [markout(30.0, 10.0)], None, fees_cfg(), learning_cfg()
        )
        assert result.tags == []

    def test_settled_trade_with_no_exit_signal_still_grades_on_entry_markout(self):
        rtr = rt(pnl="0.50", fees="0.02", exit_sig=None)
        result = grade_round_trip(
            rtr, [markout(30.0, 0.10)], None, None, fees_cfg(), learning_cfg()
        )
        assert result.primary == Grade.GOOD_CALL
        assert result.tags == []

    def test_returns_none_when_no_valid_entry_markout(self):
        rtr = rt(pnl="0.50", fees="0.10")
        result = grade_round_trip(
            rtr, [markout(30.0, None)], None, None, fees_cfg(), learning_cfg()
        )
        assert result is None

    def test_uses_largest_horizon_with_valid_data_as_reference(self):
        rtr = rt(pnl="1.00", fees="0.10")
        # 60s horizon has no data (None); 30s does -> must fall back to 30s,
        # not silently return None or use an earlier, less-informative one.
        result = grade_round_trip(
            rtr,
            [markout(15.0, -0.5), markout(30.0, 0.10), markout(60.0, None)],
            None,
            None,
            fees_cfg(),
            learning_cfg(),
        )
        assert result.primary == Grade.GOOD_CALL

    def test_bad_execution_when_edge_erodes_before_fill(self):
        """Even though the long-horizon markout looks great, the edge had
        already visibly reversed by the moment the (delayed) fill actually
        happened -- that should dominate the outcome-based grades."""
        sig = entry_buy_signal(now=T0)
        rtr = rt(pnl="0.50", fees="0.10", entry_sig=sig, entry_time=T0 + timedelta(seconds=1.5))
        entry_markouts = [markout(1.5, -0.10), markout(30.0, 0.20)]
        result = grade_round_trip(
            rtr, entry_markouts, None, None, fees_cfg(), learning_cfg(exit_regret=0.02)
        )
        assert result.primary == Grade.BAD_EXECUTION

    def test_no_bad_execution_when_edge_holds_at_fill(self):
        sig = entry_buy_signal(now=T0)
        rtr = rt(pnl="1.00", fees="0.10", entry_sig=sig, entry_time=T0 + timedelta(seconds=1.5))
        entry_markouts = [markout(1.5, 0.05), markout(30.0, 0.20)]
        result = grade_round_trip(
            rtr, entry_markouts, None, None, fees_cfg(), learning_cfg(exit_regret=0.02)
        )
        assert result.primary == Grade.GOOD_CALL

    def test_no_bad_execution_check_without_an_entry_signal(self):
        # entry_signal is None (the default) -> the check can't run at all,
        # not "runs and finds nothing wrong".
        rtr = rt(pnl="1.00", fees="0.10")
        entry_markouts = [markout(1.5, -0.50), markout(30.0, 0.20)]
        result = grade_round_trip(
            rtr, entry_markouts, None, None, fees_cfg(), learning_cfg(exit_regret=0.02)
        )
        assert result.primary == Grade.GOOD_CALL

    def test_exit_too_late_tag_when_better_price_was_available_earlier(self):
        converged = exit_signal("converged")
        rtr = rt(pnl="0.03", fees="0.10", exit_sig=converged, exit_price="0.55")
        better_book = book(yes_bid="0.70", yes_ask="0.72", now=T0 + timedelta(seconds=15))
        book_history = BookHistory([better_book])
        result = grade_round_trip(
            rtr,
            [markout(30.0, 0.10)],
            [markout(30.0, 0.01)],
            book_history,
            fees_cfg(),
            learning_cfg(exit_regret=0.02),
        )
        assert Grade.EXIT_TOO_LATE in result.tags

    def test_no_exit_too_late_tag_when_no_better_price_existed(self):
        converged = exit_signal("converged")
        rtr = rt(pnl="0.03", fees="0.10", exit_sig=converged, exit_price="0.55")
        same_book = book(yes_bid="0.50", yes_ask="0.52", now=T0 + timedelta(seconds=15))
        book_history = BookHistory([same_book])
        result = grade_round_trip(
            rtr,
            [markout(30.0, 0.10)],
            [markout(30.0, 0.01)],
            book_history,
            fees_cfg(),
            learning_cfg(exit_regret=0.02),
        )
        assert Grade.EXIT_TOO_LATE not in result.tags

    def test_no_exit_too_late_check_without_book_history(self):
        converged = exit_signal("converged")
        rtr = rt(pnl="0.03", fees="0.10", exit_sig=converged, exit_price="0.55")
        result = grade_round_trip(
            rtr,
            [markout(30.0, 0.10)],
            [markout(30.0, 0.01)],
            None,
            fees_cfg(),
            learning_cfg(exit_regret=0.02),
        )
        assert Grade.EXIT_TOO_LATE not in result.tags


class TestGradeMissedSignal:
    def _trace_and_books(self, close_in_s=600.0):
        from tests.test_state_machine import snap as make_snap

        win = window(close_in_s=close_in_s)
        bk = book(yes_bid="0.48", yes_ask="0.50", now=T0)
        entry_snap = make_snap(now=T0, win=win, bk=bk, fair=FairValue(yes=0.95, no=0.05))
        trace = [(EngineState(), entry_snap)]
        return trace, BookHistory([bk])

    def test_missed_by_user_when_would_have_profited(self):
        trace, book_history = self._trace_and_books()
        sig = entry_buy_signal(now=T0)
        sr = SignalRecord(signal=sig, filled=False, fill=None)
        grade = grade_missed_signal(
            sr,
            0,
            trace,
            book_history,
            engine_cfg(),
            fees_cfg(),
            0.5,
            bt_cfg(human_delay_min_s=1.0, human_delay_max_s=1.0),
            "yes",
            random.Random(0),
            learning_cfg(),
        )
        assert grade == Grade.MISSED_BY_USER

    def test_skip_was_right_when_would_have_lost(self):
        trace, book_history = self._trace_and_books()
        sig = entry_buy_signal(now=T0)
        sr = SignalRecord(signal=sig, filled=False, fill=None)
        grade = grade_missed_signal(
            sr,
            0,
            trace,
            book_history,
            engine_cfg(),
            fees_cfg(),
            0.5,
            bt_cfg(human_delay_min_s=1.0, human_delay_max_s=1.0),
            "no",
            random.Random(0),
            learning_cfg(),
        )
        assert grade == Grade.SKIP_WAS_RIGHT

    def test_none_when_outcome_cannot_be_determined(self):
        trace, book_history = self._trace_and_books()
        sig = entry_buy_signal(now=T0)
        sr = SignalRecord(signal=sig, filled=False, fill=None)
        grade = grade_missed_signal(
            sr,
            0,
            trace,
            book_history,
            engine_cfg(),
            fees_cfg(),
            0.5,
            bt_cfg(human_delay_min_s=1.0, human_delay_max_s=1.0),
            None,  # settlement unknown, and no exit fires within this short trace
            random.Random(0),
            learning_cfg(),
        )
        assert grade is None


class TestGradeFilterEvent:
    def test_missed_by_filter_when_profitable(self):
        fe = FilterEvent("W", Side.YES, T0, "below_enter_edge", Decimal("0.50"))
        assert grade_filter_event(fe) == Grade.MISSED_BY_FILTER

    def test_filter_saved_us_when_unprofitable(self):
        fe = FilterEvent("W", Side.YES, T0, "spread_filter", Decimal("-0.20"))
        assert grade_filter_event(fe) == Grade.FILTER_SAVED_US

    def test_filter_saved_us_when_exactly_zero(self):
        fe = FilterEvent("W", Side.YES, T0, "cooldown", Decimal("0"))
        assert grade_filter_event(fe) == Grade.FILTER_SAVED_US


class TestGradeNoSignalOpportunity:
    def test_always_missed_no_signal(self):
        from gold_edge.learning.grader import grade_no_signal_opportunity
        from gold_edge.learning.opportunities import Opportunity

        opp = Opportunity("W", Side.YES, T0, "no_signal", Decimal("0.30"), Decimal("0.40"))
        assert grade_no_signal_opportunity(opp) == Grade.MISSED_NO_SIGNAL
