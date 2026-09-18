from datetime import UTC, datetime, timedelta
from decimal import Decimal

from gold_edge.learning.attribution import (
    Cause,
    attribute_filter_event,
    attribute_missed_signal,
    attribute_no_signal_opportunity,
    attribute_round_trip,
    format_attribution_report,
    summarize_attribution,
)
from gold_edge.learning.grader import Grade, GradeResult
from gold_edge.learning.opportunities import FilterEvent, Opportunity
from gold_edge.models import Side
from tests.test_learning_grader import entry_buy_signal, exit_signal, learning_cfg, markout, rt

T0 = datetime(2026, 1, 1, tzinfo=UTC)


class TestAttributeRoundTrip:
    def test_winning_trade_is_not_attributed(self):
        rtr = rt(pnl="1.00", fees="0.10")
        grade = GradeResult(primary=Grade.GOOD_CALL)
        assert attribute_round_trip(rtr, grade, [markout(30.0, 0.10)], learning_cfg()) is None

    def test_lucky_trade_is_not_attributed(self):
        rtr = rt(pnl="0.50", fees="0.10")
        grade = GradeResult(primary=Grade.LUCKY)
        assert attribute_round_trip(rtr, grade, [markout(30.0, -0.10)], learning_cfg()) is None

    def test_bad_execution_attributes_to_user_delay(self):
        sig = entry_buy_signal(now=T0)
        rtr = rt(pnl="-0.30", fees="0.10", entry_sig=sig, entry_time=T0 + timedelta(seconds=1.5))
        grade = GradeResult(primary=Grade.BAD_EXECUTION)
        entry_markouts = [markout(1.5, -0.10), markout(30.0, 0.20)]
        result = attribute_round_trip(rtr, grade, entry_markouts, learning_cfg())
        assert result.cause == Cause.USER_DELAY
        assert result.dollar_impact == Decimal("0.30")
        assert result.window_ticker == rtr.window_ticker

    def test_costs_ate_edge_attributes_to_fees_using_fees_paid(self):
        rtr = rt(pnl="-0.02", fees="0.10")
        grade = GradeResult(primary=Grade.COSTS_ATE_EDGE)
        result = attribute_round_trip(rtr, grade, [markout(30.0, 0.10)], learning_cfg())
        assert result.cause == Cause.FEES
        assert result.dollar_impact == Decimal("0.10")

    def test_costs_ate_edge_attributed_even_at_exactly_zero_net_pnl(self):
        rtr = rt(pnl="0.00", fees="0.10")
        grade = GradeResult(primary=Grade.COSTS_ATE_EDGE)
        result = attribute_round_trip(rtr, grade, [markout(30.0, 0.10)], learning_cfg())
        assert result is not None
        assert result.cause == Cause.FEES

    def test_stopped_wrongly_attributes_to_exit_rule(self):
        stop = exit_signal("stop")
        rtr = rt(pnl="-0.80", fees="0.10", exit_sig=stop)
        grade = GradeResult(primary=Grade.STOPPED_WRONGLY)
        result = attribute_round_trip(rtr, grade, [markout(30.0, 0.10)], learning_cfg())
        assert result.cause == Cause.EXIT_RULE
        assert result.dollar_impact == Decimal("0.80")

    def test_stopped_correctly_attributes_to_model_error(self):
        stop = exit_signal("stop")
        rtr = rt(pnl="-0.80", fees="0.10", exit_sig=stop)
        grade = GradeResult(primary=Grade.STOPPED_CORRECTLY)
        result = attribute_round_trip(rtr, grade, [markout(30.0, 0.10)], learning_cfg())
        assert result.cause == Cause.MODEL_ERROR
        assert result.dollar_impact == Decimal("0.80")

    def test_bad_model_attributes_to_model_error(self):
        rtr = rt(pnl="-0.50", fees="0.10")
        grade = GradeResult(primary=Grade.BAD_MODEL)
        result = attribute_round_trip(rtr, grade, [markout(30.0, -0.10)], learning_cfg())
        assert result.cause == Cause.MODEL_ERROR
        assert result.dollar_impact == Decimal("0.50")

    def test_good_but_unlucky_attributes_to_volatility_without_a_shock_swing(self):
        rtr = rt(pnl="-0.50", fees="0.10")
        grade = GradeResult(primary=Grade.GOOD_BUT_UNLUCKY)
        # shortest vs longest horizon swing = 0.10 - 0.08 = 0.02, well below
        # the 0.08 shock threshold.
        entry_markouts = [markout(5.0, 0.08), markout(30.0, 0.10)]
        result = attribute_round_trip(rtr, grade, entry_markouts, learning_cfg())
        assert result.cause == Cause.VOLATILITY
        assert result.dollar_impact == Decimal("0.50")

    def test_good_but_unlucky_attributes_to_news_vol_shock_on_a_big_swing(self):
        rtr = rt(pnl="-0.50", fees="0.10")
        grade = GradeResult(primary=Grade.GOOD_BUT_UNLUCKY)
        # shortest vs longest horizon swing = 0.15 - (-0.20) = 0.35 >= 0.08.
        entry_markouts = [markout(5.0, -0.20), markout(30.0, 0.15)]
        result = attribute_round_trip(rtr, grade, entry_markouts, learning_cfg())
        assert result.cause == Cause.NEWS_VOL_SHOCK

    def test_shock_detection_needs_at_least_two_valid_markouts(self):
        rtr = rt(pnl="-0.50", fees="0.10")
        grade = GradeResult(primary=Grade.GOOD_BUT_UNLUCKY)
        result = attribute_round_trip(rtr, grade, [markout(30.0, 0.10)], learning_cfg())
        assert result.cause == Cause.VOLATILITY


class TestAttributeMissedSignal:
    def test_missed_by_user_attributes_to_user_delay_with_realistic_pnl(self):
        from gold_edge.backtest.replay import SignalRecord

        sig = entry_buy_signal(now=T0)
        record = SignalRecord(signal=sig, filled=False, fill=None)
        result = attribute_missed_signal(record, Grade.MISSED_BY_USER, Decimal("0.15"))
        assert result.cause == Cause.USER_DELAY
        assert result.dollar_impact == Decimal("0.15")
        assert result.window_ticker == sig.window_ticker

    def test_skip_was_right_is_not_attributed(self):
        from gold_edge.backtest.replay import SignalRecord

        sig = entry_buy_signal(now=T0)
        record = SignalRecord(signal=sig, filled=False, fill=None)
        assert attribute_missed_signal(record, Grade.SKIP_WAS_RIGHT, Decimal("-0.05")) is None


class TestAttributeFilterEvent:
    def test_spread_filter_maps_to_spread_cause(self):
        fe = FilterEvent("W", Side.YES, T0, "spread_filter", Decimal("0.20"))
        result = attribute_filter_event(fe, Grade.MISSED_BY_FILTER)
        assert result.cause == Cause.SPREAD
        assert result.dollar_impact == Decimal("0.20")

    def test_vol_spike_maps_to_volatility_cause(self):
        fe = FilterEvent("W", Side.YES, T0, "vol_spike", Decimal("0.10"))
        result = attribute_filter_event(fe, Grade.MISSED_BY_FILTER)
        assert result.cause == Cause.VOLATILITY

    def test_stale_data_maps_to_stale_data_cause(self):
        fe = FilterEvent("W", Side.YES, T0, "stale_data", Decimal("0.10"))
        result = attribute_filter_event(fe, Grade.MISSED_BY_FILTER)
        assert result.cause == Cause.STALE_DATA

    def test_unmapped_reason_falls_back_to_generic_filter_cause(self):
        fe = FilterEvent("W", Side.YES, T0, "cooldown", Decimal("0.10"))
        result = attribute_filter_event(fe, Grade.MISSED_BY_FILTER)
        assert result.cause == Cause.FILTER

    def test_filter_saved_us_is_not_attributed(self):
        fe = FilterEvent("W", Side.YES, T0, "spread_filter", Decimal("-0.20"))
        assert attribute_filter_event(fe, Grade.FILTER_SAVED_US) is None


class TestAttributeNoSignalOpportunity:
    def test_always_attributes_to_model_error(self):
        opp = Opportunity("W", Side.YES, T0, "no_signal", Decimal("0.30"), Decimal("0.40"))
        result = attribute_no_signal_opportunity(opp)
        assert result.cause == Cause.MODEL_ERROR
        assert result.dollar_impact == Decimal("0.30")
        assert result.window_ticker == "W"


class TestSummarizeAttribution:
    def test_sums_dollar_impact_per_cause(self):
        rtr = rt(pnl="-0.50", fees="0.10")
        results = [
            attribute_round_trip(
                rtr, GradeResult(primary=Grade.BAD_MODEL), [markout(30.0, -0.10)], learning_cfg()
            ),
            attribute_round_trip(
                rtr, GradeResult(primary=Grade.BAD_MODEL), [markout(30.0, -0.10)], learning_cfg()
            ),
            attribute_filter_event(
                FilterEvent("W", Side.YES, T0, "spread_filter", Decimal("0.20")),
                Grade.MISSED_BY_FILTER,
            ),
        ]
        totals = summarize_attribution(results)
        assert totals[Cause.MODEL_ERROR] == Decimal("1.00")
        assert totals[Cause.SPREAD] == Decimal("0.20")

    def test_empty_input_gives_empty_totals(self):
        assert summarize_attribution([]) == {}


class TestFormatAttributionReport:
    def test_matches_claude_md_example_shape(self):
        by_cause = {
            Cause.USER_DELAY: Decimal("2.60"),
            Cause.FEES: Decimal("1.10"),
            Cause.MODEL_ERROR: Decimal("0.50"),
        }
        report = format_attribution_report(Decimal("-4.20"), by_cause)
        assert report == (
            "Of -$4.20 today, $2.60 was user delay, $1.10 was fees, $0.50 was model error."
        )

    def test_orders_causes_by_descending_dollar_impact(self):
        by_cause = {Cause.FEES: Decimal("0.10"), Cause.MODEL_ERROR: Decimal("5.00")}
        report = format_attribution_report(Decimal("-5.10"), by_cause)
        assert report.index("model error") < report.index("fees")

    def test_zero_impact_causes_are_omitted(self):
        by_cause = {Cause.FEES: Decimal("0"), Cause.MODEL_ERROR: Decimal("1.00")}
        report = format_attribution_report(Decimal("-1.00"), by_cause)
        assert "fees" not in report

    def test_no_attributable_causes(self):
        report = format_attribution_report(Decimal("0.00"), {})
        assert report == "Net P&L: $0.00. No attributable losses or misses."

    def test_positive_total_formats_without_a_minus_sign(self):
        report = format_attribution_report(Decimal("3.00"), {Cause.FEES: Decimal("1.00")})
        assert report.startswith("Of $3.00 today")
