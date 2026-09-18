from datetime import UTC, datetime
from decimal import Decimal

from gold_edge.learning.attribution import AttributionResult, Cause
from gold_edge.learning.delay_profile import DelayProfile
from gold_edge.learning.drift import DriftEvent
from gold_edge.learning.grader import Grade
from gold_edge.learning.insights import GradedTrade, session_report_card, weekly_report
from gold_edge.learning.opportunities import FilterStats, Opportunity
from gold_edge.learning.patterns import BucketStat
from gold_edge.learning.proposer import Proposal
from gold_edge.models import Side

T0 = datetime(2026, 1, 1, tzinfo=UTC)


class TestSessionReportCard:
    def test_handles_no_trades(self):
        report = session_report_card([], [], [], [])
        assert "Trades: 0" in report
        assert "No graded trades this session." in report

    def test_reports_net_pnl_and_fees(self):
        trades = [
            GradedTrade("W1", Grade.GOOD_CALL, Decimal("1.00"), Decimal("0.10")),
            GradedTrade("W2", Grade.BAD_MODEL, Decimal("-0.50"), Decimal("0.05")),
        ]
        report = session_report_card(trades, [], [], [])
        assert "Trades: 2" in report
        assert "Net P&L: $0.50" in report
        assert "Fees paid: $0.15" in report

    def test_grade_breakdown_counts(self):
        trades = [
            GradedTrade("W1", Grade.GOOD_CALL, Decimal("1.00"), Decimal("0.10")),
            GradedTrade("W2", Grade.GOOD_CALL, Decimal("1.00"), Decimal("0.10")),
            GradedTrade("W3", Grade.BAD_MODEL, Decimal("-0.50"), Decimal("0.05")),
        ]
        report = session_report_card(trades, [], [], [])
        assert "GOOD_CALL: 2" in report
        assert "BAD_MODEL: 1" in report

    def test_best_and_worst_call_identified(self):
        trades = [
            GradedTrade("BEST", Grade.GOOD_CALL, Decimal("2.00"), Decimal("0.10")),
            GradedTrade("WORST", Grade.BAD_MODEL, Decimal("-3.00"), Decimal("0.05")),
        ]
        report = session_report_card(trades, [], [], [])
        assert "Best call: BEST" in report
        assert "Worst call: WORST" in report

    def test_includes_attribution_rollup(self):
        trades = [GradedTrade("W1", Grade.BAD_MODEL, Decimal("-0.50"), Decimal("0.05"))]
        attribution = [AttributionResult("W1", Cause.MODEL_ERROR, Decimal("0.50"), "x")]
        report = session_report_card(trades, attribution, [], [])
        assert "was model error" in report

    def test_includes_top_opportunities_sorted_descending(self):
        opps = [
            Opportunity("W1", Side.YES, T0, "below_enter_edge", Decimal("0.10"), Decimal("0.20")),
            Opportunity("W2", Side.NO, T0, "spread_filter", Decimal("0.50"), Decimal("0.60")),
        ]
        report = session_report_card([], [], opps, [], top_n_opportunities=2)
        assert report.index("W2") < report.index("W1")

    def test_includes_filter_scorecard(self):
        stats = [FilterStats("spread_filter", 10, Decimal("2.00"), Decimal("0.50"))]
        report = session_report_card([], [], [], stats)
        assert "spread_filter (n=10)" in report
        assert "saved $2.00" in report
        assert "net $1.50" in report


class TestWeeklyReport:
    def test_no_significant_patterns(self):
        report = weekly_report([], DelayProfile(), [], None, min_bucket_n=30)
        assert "None found this week." in report

    def test_reports_significant_pattern_with_ci_and_n(self):
        stats = [
            BucketStat(
                "side", "YES", 40, Decimal("1.0"), Decimal("0.5"), Decimal("1.5"), 0.001, True
            )
        ]
        report = weekly_report(stats, DelayProfile(), [], None, min_bucket_n=30)
        assert "side=YES (n=40)" in report
        assert "95% CI" in report

    def test_notes_insufficient_significance_count(self):
        stats = [
            BucketStat(
                "side", "YES", 40, Decimal("1.0"), Decimal("0.5"), Decimal("1.5"), 0.001, True
            ),
            BucketStat(
                "side", "NO", 40, Decimal("0.0"), Decimal("-1.0"), Decimal("1.0"), 0.80, False
            ),
        ]
        report = weekly_report(stats, DelayProfile(), [], None, min_bucket_n=30)
        assert "1 other bucket(s)" in report

    def test_delay_profile_not_learned(self):
        report = weekly_report([], DelayProfile(n=5), [], None, min_bucket_n=30)
        assert "Not yet learned" in report

    def test_delay_profile_learned(self):
        profile = DelayProfile(samples=[1.0, 1.5, 2.0] * 20, n=60)
        report = weekly_report([], profile, [], None, min_bucket_n=30)
        assert "Learned from 60 fills" in report

    def test_pending_proposals_listed(self):
        proposal = Proposal(
            id="p1",
            param_changes={"enter_edge": 0.04},
            n_holdout_trades=250,
            holdout_pnl_delta=Decimal("5.00"),
            holdout_pnl_delta_ci_low=Decimal("1.00"),
            holdout_pnl_delta_ci_high=Decimal("9.00"),
            holdout_drawdown_delta=Decimal("0.05"),
            rationale="Raise enter_edge for more selective entries.",
            created_at=T0,
        )
        report = weekly_report([], DelayProfile(), [proposal], None, min_bucket_n=30)
        assert "Pending proposals: 1" in report
        assert "Raise enter_edge" in report

    def test_no_drift_check_yet(self):
        report = weekly_report([], DelayProfile(), [], None, min_bucket_n=30)
        assert "No drift check has run yet." in report

    def test_no_drift_detected(self):
        event = DriftEvent(detected_at=T0, degraded_metrics=[], suggestion="No action needed.")
        report = weekly_report([], DelayProfile(), [], event, min_bucket_n=30)
        assert "No drift detected." in report

    def test_drift_detected_lists_degraded_metrics(self):
        event = DriftEvent(
            detected_at=T0,
            degraded_metrics=["calibration error 0.3 vs baseline 0.1"],
            suggestion="Consider rolling back.",
        )
        report = weekly_report([], DelayProfile(), [], event, min_bucket_n=30)
        assert "DEGRADED: calibration error" in report
        assert "Consider rolling back." in report
