from datetime import UTC, datetime
from decimal import Decimal

from gold_edge.learning.drift import (
    DriftBaseline,
    DriftMetrics,
    DriftMonitor,
    check_drift,
)

T0 = datetime(2026, 1, 1, tzinfo=UTC)


def baseline(**overrides) -> DriftBaseline:
    defaults = dict(
        calibration_error=0.10,
        markout_sign_rate=0.60,
        pnl_per_round_trip=Decimal("0.20"),
        median_delay_s=1.5,
    )
    defaults.update(overrides)
    return DriftBaseline(**defaults)


def live(**overrides) -> DriftMetrics:
    defaults = dict(
        calibration_error=0.10,
        markout_sign_rate=0.60,
        pnl_per_round_trip=Decimal("0.20"),
        median_delay_s=1.5,
        n_round_trips=100,
    )
    defaults.update(overrides)
    return DriftMetrics(**defaults)


class TestCheckDrift:
    def test_no_drift_when_metrics_match_baseline(self):
        event = check_drift(baseline(), live(), now=T0)
        assert event.has_drift is False
        assert event.degraded_metrics == []
        assert event.suggestion == "No action needed."

    def test_detects_calibration_error_degradation(self):
        event = check_drift(baseline(calibration_error=0.10), live(calibration_error=0.20), now=T0)
        assert event.has_drift is True
        assert any("calibration error" in m for m in event.degraded_metrics)

    def test_no_drift_for_small_calibration_change(self):
        event = check_drift(baseline(calibration_error=0.10), live(calibration_error=0.12), now=T0)
        assert event.has_drift is False

    def test_detects_markout_sign_rate_drop(self):
        event = check_drift(baseline(markout_sign_rate=0.60), live(markout_sign_rate=0.30), now=T0)
        assert any("markout sign rate" in m for m in event.degraded_metrics)

    def test_detects_pnl_per_trip_degradation(self):
        event = check_drift(
            baseline(pnl_per_round_trip=Decimal("0.20")),
            live(pnl_per_round_trip=Decimal("0.05")),
            now=T0,
        )
        assert any("P&L per round trip" in m for m in event.degraded_metrics)

    def test_no_pnl_check_when_baseline_pnl_not_positive(self):
        event = check_drift(
            baseline(pnl_per_round_trip=Decimal("0")),
            live(pnl_per_round_trip=Decimal("-5.0")),
            now=T0,
        )
        assert not any("P&L per round trip" in m for m in event.degraded_metrics)

    def test_detects_delay_blowup(self):
        event = check_drift(baseline(median_delay_s=1.5), live(median_delay_s=5.0), now=T0)
        assert any("median fill delay" in m for m in event.degraded_metrics)

    def test_suggestion_present_when_drift_detected(self):
        event = check_drift(baseline(calibration_error=0.10), live(calibration_error=0.30), now=T0)
        assert "rolling back" in event.suggestion.lower()

    def test_reports_multiple_degraded_metrics_together(self):
        event = check_drift(
            baseline(),
            live(calibration_error=0.30, markout_sign_rate=0.20),
            now=T0,
        )
        assert len(event.degraded_metrics) == 2


class TestDriftMonitor:
    def test_current_status_none_when_empty(self):
        monitor = DriftMonitor()
        assert monitor.current_status is None

    def test_current_status_is_latest_event(self):
        monitor = DriftMonitor()
        monitor.record(check_drift(baseline(), live(), now=T0))
        drifted = check_drift(
            baseline(calibration_error=0.10), live(calibration_error=0.30), now=T0
        )
        monitor.record(drifted)
        assert monitor.current_status.has_drift is True

    def test_history_capped_at_max_history(self):
        monitor = DriftMonitor(max_history=3)
        for _ in range(5):
            monitor.record(check_drift(baseline(), live(), now=T0))
        assert len(monitor.events) == 3
