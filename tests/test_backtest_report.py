from datetime import UTC, datetime, timedelta
from decimal import Decimal

from gold_edge.backtest.replay import (
    BacktestResult,
    RoundTripRecord,
    SignalRecord,
    WindowReplayResult,
)
from gold_edge.backtest.report import build_summary, calibration_table, format_sweep_report
from gold_edge.engine.signals import build_signal
from gold_edge.models import Action, Side
from tests.test_state_machine import book

T0 = datetime(2026, 1, 1, tzinfo=UTC)


def rt(
    pnl: str, fees: str = "0.10", entry_time=T0, exit_time=None, reason="sold"
) -> RoundTripRecord:
    return RoundTripRecord(
        window_ticker="KXGOLD15M-A",
        side="YES",
        size=Decimal(1),
        entry_price=Decimal("0.50"),
        entry_time=entry_time,
        exit_price=Decimal("0.60"),
        exit_time=exit_time or (entry_time + timedelta(seconds=30)),
        exit_reason=reason,
        pnl=Decimal(pnl),
        fees_paid=Decimal(fees),
    )


def entry_signal(fair: float, side: Side = Side.YES, now=T0) -> SignalRecord:
    sig = build_signal(
        action=Action.BUY,
        side=side,
        window_ticker="KXGOLD15M-A",
        book=book(now=now),
        fair=fair,
        size=Decimal(1),
        edge_after_costs=0.04,
        reason="enter_edge",
        now=now,
        ttl_s=6.0,
    )
    return SignalRecord(signal=sig, filled=True, fill=None)


class TestBuildSummary:
    def test_empty_result_has_zero_pnl_and_no_crash(self):
        result = BacktestResult(windows=[])
        summary = build_summary(result)
        assert summary.net_pnl == Decimal("0")
        assert summary.round_trip_count == 0
        assert summary.max_drawdown == Decimal("0")

    def test_totals_gross_and_net_pnl_and_fees(self):
        wr = WindowReplayResult(
            ticker="KXGOLD15M-A",
            signal_records=[],
            round_trips=[rt("1.00", "0.10"), rt("-0.50", "0.08")],
            settlement_result="yes",
            ending_realized_pnl_today=Decimal("0.50"),
        )
        result = BacktestResult(windows=[wr])
        summary = build_summary(result)
        assert summary.net_pnl == Decimal("0.50")
        assert summary.total_fees == Decimal("0.18")
        assert summary.gross_pnl == Decimal("0.68")
        assert summary.round_trip_count == 2

    def test_max_drawdown_tracks_worst_peak_to_trough(self):
        wr = WindowReplayResult(
            ticker="KXGOLD15M-A",
            signal_records=[],
            round_trips=[
                rt("2.00", entry_time=T0, exit_time=T0 + timedelta(seconds=10)),
                rt(
                    "-3.00",
                    entry_time=T0 + timedelta(seconds=20),
                    exit_time=T0 + timedelta(seconds=30),
                ),
                rt(
                    "1.00",
                    entry_time=T0 + timedelta(seconds=40),
                    exit_time=T0 + timedelta(seconds=50),
                ),
            ],
            settlement_result="yes",
            ending_realized_pnl_today=Decimal("0"),
        )
        result = BacktestResult(windows=[wr])
        summary = build_summary(result)
        # equity: 2, -1, 0 -> peak 2, trough -1 -> drawdown 3
        assert summary.max_drawdown == Decimal("3.00")

    def test_signals_filled_and_missed_counted(self):
        filled_sig = entry_signal(0.6)
        missed_sig = SignalRecord(signal=filled_sig.signal, filled=False, fill=None)
        wr = WindowReplayResult(
            ticker="KXGOLD15M-A",
            signal_records=[filled_sig, missed_sig],
            round_trips=[],
            settlement_result="yes",
            ending_realized_pnl_today=Decimal("0"),
        )
        result = BacktestResult(windows=[wr])
        summary = build_summary(result)
        assert summary.signals_issued == 2
        assert summary.signals_filled == 1
        assert summary.signals_missed == 1


class TestCalibrationTable:
    def test_buckets_by_fair_value_and_actual_win_rate(self):
        # Two YES signals with fair=0.85 in a window that settled "yes" (won).
        sig_win_1 = entry_signal(0.85)
        sig_win_2 = entry_signal(0.87)
        wr = WindowReplayResult(
            ticker="KXGOLD15M-A",
            signal_records=[sig_win_1, sig_win_2],
            round_trips=[],
            settlement_result="yes",
            ending_realized_pnl_today=Decimal("0"),
        )
        result = BacktestResult(windows=[wr])
        table = calibration_table(result, bucket_width=0.1)
        assert len(table) == 1
        bucket = table[0]
        assert bucket["n"] == 2
        assert bucket["actual_win_rate"] == 1.0
        assert 0.85 <= bucket["mean_predicted"] <= 0.87

    def test_no_signals_returns_empty_table(self):
        result = BacktestResult(windows=[])
        assert calibration_table(result) == []

    def test_skips_signals_from_windows_with_unknown_settlement(self):
        sig = entry_signal(0.85)
        wr = WindowReplayResult(
            ticker="KXGOLD15M-A",
            signal_records=[sig],
            round_trips=[],
            settlement_result=None,
            ending_realized_pnl_today=Decimal("0"),
        )
        result = BacktestResult(windows=[wr])
        assert calibration_table(result) == []


class TestFormatSweepReport:
    def test_includes_params_and_both_train_and_test_metrics(self):
        candidates = [
            {
                "params": {"enter_edge": 0.03},
                "train_net_pnl": Decimal("5.00"),
                "train_round_trips": 10,
                "test_net_pnl": Decimal("-1.00"),
                "test_round_trips": 4,
                "test_max_drawdown": Decimal("2.00"),
            }
        ]
        text = format_sweep_report(candidates)
        assert "enter_edge=0.03" in text
        assert "train: net_pnl=$5.00" in text
        assert "test:  net_pnl=$-1.00" in text

    def test_empty_candidates_does_not_crash(self):
        assert "sweep" in format_sweep_report([]).lower()
