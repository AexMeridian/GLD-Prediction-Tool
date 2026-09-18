import asyncio
import sqlite3
from datetime import UTC, datetime
from decimal import Decimal

from gold_edge.learning.markouts import Markout
from gold_edge.recorder import Recorder

T0 = datetime(2026, 1, 1, tzinfo=UTC)


def test_record_markout_round_trips_through_sqlite(tmp_path):
    recorder = Recorder(tmp_path / "test.sqlite")
    m = Markout(
        horizon_s=5.0,
        at=T0,
        pyth_price=2000.0,
        market_mid=0.50,
        market_bid=0.49,
        market_ask=0.51,
        fair=0.52,
        edge_markout=0.02,
    )
    asyncio.run(recorder.record_markout("sig-1", m))
    recorder.close()

    conn = sqlite3.connect(tmp_path / "test.sqlite")
    row = conn.execute("SELECT * FROM markouts WHERE signal_id = ?", ("sig-1",)).fetchone()
    conn.close()
    assert row is not None
    assert row[1] == "sig-1"
    assert row[2] == 5.0
    assert row[9] == 0.02  # edge_markout


def test_record_grade_round_trips_through_sqlite(tmp_path):
    sqlite_path = tmp_path / "test.sqlite"
    recorder = Recorder(sqlite_path)
    asyncio.run(
        recorder.record_grade(
            entry_signal_id="entry-1",
            exit_signal_id="exit-1",
            window_ticker="KXGOLD15M-TEST",
            primary_grade="GOOD_CALL",
            tags=["EXIT_TOO_EARLY"],
            net_pnl=Decimal("0.35"),
            graded_at=T0,
        )
    )
    recorder.close()

    conn = sqlite3.connect(sqlite_path)
    row = conn.execute(
        "SELECT entry_signal_id, exit_signal_id, window_ticker, primary_grade, tags, "
        "net_pnl FROM grades"
    ).fetchone()
    conn.close()
    assert row == ("entry-1", "exit-1", "KXGOLD15M-TEST", "GOOD_CALL", '["EXIT_TOO_EARLY"]', "0.35")


def test_record_attribution_round_trips_through_sqlite(tmp_path):
    recorder = Recorder(tmp_path / "test.sqlite")
    asyncio.run(
        recorder.record_attribution(
            window_ticker="KXGOLD15M-TEST",
            cause="model_error",
            dollar_impact=Decimal("0.50"),
            detail="market moved against the call -- fair value was wrong",
            attributed_at=T0,
        )
    )
    recorder.close()

    conn = sqlite3.connect(tmp_path / "test.sqlite")
    row = conn.execute(
        "SELECT window_ticker, cause, dollar_impact, detail, attributed_at FROM attribution"
    ).fetchone()
    conn.close()
    assert row[0] == "KXGOLD15M-TEST"
    assert row[1] == "model_error"
    assert row[2] == "0.50"
    assert row[4] == T0.isoformat()


def test_record_opportunity_round_trips_through_sqlite(tmp_path):
    recorder = Recorder(tmp_path / "test.sqlite")
    asyncio.run(
        recorder.record_opportunity(
            window_ticker="KXGOLD15M-TEST",
            side="YES",
            at=T0,
            reason="below_enter_edge",
            realistic_net_pnl=Decimal("0.10"),
            oracle_net_pnl=Decimal("0.20"),
        )
    )
    recorder.close()
    conn = sqlite3.connect(tmp_path / "test.sqlite")
    row = conn.execute(
        "SELECT window_ticker, side, reason, realistic_net_pnl, oracle_net_pnl FROM opportunities"
    ).fetchone()
    conn.close()
    assert row == ("KXGOLD15M-TEST", "YES", "below_enter_edge", "0.10", "0.20")


def test_record_filter_event_round_trips_through_sqlite(tmp_path):
    recorder = Recorder(tmp_path / "test.sqlite")
    asyncio.run(
        recorder.record_filter_event(
            window_ticker="KXGOLD15M-TEST",
            side="NO",
            at=T0,
            reason="spread_filter",
            realistic_net_pnl=Decimal("-0.05"),
        )
    )
    recorder.close()
    conn = sqlite3.connect(tmp_path / "test.sqlite")
    row = conn.execute(
        "SELECT window_ticker, side, reason, realistic_net_pnl FROM filter_events"
    ).fetchone()
    conn.close()
    assert row == ("KXGOLD15M-TEST", "NO", "spread_filter", "-0.05")


def test_record_pattern_stat_round_trips_through_sqlite(tmp_path):
    recorder = Recorder(tmp_path / "test.sqlite")
    asyncio.run(
        recorder.record_pattern_stat(
            dimension="side",
            bucket="YES",
            n=40,
            mean_pnl=Decimal("1.00"),
            ci_low=Decimal("0.50"),
            ci_high=Decimal("1.50"),
            p_value=0.001,
            significant=True,
            computed_at=T0,
        )
    )
    recorder.close()
    conn = sqlite3.connect(tmp_path / "test.sqlite")
    row = conn.execute(
        "SELECT dimension, bucket, n, mean_pnl, significant FROM pattern_stats"
    ).fetchone()
    conn.close()
    assert row == ("side", "YES", 40, "1.00", 1)


def test_record_model_version_upserts_on_conflict(tmp_path):
    recorder = Recorder(tmp_path / "test.sqlite")
    asyncio.run(
        recorder.record_model_version(
            version_hash="abc123",
            kind="calibrator",
            evidence={"brier": 0.1},
            promoted=False,
            created_at=T0,
        )
    )
    asyncio.run(
        recorder.record_model_version(
            version_hash="abc123",
            kind="calibrator",
            evidence={"brier": 0.1},
            promoted=True,
            created_at=T0,
        )
    )
    recorder.close()
    conn = sqlite3.connect(tmp_path / "test.sqlite")
    rows = conn.execute("SELECT version_hash, promoted FROM model_versions").fetchall()
    conn.close()
    assert rows == [("abc123", 1)]


def test_record_config_version_round_trips_through_sqlite(tmp_path):
    recorder = Recorder(tmp_path / "test.sqlite")
    asyncio.run(
        recorder.record_config_version(
            version_hash="cfg123",
            param_changes={"enter_edge": 0.04},
            evidence={"n": 250},
            promoted=True,
            created_at=T0,
        )
    )
    recorder.close()
    conn = sqlite3.connect(tmp_path / "test.sqlite")
    row = conn.execute(
        "SELECT version_hash, param_changes_json, promoted FROM config_versions"
    ).fetchone()
    conn.close()
    assert row == ("cfg123", '{"enter_edge": 0.04}', 1)


def test_record_and_update_proposal(tmp_path):
    recorder = Recorder(tmp_path / "test.sqlite")
    asyncio.run(
        recorder.record_proposal(
            proposal_id="p1",
            param_changes={"enter_edge": 0.04},
            n_holdout_trades=250,
            holdout_pnl_delta=Decimal("5.00"),
            holdout_pnl_delta_ci_low=Decimal("1.00"),
            holdout_pnl_delta_ci_high=Decimal("9.00"),
            holdout_drawdown_delta=Decimal("0.05"),
            rationale="test rationale",
            status="pending",
            created_at=T0,
        )
    )
    asyncio.run(recorder.update_proposal_status("p1", "approved"))
    recorder.close()
    conn = sqlite3.connect(tmp_path / "test.sqlite")
    row = conn.execute("SELECT id, status, n_holdout_trades FROM proposals").fetchone()
    conn.close()
    assert row == ("p1", "approved", 250)


def test_record_shadow_signal_round_trips_through_sqlite(tmp_path):
    recorder = Recorder(tmp_path / "test.sqlite")
    asyncio.run(
        recorder.record_shadow_signal(
            candidate_param_changes={"entry_cutoff_s": 30.0},
            window_ticker="KXGOLD15M-TEST",
            live_net_pnl=Decimal("0"),
            candidate_net_pnl=Decimal("0.40"),
            recorded_at=T0,
        )
    )
    recorder.close()
    conn = sqlite3.connect(tmp_path / "test.sqlite")
    row = conn.execute(
        "SELECT window_ticker, live_net_pnl, candidate_net_pnl FROM shadow_signals"
    ).fetchone()
    conn.close()
    assert row == ("KXGOLD15M-TEST", "0", "0.40")


def test_record_drift_event_round_trips_through_sqlite(tmp_path):
    recorder = Recorder(tmp_path / "test.sqlite")
    asyncio.run(
        recorder.record_drift_event(
            degraded_metrics=["calibration error 0.3 vs 0.1"],
            suggestion="Consider rolling back.",
            detected_at=T0,
        )
    )
    recorder.close()
    conn = sqlite3.connect(tmp_path / "test.sqlite")
    row = conn.execute("SELECT degraded_metrics_json, suggestion FROM drift_events").fetchone()
    conn.close()
    assert row == ('["calibration error 0.3 vs 0.1"]', "Consider rolling back.")


def test_record_fill_latency_round_trips_through_sqlite(tmp_path):
    recorder = Recorder(tmp_path / "test.sqlite")
    asyncio.run(
        recorder.record_fill_latency(
            signal_id="sig-1", action="BUY", reason="enter_edge", delay_s=1.5, was_missed=False
        )
    )
    recorder.close()
    conn = sqlite3.connect(tmp_path / "test.sqlite")
    row = conn.execute(
        "SELECT signal_id, action, reason, delay_s, was_missed FROM user_fills_latency"
    ).fetchone()
    conn.close()
    assert row == ("sig-1", "BUY", "enter_edge", 1.5, 0)
