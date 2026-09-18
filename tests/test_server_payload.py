import json
from datetime import UTC, datetime
from decimal import Decimal

from gold_edge.config import FeesConfig
from gold_edge.engine.state_machine import EngineState
from gold_edge.learning.actions import PromoteOutcome
from gold_edge.learning.registry import GateResult
from gold_edge.model.fair_value import FairValue
from gold_edge.models import Position, PositionState, Side
from gold_edge.recorder import Recorder
from gold_edge.server import (
    _build_state_payload,
    _learning_payload,
    _position_to_json,
    _promote_outcome_payload,
    _review_payload,
    _signal_to_json,
    _version_status_payload,
    settle_position_pnl,
)
from tests.test_state_machine import book, window

T0 = datetime(2026, 1, 1, tzinfo=UTC)


def fees_cfg(**overrides) -> FeesConfig:
    defaults = dict(base_rate=0.07, fee_multiplier=1.0, maker_fees_enabled=True)
    defaults.update(overrides)
    return FeesConfig(**defaults)


def test_position_to_json_computes_unrealized_pnl():
    position = Position(
        window_ticker="KXGOLD15M-TEST",
        side=Side.YES,
        size=Decimal(10),
        entry_price=Decimal("0.50"),
        entered_at=T0,
        state=PositionState.LONG_YES,
    )
    bk = book(yes_bid="0.60", yes_ask="0.62")
    payload = _position_to_json(position, bk, fees_cfg())
    assert payload["side"] == "YES"
    assert payload["entry_price"] == 0.50
    # exit_value = 0.60 - fee(0.60,10 contracts); unrealized = (exit_value-entry)*size
    assert payload["unrealized_pnl"] > 0
    json.dumps(payload)  # must not raise


def test_position_to_json_none_when_no_position():
    assert _position_to_json(None, book(), fees_cfg()) is None


def test_signal_to_json_is_json_safe():
    from gold_edge.engine.signals import build_signal
    from gold_edge.models import Action

    signal = build_signal(
        action=Action.BUY,
        side=Side.YES,
        window_ticker="KXGOLD15M-TEST",
        book=book(),
        fair=0.55,
        size=Decimal(1),
        edge_after_costs=0.04,
        reason="enter_edge",
        now=T0,
        ttl_s=6.0,
    )
    payload = _signal_to_json(signal)
    assert payload["action"] == "BUY"
    assert payload["side"] == "YES"
    assert isinstance(payload["limit_price"], float)
    json.dumps(payload)


def test_build_state_payload_is_json_serializable_with_no_data_yet():
    payload = _build_state_payload(
        window=None,
        latest_price=None,
        book=None,
        fair=None,
        engine_state=EngineState(),
        fees_cfg=fees_cfg(),
        pyth_age_s=None,
        kalshi_age_s=None,
        stale_s=3.0,
        kill_switch=False,
        session_log_tail=[],
    )
    assert payload["window"] is None
    assert payload["position"] is None
    json.dumps(payload)


def test_build_state_payload_full():
    win = window(close_in_s=300.0)
    bk = book(yes_bid="0.55", yes_ask="0.57")
    fair = FairValue(yes=0.60, no=0.40)
    position = Position(
        window_ticker=win.ticker,
        side=Side.YES,
        size=Decimal(1),
        entry_price=Decimal("0.50"),
        entered_at=T0,
        state=PositionState.LONG_YES,
    )
    engine_state = EngineState(position_state=PositionState.LONG_YES, position=position)
    payload = _build_state_payload(
        window=win,
        latest_price=2005.0,
        book=bk,
        fair=fair,
        engine_state=engine_state,
        fees_cfg=fees_cfg(),
        pyth_age_s=0.4,
        kalshi_age_s=0.6,
        stale_s=3.0,
        kill_switch=True,
        session_log_tail=[{"kind": "note", "at": T0.isoformat(), "payload": {"note": "x"}}],
    )
    assert payload["window"]["ticker"] == win.ticker
    assert payload["market"]["s"] == 2005.0
    assert payload["fair"]["yes"] == 0.60
    assert payload["position"]["side"] == "YES"
    assert payload["feed_status"]["pyth_stale"] is False
    assert payload["session"]["kill_switch"] is True
    assert len(payload["session_log"]) == 1
    json.dumps(payload)


def test_build_state_payload_marks_stale_feeds():
    payload = _build_state_payload(
        window=window(),
        latest_price=2000.0,
        book=book(),
        fair=FairValue(yes=0.5, no=0.5),
        engine_state=EngineState(),
        fees_cfg=fees_cfg(),
        pyth_age_s=10.0,
        kalshi_age_s=0.1,
        stale_s=3.0,
        kill_switch=False,
        session_log_tail=[],
    )
    assert payload["feed_status"]["pyth_stale"] is True
    assert payload["feed_status"]["kalshi_stale"] is False


def test_pending_signal_is_included_in_payload():
    from gold_edge.engine.signals import build_signal
    from gold_edge.models import Action

    win = window()
    bk = book()
    signal = build_signal(
        action=Action.BUY,
        side=Side.YES,
        window_ticker=win.ticker,
        book=bk,
        fair=0.55,
        size=Decimal(1),
        edge_after_costs=0.04,
        reason="enter_edge",
        now=T0,
        ttl_s=6.0,
    )
    engine_state = EngineState(pending_signal=signal)
    payload = _build_state_payload(
        window=win,
        latest_price=2000.0,
        book=bk,
        fair=FairValue(yes=0.55, no=0.45),
        engine_state=engine_state,
        fees_cfg=fees_cfg(),
        pyth_age_s=0.1,
        kalshi_age_s=0.1,
        stale_s=3.0,
        kill_switch=False,
        session_log_tail=[],
    )
    assert payload["signal"] is not None
    assert payload["signal"]["id"] == signal.id


def _position(side: Side, size: str, entry_price: str) -> Position:
    return Position(
        window_ticker="KXGOLD15M-TEST",
        side=side,
        size=Decimal(size),
        entry_price=Decimal(entry_price),
        entered_at=T0,
        state=PositionState.LONG_YES if side is Side.YES else PositionState.LONG_NO,
    )


def test_settle_position_pnl_yes_wins():
    position = _position(Side.YES, "10", "0.50")
    # entry_fee = ceil(0.07*10*0.5*0.5) = ceil(0.175) = 0.18; payoff = 5.00
    pnl = settle_position_pnl(position, "yes", fees_cfg())
    assert pnl == Decimal("4.82")


def test_settle_position_pnl_yes_loses():
    position = _position(Side.YES, "10", "0.50")
    pnl = settle_position_pnl(position, "no", fees_cfg())
    assert pnl == Decimal("-5.18")


def test_settle_position_pnl_no_wins():
    position = _position(Side.NO, "5", "0.30")
    # entry_fee = ceil(0.07*5*0.3*0.7) = 0.08; payoff = (1-0.3)*5 = 3.50
    pnl = settle_position_pnl(position, "no", fees_cfg())
    assert pnl == Decimal("3.42")


def test_settle_position_pnl_no_loses():
    position = _position(Side.NO, "5", "0.30")
    pnl = settle_position_pnl(position, "yes", fees_cfg())
    assert pnl == Decimal("-1.58")


class TestReviewPayload:
    def test_no_sqlite_file_yet(self, tmp_path):
        payload = _review_payload(tmp_path / "missing.sqlite", None, None)
        assert "No recorded data yet." in payload["report"]
        json.dumps(payload)

    def test_empty_but_valid_database(self, tmp_path):
        sqlite_path = tmp_path / "test.sqlite"
        Recorder(sqlite_path).close()
        payload = _review_payload(sqlite_path, None, None)
        assert "learn" in payload["report"]

    def test_reports_a_recorded_grade(self, tmp_path):
        import asyncio

        sqlite_path = tmp_path / "test.sqlite"
        recorder = Recorder(sqlite_path)
        asyncio.run(
            recorder.record_grade(
                entry_signal_id="e1",
                exit_signal_id="x1",
                window_ticker="KXGOLD15M-TEST",
                primary_grade="GOOD_CALL",
                tags=[],
                net_pnl=Decimal("0.50"),
                graded_at=T0,
            )
        )
        recorder.close()
        payload = _review_payload(sqlite_path, None, None)
        assert "GOOD_CALL" in payload["report"]
        json.dumps(payload)


class TestLearningPayload:
    def test_no_sqlite_file_yet(self, tmp_path):
        payload = _learning_payload(tmp_path / "missing.sqlite")
        assert payload == {"patterns": [], "proposals": [], "versions": [], "drift": None}

    def test_reports_a_proposal(self, tmp_path):
        import asyncio

        sqlite_path = tmp_path / "test.sqlite"
        recorder = Recorder(sqlite_path)
        asyncio.run(
            recorder.record_proposal(
                proposal_id="p1",
                param_changes={"enter_edge": 0.04},
                n_holdout_trades=250,
                holdout_pnl_delta=Decimal("5.00"),
                holdout_pnl_delta_ci_low=Decimal("1.00"),
                holdout_pnl_delta_ci_high=Decimal("9.00"),
                holdout_drawdown_delta=Decimal("0.05"),
                rationale="test",
                status="pending",
                created_at=T0,
            )
        )
        recorder.close()
        payload = _learning_payload(sqlite_path)
        assert len(payload["proposals"]) == 1
        assert payload["proposals"][0]["id"] == "p1"
        assert payload["proposals"][0]["status"] == "pending"
        json.dumps(payload)


class TestVersionStatusPayload:
    def test_no_sqlite_file_yet(self, tmp_path):
        payload = _version_status_payload(tmp_path / "missing.sqlite")
        assert payload == {"current_version": None, "drift": None}

    def test_no_promoted_version_yet(self, tmp_path):
        sqlite_path = tmp_path / "test.sqlite"
        Recorder(sqlite_path).close()
        payload = _version_status_payload(sqlite_path)
        assert payload["current_version"] is None
        assert payload["drift"] is None

    def test_reports_current_promoted_version_and_drift(self, tmp_path):
        import asyncio

        sqlite_path = tmp_path / "test.sqlite"
        recorder = Recorder(sqlite_path)
        asyncio.run(
            recorder.record_config_version(
                version_hash="abc123",
                param_changes={"enter_edge": 0.04},
                evidence={},
                promoted=True,
                created_at=T0,
            )
        )
        asyncio.run(
            recorder.record_drift_event(
                degraded_metrics=["calibration error 0.3 vs 0.1"],
                suggestion="Consider rolling back.",
                detected_at=T0,
            )
        )
        recorder.close()
        payload = _version_status_payload(sqlite_path)
        assert payload["current_version"]["version_hash"] == "abc123"
        assert payload["drift"]["has_drift"] is True
        json.dumps(payload)


class TestPromoteOutcomePayload:
    def test_passed_with_promoted_params(self):
        outcome = PromoteOutcome(
            found=True,
            gate_result=GateResult(passed=True, reasons=[]),
            promoted_version=None,
            shadow_sessions_seen=10,
            shadow_beats_live=True,
        )
        payload = _promote_outcome_payload(outcome)
        assert payload["passed"] is True
        assert payload["promoted_params"] is None
        json.dumps(payload)

    def test_rejected_reports_reasons(self):
        outcome = PromoteOutcome(
            found=True,
            gate_result=GateResult(passed=False, reasons=["did not beat live config"]),
            promoted_version=None,
            shadow_sessions_seen=2,
            shadow_beats_live=False,
        )
        payload = _promote_outcome_payload(outcome)
        assert payload["passed"] is False
        assert payload["reasons"] == ["did not beat live config"]
