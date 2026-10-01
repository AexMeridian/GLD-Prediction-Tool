import asyncio
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from gold_edge.learning.drift import DriftBaseline
from gold_edge.learning.drift_metrics import (
    baseline_from_evidence,
    baseline_to_evidence,
    compute_live_drift_metrics,
    snapshot_drift_baseline,
)
from gold_edge.learning.markouts import Markout
from gold_edge.models import Action, Side, Signal, SignalStatus
from gold_edge.recorder import Recorder

T0 = datetime(2026, 1, 1, tzinfo=UTC)
MIN_ROUND_TRIPS = 30


def _populate(sqlite_path, n: int, start: datetime, prefix: str = "") -> None:
    """n graded round trips, one per minute starting at `start`: signal side
    alternates YES/NO, settlement always matches the signal's side (so
    calibration_error == 0 for a `fair` of exactly the outcome -- here fair
    is deliberately imperfect at 0.6/0.6 so brier_score is a known nonzero
    constant), 3-in-4 markouts positive (sign rate 0.75), constant net_pnl,
    and a known delay_s distribution. `prefix` keeps ids/tickers unique
    across multiple calls against the same sqlite file."""
    recorder = Recorder(sqlite_path)
    for i in range(n):
        at = start + timedelta(minutes=i)
        side = Side.YES if i % 2 == 0 else Side.NO
        ticker = f"{prefix}W{i}"
        signal = Signal(
            id=f"{prefix}s{i}",
            window_ticker=ticker,
            action=Action.BUY,
            side=side,
            limit_price=Decimal("0.60"),
            size=Decimal("1"),
            fair=0.60,
            market_price=Decimal("0.55"),
            edge_after_costs=0.05,
            reason="test",
            created_at=at,
            expires_at=at + timedelta(seconds=6),
            status=SignalStatus.FILLED,
        )
        asyncio.run(recorder.record_signal(signal))
        # settlement matches the signal's side -> outcome=1.0 for every row,
        # so brier_score([0.6]*n, [1.0]*n) == (1-0.6)^2 == 0.16 exactly.
        asyncio.run(recorder.record_settlement(ticker, {}, side.value.lower(), at))
        asyncio.run(
            recorder.record_grade(
                entry_signal_id=signal.id,
                exit_signal_id=None,
                window_ticker=ticker,
                primary_grade="GOOD_CALL",
                tags=[],
                net_pnl=Decimal("0.05"),
                graded_at=at,
            )
        )
        edge_markout = 0.02 if i % 4 != 3 else -0.02  # 3/4 positive -> sign rate 0.75
        asyncio.run(
            recorder.record_markout(
                signal.id,
                Markout(
                    horizon_s=60.0, at=at, pyth_price=100.0, market_mid=0.6,
                    market_bid=0.59, market_ask=0.61, fair=0.6, edge_markout=edge_markout,
                ),
            )
        )
        delay = 1.0 if i % 2 == 0 else 3.0  # median 2.0
        asyncio.run(
            recorder.record_fill_latency(signal.id, "BUY", "entry", delay, was_missed=False)
        )
    recorder.close()


def test_returns_none_below_min_round_trips(tmp_path):
    sqlite_path = tmp_path / "t.sqlite"
    _populate(sqlite_path, MIN_ROUND_TRIPS - 1, T0)
    assert snapshot_drift_baseline(sqlite_path, as_of=T0 + timedelta(days=1)) is None


def test_snapshot_drift_baseline_aggregates_known_values(tmp_path):
    sqlite_path = tmp_path / "t.sqlite"
    _populate(sqlite_path, 40, T0)  # divisible by 4 -> exact 3-in-4 sign rate
    baseline = snapshot_drift_baseline(sqlite_path, as_of=T0 + timedelta(days=1))
    assert baseline is not None
    assert abs(baseline.calibration_error - 0.16) < 1e-9
    assert abs(baseline.markout_sign_rate - 0.75) < 1e-9
    assert baseline.pnl_per_round_trip == Decimal("0.05")
    assert abs(baseline.median_delay_s - 2.0) < 1e-9


def test_compute_live_drift_metrics_respects_the_date_range(tmp_path):
    sqlite_path = tmp_path / "t.sqlite"
    # 30 round trips before the window, 30 inside it -- only the live window
    # should be counted by compute_live_drift_metrics.
    _populate(sqlite_path, MIN_ROUND_TRIPS, T0, prefix="base")
    live_start = T0 + timedelta(days=10)
    _populate(sqlite_path, MIN_ROUND_TRIPS, live_start, prefix="live")
    since = live_start - timedelta(minutes=1)
    now = live_start + timedelta(days=1)
    live = compute_live_drift_metrics(sqlite_path, since, now)
    assert live is not None
    assert live.n_round_trips == MIN_ROUND_TRIPS


def test_baseline_evidence_round_trip():
    baseline = DriftBaseline(
        calibration_error=0.16, markout_sign_rate=0.75,
        pnl_per_round_trip=Decimal("0.05"), median_delay_s=2.0,
    )
    evidence = {"rationale": "unrelated", **baseline_to_evidence(baseline)}
    restored = baseline_from_evidence(evidence)
    assert restored == baseline


def test_baseline_from_evidence_missing_keys_is_none():
    assert baseline_from_evidence({"rationale": "no drift keys here"}) is None
