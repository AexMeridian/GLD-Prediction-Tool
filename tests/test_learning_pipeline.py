from datetime import UTC, datetime, timedelta
from decimal import Decimal

from gold_edge.backtest.replay import RecordedWindow
from gold_edge.learning.delay_profile import FillLatencySample
from gold_edge.learning.pipeline import run_learning_pipeline
from gold_edge.models import Tick, Window
from tests.test_backtest_replay import bt_cfg, model_cfg, vol_cfg
from tests.test_learning_grader import learning_cfg
from tests.test_state_machine import book, engine_cfg, fees_cfg

T0 = datetime(2026, 1, 1, tzinfo=UTC)


def strong_edge_window(ticker: str, day_offset: int = 0, length_s: float = 80.0) -> RecordedWindow:
    open_time = T0 + timedelta(days=day_offset)
    win = Window(
        ticker=ticker,
        event_ticker=f"{ticker}EVT",
        series_ticker="KXGOLD15M",
        open_time=open_time,
        close_time=open_time + timedelta(seconds=length_s),
        s0=Decimal("2000.00"),
        status="open",
    )
    n = int(length_s) + 1
    books = [
        book(
            yes_bid="0.83",
            yes_ask="0.85",
            no_bid="0.10",
            no_ask="0.12",
            now=open_time + timedelta(seconds=i),
        ).model_copy(update={"window_ticker": win.ticker})
        for i in range(n)
    ]
    return RecordedWindow(window=win, books=books, settlement_result="yes")


def strong_edge_ticks(day_offset: int = 0, n_seconds: int = 80) -> list[Tick]:
    open_time = T0 + timedelta(days=day_offset)
    return [
        Tick(
            symbol="Metal.XAU/USD",
            price=2020.0,
            conf=0.1,
            expo=-2,
            publish_time=open_time + timedelta(seconds=i),
            receive_time=open_time + timedelta(seconds=i),
        )
        for i in range(n_seconds + 1)
    ]


class TestRunLearningPipeline:
    def test_runs_end_to_end_without_crashing_on_a_small_dataset(self):
        windows = [strong_edge_window(f"KXGOLD15M-P{d}", day_offset=d) for d in range(3)]
        ticks = [t for d in range(3) for t in strong_edge_ticks(day_offset=d)]

        result = run_learning_pipeline(
            ticks=ticks,
            windows=windows,
            engine_cfg=engine_cfg(entry_cutoff_s=90.0),  # still never enters on this data
            fees_cfg=fees_cfg(),
            vol_cfg=vol_cfg(),
            model_cfg=model_cfg(),
            backtest_cfg=bt_cfg(human_delay_min_s=1.0, human_delay_max_s=1.0),
            learning_cfg=learning_cfg(),
            fill_latency_samples=[],
            proposer_param_grid={"entry_cutoff_s": [90.0, 30.0]},
            seed=0,
        )
        # Default entry_cutoff_s never lets the engine actually trade on this
        # crafted data, but the opportunity scanner should have found the
        # blocked mispricing every window (it's a real, cost-clearing edge).
        assert len(result.opportunities) >= 1
        assert len(result.graded_filter_events) >= 1
        assert result.proposals  # entry_cutoff_s=30 should surface as a proposal
        assert any(p.param_changes.get("entry_cutoff_s") == 30.0 for p in result.proposals)

    def test_produces_no_round_trips_when_engine_never_enters(self):
        windows = [strong_edge_window("KXGOLD15M-P0")]
        ticks = strong_edge_ticks()
        result = run_learning_pipeline(
            ticks=ticks,
            windows=windows,
            engine_cfg=engine_cfg(entry_cutoff_s=90.0),
            fees_cfg=fees_cfg(),
            vol_cfg=vol_cfg(),
            model_cfg=model_cfg(),
            backtest_cfg=bt_cfg(human_delay_min_s=1.0, human_delay_max_s=1.0),
            learning_cfg=learning_cfg(),
            fill_latency_samples=[],
            seed=0,
        )
        assert result.graded_round_trips == []
        assert result.graded_trades == []

    def test_engine_that_actually_trades_produces_graded_round_trips(self):
        # entry_cutoff_s=30 lets the engine actually enter and win this
        # window (see test_learning_proposer.py's identical scenario).
        windows = [strong_edge_window("KXGOLD15M-P0")]
        ticks = strong_edge_ticks()
        result = run_learning_pipeline(
            ticks=ticks,
            windows=windows,
            engine_cfg=engine_cfg(entry_cutoff_s=30.0),
            fees_cfg=fees_cfg(),
            vol_cfg=vol_cfg(),
            model_cfg=model_cfg(),
            backtest_cfg=bt_cfg(human_delay_min_s=1.0, human_delay_max_s=1.0),
            learning_cfg=learning_cfg(),
            fill_latency_samples=[],
            seed=0,
        )
        assert len(result.graded_round_trips) == 1
        assert result.graded_trades[0].net_pnl > Decimal("0")

    def test_delay_profile_reflects_provided_fill_samples(self):
        windows = [strong_edge_window("KXGOLD15M-P0")]
        ticks = strong_edge_ticks()
        samples = [
            FillLatencySample("s1", "BUY", "enter_edge", 1.5, was_missed=False) for _ in range(60)
        ]
        result = run_learning_pipeline(
            ticks=ticks,
            windows=windows,
            engine_cfg=engine_cfg(),
            fees_cfg=fees_cfg(),
            vol_cfg=vol_cfg(),
            model_cfg=model_cfg(),
            backtest_cfg=bt_cfg(human_delay_min_s=1.0, human_delay_max_s=1.0),
            learning_cfg=learning_cfg(),
            fill_latency_samples=samples,
            seed=0,
        )
        assert result.delay_profile.is_learned is True
        assert result.delay_profile.n == 60

    def test_no_proposals_with_too_few_windows(self):
        windows = [strong_edge_window("KXGOLD15M-P0")]
        ticks = strong_edge_ticks()
        result = run_learning_pipeline(
            ticks=ticks,
            windows=windows,
            engine_cfg=engine_cfg(),
            fees_cfg=fees_cfg(),
            vol_cfg=vol_cfg(),
            model_cfg=model_cfg(),
            backtest_cfg=bt_cfg(human_delay_min_s=1.0, human_delay_max_s=1.0),
            learning_cfg=learning_cfg(),
            fill_latency_samples=[],
            seed=0,
        )
        # Only one window -> no train/test split possible.
        assert result.proposals == []
