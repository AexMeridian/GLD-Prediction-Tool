"""End-to-end smoke test for the `learn` -> `proposals` -> `promote` ->
`rollback` CLI pipeline, against a real synthetic SQLite database -- the
same kind of integration check that caught real bugs (oracle-vs-settlement,
missing dedup) in the opportunities/grader modules earlier in this project.
Unlike `record`/`live`/`fair-value`, these commands need no live network
credentials, so they're safe to exercise directly.
"""

import asyncio
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from gold_edge.cli import _run_learn, _run_promote, _run_proposals, _run_rollback
from gold_edge.config import (
    BacktestConfig,
    EngineConfig,
    FeesConfig,
    GoldProxyConfig,
    KalshiConfig,
    LearningConfig,
    ModelConfig,
    PythConfig,
    RecordingConfig,
    Settings,
    VolatilityConfig,
)
from gold_edge.models import BookSnapshot, Tick, Window
from gold_edge.recorder import Recorder

T0 = datetime(2026, 1, 1, tzinfo=UTC)


def make_settings(sqlite_path, persist_s: float = 1.5) -> Settings:
    return Settings(
        kalshi=KalshiConfig(rest_base="x", ws_url="x", series_ticker="KXGOLD15M"),
        pyth=PythConfig(
            hermes_base="x", price_feed_symbol="x", price_feed_id="x", price_feed_query="x"
        ),
        gold_proxy=GoldProxyConfig(
            ws_url="x", product_id="x", basis_half_life_s=300.0, basis_max_pair_age_s=30.0
        ),
        fees=FeesConfig(base_rate=0.07, fee_multiplier=1.0, maker_fees_enabled=True),
        volatility=VolatilityConfig(
            ewma_half_life_s=60.0,
            short_horizon_s=60.0,
            min_sigma_per_minute=0.0005,
            vol_spike_limit=0.5,
        ),
        model=ModelConfig(min_fair_value=0.01, max_fair_value=0.99),
        engine=EngineConfig(
            enter_edge=0.03,
            persist_s=persist_s,
            cooldown_s=20.0,
            entry_cutoff_s=30.0,
            exit_cutoff_s=75.0,
            stale_s=3.0,
            max_spread=0.04,
            max_round_trips=6,
            daily_loss_stop=25.0,
            converge_band=0.01,
            stop=0.08,
            hold_to_settlement_when_itm=True,
            hold_to_settlement_fair_threshold=0.90,
            signal_ttl_s=6.0,
        ),
        backtest=BacktestConfig(human_delay_min_s=1.0, human_delay_max_s=1.0),
        learning=LearningConfig(
            min_opportunity=0.03,
            min_bucket_n=30,
            exit_regret=0.02,
            min_proposal_trades=1,
            promotion_margin=0.01,
            max_dd_worsen=0.10,
            shadow_sessions=1,
            filter_min_prob=0.55,
            markout_horizons_s=[5.0, 15.0, 30.0, 60.0],
            news_shock_threshold=0.08,
        ),
        recording=RecordingConfig(
            sqlite_path=str(sqlite_path), parquet_dir=str(sqlite_path.parent)
        ),
    )


def seed_strong_edge_window(sqlite_path, ticker: str, day_offset: int = 0, length_s: float = 80.0):
    """Same crafted scenario as test_learning_proposer.py/pipeline.py: a
    real 1% move the book hasn't caught up to yet."""
    open_time = T0 + timedelta(days=day_offset)
    close_time = open_time + timedelta(seconds=length_s)
    window = Window(
        ticker=ticker,
        event_ticker=f"{ticker}EVT",
        series_ticker="KXGOLD15M",
        open_time=open_time,
        close_time=close_time,
        s0=Decimal("2000.00"),
        status="open",
    )
    recorder = Recorder(sqlite_path)
    asyncio.run(recorder.record_window(window))
    for i in range(int(length_s) + 1):
        t = open_time + timedelta(seconds=i)
        asyncio.run(
            recorder.record_tick(
                Tick(
                    symbol="Metal.XAU/USD",
                    price=2020.0,
                    conf=0.1,
                    expo=-2,
                    publish_time=t,
                    receive_time=t,
                )
            )
        )
        asyncio.run(
            recorder.record_book_snapshot(
                BookSnapshot(
                    window_ticker=ticker,
                    yes_bid=Decimal("0.83"),
                    yes_ask=Decimal("0.85"),
                    yes_bid_size=Decimal(100),
                    yes_ask_size=Decimal(100),
                    no_bid=Decimal("0.10"),
                    no_ask=Decimal("0.12"),
                    no_bid_size=Decimal(100),
                    no_ask_size=Decimal(100),
                    receive_time=t,
                )
            )
        )
    asyncio.run(recorder.record_settlement(ticker, {"result": "yes"}, "yes", close_time))
    recorder.close()


class TestLearnPipelineCli:
    def test_learn_runs_and_persists_grades_and_proposals(self, tmp_path, capsys):
        sqlite_path = tmp_path / "test.sqlite"
        for d in range(3):
            seed_strong_edge_window(sqlite_path, f"KXGOLD15M-CLI{d}", day_offset=d)
        settings = make_settings(sqlite_path)

        _run_learn(settings, start=None, end=None, seed=0)
        output = capsys.readouterr().out
        assert "Session Report Card" in output
        assert "Weekly Report" in output

        import sqlite3

        conn = sqlite3.connect(sqlite_path)
        try:
            n_grades = conn.execute("SELECT COUNT(*) FROM grades").fetchone()[0]
            n_proposals = conn.execute("SELECT COUNT(*) FROM proposals").fetchone()[0]
        finally:
            conn.close()
        assert n_grades >= 1  # entry_cutoff_s=30 lets the engine actually trade and win
        assert n_proposals >= 0  # may or may not find an improving candidate; must not crash

    def test_full_learn_and_proposals_finds_a_real_improvement(self, tmp_path, capsys):
        """`persist_s=100` on the live config means persistence can never
        clear before ENTRY_CUTOFF_S closes the entry window, so the engine
        never trades at all under the recorded/live config -- but the
        proposer's grid sweeps persist_s down to 1.0-2.0s, which DOES clear
        in time and rides the real edge to a winning settlement. This
        guarantees at least one proposal, rather than depending on chance."""
        sqlite_path = tmp_path / "test.sqlite"
        for d in range(3):
            seed_strong_edge_window(sqlite_path, f"KXGOLD15M-CYCLE{d}", day_offset=d)
        settings = make_settings(sqlite_path, persist_s=100.0)

        _run_learn(settings, start=None, end=None, seed=0)
        capsys.readouterr()

        _run_proposals(settings)
        proposals_output = capsys.readouterr().out

        import sqlite3

        conn = sqlite3.connect(sqlite_path)
        try:
            rows = conn.execute("SELECT id FROM proposals").fetchall()
        finally:
            conn.close()
        assert rows, "expected persist_s to surface as an improving proposal"

        proposal_id = rows[0][0]
        assert proposal_id in proposals_output

        # No shadow_signals were recorded for this candidate, so promotion
        # must be correctly rejected on that gate -- not silently approved.
        _run_promote(settings, proposal_id)
        promote_output = capsys.readouterr().out
        assert "Rejected" in promote_output
        assert "shadow mode" in promote_output

    def test_promote_succeeds_and_rollback_restores_prior_config(self, tmp_path, capsys):
        """Isolates _run_promote/_run_rollback from the proposer's exact
        output (which candidate wins the grid sweep isn't this test's
        concern) by inserting a proposal and matching shadow evidence
        directly, exactly as `learn` + enough shadow sessions would have."""
        sqlite_path = tmp_path / "test.sqlite"
        settings = make_settings(sqlite_path)
        recorder = Recorder(sqlite_path)
        # A prior promotion from "yesterday" -- rollback restores to the
        # previous version, so there must be one on record already (a
        # single-ever promotion has nothing earlier to roll back to; see
        # test_learning_registry.py's test_rollback_with_no_promotions_*).
        yesterday = T0 - timedelta(days=1)
        asyncio.run(
            recorder.record_config_version(
                version_hash="prior-version",
                param_changes={"enter_edge": 0.03},
                evidence={"note": "original baseline"},
                promoted=True,
                created_at=yesterday,
            )
        )
        asyncio.run(
            recorder.record_proposal(
                proposal_id="p-test",
                param_changes={"enter_edge": 0.04},
                n_holdout_trades=1,
                holdout_pnl_delta=Decimal("5.00"),
                holdout_pnl_delta_ci_low=Decimal("1.00"),
                holdout_pnl_delta_ci_high=Decimal("9.00"),
                holdout_drawdown_delta=Decimal("0.00"),
                rationale="test proposal",
                status="pending",
                created_at=T0,
            )
        )
        asyncio.run(
            recorder.record_shadow_signal(
                candidate_param_changes={"enter_edge": 0.04},
                window_ticker="KXGOLD15M-SHADOW",
                live_net_pnl=Decimal("0"),
                candidate_net_pnl=Decimal("1.00"),
                recorded_at=T0,
            )
        )
        recorder.close()

        _run_promote(settings, "p-test")
        promote_output = capsys.readouterr().out
        assert "Promoted" in promote_output

        _run_rollback(settings)
        rollback_output = capsys.readouterr().out
        assert "Rolled back" in rollback_output
        assert "'enter_edge': 0.03" in rollback_output

    def test_learn_with_no_recorded_windows_does_not_crash(self, tmp_path, capsys):
        sqlite_path = tmp_path / "empty.sqlite"
        settings = make_settings(sqlite_path)
        from gold_edge.recorder import Recorder as _R

        _R(sqlite_path).close()  # create an empty but valid schema
        _run_learn(settings, start=None, end=None, seed=0)
        output = capsys.readouterr().out
        assert "No windows" in output

    def test_proposals_with_no_data_reports_helpfully(self, tmp_path, capsys):
        sqlite_path = tmp_path / "empty.sqlite"
        settings = make_settings(sqlite_path)
        from gold_edge.recorder import Recorder as _R

        _R(sqlite_path).close()
        _run_proposals(settings)
        output = capsys.readouterr().out
        assert "No proposals" in output

    def test_rollback_with_no_promotions_reports_helpfully(self, tmp_path, capsys):
        sqlite_path = tmp_path / "empty.sqlite"
        settings = make_settings(sqlite_path)
        from gold_edge.recorder import Recorder as _R

        _R(sqlite_path).close()
        _run_rollback(settings)
        output = capsys.readouterr().out
        assert "No promoted config version" in output
