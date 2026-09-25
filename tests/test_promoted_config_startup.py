"""A promotion (learning/actions.py's promote_proposal) only ever persists
a new config_versions row to SQLite. server._settings_with_promoted_config
is where that promise is actually kept -- read once at `live` startup, so
an approved proposal takes effect the next session instead of being
silently recorded and never applied. See CLAUDE.md's "no change
mid-session": this must happen at startup, never mid-session."""

import asyncio
from datetime import UTC, datetime

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
from gold_edge.recorder import Recorder
from gold_edge.server import _settings_with_promoted_config

T0 = datetime(2026, 1, 1, tzinfo=UTC)


def make_settings(sqlite_path) -> Settings:
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
            persist_s=1.5,
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
            filter_min_prob=0.5,
            markout_horizons_s=[5.0, 15.0, 30.0, 60.0],
            news_shock_threshold=0.08,
        ),
        recording=RecordingConfig(
            sqlite_path=str(sqlite_path), parquet_dir=str(sqlite_path.parent)
        ),
    )


def test_no_sqlite_file_yet_returns_settings_unchanged(tmp_path):
    settings = make_settings(tmp_path / "missing.sqlite")
    result = _settings_with_promoted_config(settings, tmp_path / "missing.sqlite")
    assert result is settings


def test_no_promoted_version_returns_settings_unchanged(tmp_path):
    sqlite_path = tmp_path / "test.sqlite"
    Recorder(sqlite_path).close()
    settings = make_settings(sqlite_path)
    result = _settings_with_promoted_config(settings, sqlite_path)
    assert result.engine.enter_edge == 0.03


def test_promoted_version_overrides_engine_params(tmp_path):
    sqlite_path = tmp_path / "test.sqlite"
    recorder = Recorder(sqlite_path)
    asyncio.run(
        recorder.record_config_version(
            version_hash="abc123",
            param_changes={"enter_edge": 0.05, "stop": 0.10},
            evidence={},
            promoted=True,
            created_at=T0,
        )
    )
    recorder.close()

    settings = make_settings(sqlite_path)
    result = _settings_with_promoted_config(settings, sqlite_path)
    assert result.engine.enter_edge == 0.05
    assert result.engine.stop == 0.10
    # Untouched fields keep their config.yaml value.
    assert result.engine.persist_s == 1.5


def test_only_the_latest_promoted_version_applies(tmp_path):
    sqlite_path = tmp_path / "test.sqlite"
    recorder = Recorder(sqlite_path)
    asyncio.run(
        recorder.record_config_version(
            version_hash="v1",
            param_changes={"enter_edge": 0.05},
            evidence={},
            promoted=True,
            created_at=T0,
        )
    )
    asyncio.run(
        recorder.record_config_version(
            version_hash="v2",
            param_changes={"enter_edge": 0.06},
            evidence={},
            promoted=True,
            created_at=T0.replace(hour=1),
        )
    )
    recorder.close()

    settings = make_settings(sqlite_path)
    result = _settings_with_promoted_config(settings, sqlite_path)
    assert result.engine.enter_edge == 0.06


def test_unpromoted_version_is_ignored(tmp_path):
    sqlite_path = tmp_path / "test.sqlite"
    recorder = Recorder(sqlite_path)
    asyncio.run(
        recorder.record_config_version(
            version_hash="rejected",
            param_changes={"enter_edge": 0.09},
            evidence={},
            promoted=False,
            created_at=T0,
        )
    )
    recorder.close()

    settings = make_settings(sqlite_path)
    result = _settings_with_promoted_config(settings, sqlite_path)
    assert result.engine.enter_edge == 0.03
