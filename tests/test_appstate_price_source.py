"""Tests for AppState.effective_price_age_source -- the one place that
decides whether the engine/dashboard trust Pyth XAU/USD, fall back to the
PAXG proxy, or have nothing usable at all. See feeds/gold_proxy.py and
model/basis.py for why this fallback exists (Pyth's free tier has no 24/7
gold feed)."""

from datetime import UTC, datetime, timedelta

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
from gold_edge.model.basis import BasisTracker
from gold_edge.server import AppState

T0 = datetime(2026, 1, 1, tzinfo=UTC)


def make_settings(stale_s: float = 3.0) -> Settings:
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
            stale_s=stale_s,
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
        recording=RecordingConfig(sqlite_path="x.sqlite", parquet_dir="x"),
    )


def make_app(stale_s: float = 3.0) -> AppState:
    return AppState(settings=make_settings(stale_s), signer=object(), pyth_api_key="x")


def test_uses_pyth_when_fresh_and_market_open():
    app = make_app()
    app.pyth_latest_price = 4380.0
    app.pyth_receive_time = T0
    app.pyth_market_open = True
    price, age, source = app.effective_price_age_source(T0 + timedelta(seconds=1))
    assert (price, source) == (4380.0, "pyth_xau")
    assert age == 1.0


def test_falls_back_to_proxy_when_pyth_market_confirmed_closed():
    app = make_app()
    app.pyth_latest_price = 4380.0
    app.pyth_receive_time = T0
    app.pyth_market_open = False  # frozen/heartbeat ticks despite a real halt
    app.proxy_latest_price = 4375.0
    app.proxy_receive_time = T0
    app.basis_tracker = BasisTracker(half_life_s=300.0)
    app.basis_tracker.update(4380.0, 4375.0, T0, T0)  # basis = 5.0

    price, age, source = app.effective_price_age_source(T0 + timedelta(seconds=1))
    assert source == "paxg_proxy"
    assert price == 4380.0  # 4375 proxy + 5.0 basis
    assert age == 1.0


def test_falls_back_to_proxy_when_pyth_receive_time_stale():
    app = make_app(stale_s=3.0)
    app.pyth_latest_price = 4380.0
    app.pyth_receive_time = T0
    app.pyth_market_open = None  # unknown, but receive-time gap alone is enough
    app.proxy_latest_price = 4370.0
    app.proxy_receive_time = T0 + timedelta(seconds=9)
    app.basis_tracker = BasisTracker(half_life_s=300.0)
    app.basis_tracker.update(4380.0, 4375.0, T0, T0)  # basis = 5.0

    price, age, source = app.effective_price_age_source(T0 + timedelta(seconds=10))
    assert source == "paxg_proxy"
    assert price == 4375.0  # 4370 proxy + 5.0 basis
    assert age == 1.0


def test_does_not_use_proxy_without_a_basis_estimate_yet():
    app = make_app()
    app.pyth_market_open = False
    app.proxy_latest_price = 4375.0
    app.proxy_receive_time = T0
    app.basis_tracker = BasisTracker(half_life_s=300.0)  # never updated

    price, age, source = app.effective_price_age_source(T0 + timedelta(seconds=1))
    assert source is None
    assert price is None


def test_nothing_usable_forces_staleness_past_threshold():
    app = make_app(stale_s=3.0)
    app.pyth_latest_price = 4380.0
    app.pyth_receive_time = T0
    app.pyth_market_open = False
    # No proxy data at all.
    price, age, source = app.effective_price_age_source(T0 + timedelta(seconds=1))
    assert source is None
    assert price == 4380.0  # last known price still surfaced for the dashboard
    assert age > 3.0  # but forced stale so the engine won't trade on it


def test_stale_proxy_is_not_used_either():
    app = make_app(stale_s=3.0)
    app.pyth_market_open = False
    app.proxy_latest_price = 4375.0
    app.proxy_receive_time = T0
    app.basis_tracker = BasisTracker(half_life_s=300.0)
    app.basis_tracker.update(4380.0, 4375.0, T0, T0)

    price, age, source = app.effective_price_age_source(T0 + timedelta(seconds=30))
    assert source is None
