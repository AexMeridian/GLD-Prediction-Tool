"""Pydantic settings loaded from config.yaml, overridable via .env / env vars."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"


class KalshiConfig(BaseModel):
    rest_base: str
    ws_url: str
    series_ticker: str


class PythConfig(BaseModel):
    hermes_base: str
    price_feed_symbol: str
    price_feed_id: str
    price_feed_query: str


class GoldProxyConfig(BaseModel):
    ws_url: str
    product_id: str
    basis_half_life_s: float
    basis_max_pair_age_s: float


class FeesConfig(BaseModel):
    base_rate: float
    fee_multiplier: float
    maker_fees_enabled: bool


class VolatilityConfig(BaseModel):
    ewma_half_life_s: float
    short_horizon_s: float
    min_sigma_per_minute: float
    vol_spike_limit: float


class ModelConfig(BaseModel):
    min_fair_value: float
    max_fair_value: float


class EngineConfig(BaseModel):
    enter_edge: float
    persist_s: float
    cooldown_s: float
    entry_cutoff_s: float
    exit_cutoff_s: float
    stale_s: float
    max_spread: float
    max_round_trips: int
    daily_loss_stop: float
    converge_band: float
    stop: float
    hold_to_settlement_when_itm: bool
    hold_to_settlement_fair_threshold: float
    signal_ttl_s: float


class BacktestConfig(BaseModel):
    human_delay_min_s: float
    human_delay_max_s: float


class LearningConfig(BaseModel):
    min_opportunity: float
    min_bucket_n: int
    exit_regret: float
    min_proposal_trades: int
    promotion_margin: float
    max_dd_worsen: float
    shadow_sessions: int
    filter_min_prob: float
    markout_horizons_s: list[float]
    news_shock_threshold: float


class RecordingConfig(BaseModel):
    sqlite_path: str
    parquet_dir: str


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    kalshi_api_key_id: str = ""
    kalshi_private_key_path: str = ""
    pyth_api_key: str = ""

    kalshi: KalshiConfig
    pyth: PythConfig
    gold_proxy: GoldProxyConfig
    fees: FeesConfig
    volatility: VolatilityConfig
    model: ModelConfig
    engine: EngineConfig
    backtest: BacktestConfig
    learning: LearningConfig
    recording: RecordingConfig
    live_trading: bool = False

    @property
    def sqlite_path(self) -> Path:
        return REPO_ROOT / self.recording.sqlite_path

    @property
    def parquet_dir(self) -> Path:
        return REPO_ROOT / self.recording.parquet_dir

    @property
    def kalshi_private_key_full_path(self) -> Path:
        return REPO_ROOT / self.kalshi_private_key_path


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@lru_cache(maxsize=1)
def get_settings(config_path: Path = DEFAULT_CONFIG_PATH) -> Settings:
    raw = _load_yaml(config_path)
    return Settings(**raw)
