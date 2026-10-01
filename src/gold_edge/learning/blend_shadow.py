"""Live SHADOW mode for the learned market+model blend (never places orders).

Runs the exact study pipeline forward in real time on free keyless sources --
Kalshi's public market endpoint for quotes/settlement, Coinbase for PAXG
prices -- and records what the blend WOULD have done, so the backtest's
optimistic assumptions (instant fills, proxy prices, no liquidity limit) get
checked against reality before anyone considers promoting it (CLAUDE.md:
shadow first, then user approval). Results go to their own sqlite file and
never touch the live recorder or the live engine.

Timing mirrors the study: a signal at minute boundary T uses only the PAXG
price known at T and the Kalshi quote fetched ~1.5s after T.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from gold_edge.config import FeesConfig, ModelConfig
from gold_edge.learning.blend import blend_probability
from gold_edge.learning.calibrator import IsotonicCalibrator
from gold_edge.learning.market_study import BookQuote, best_entry
from gold_edge.model.fair_value import compute_fair_value
from gold_edge.model.fees import taker_fee
from gold_edge.model.volatility import VolatilityTracker

WINDOW_MINUTES = 15
MIN_WARMUP_BARS = 10
MAX_PRICE_AGE = timedelta(minutes=10)  # same forward-fill limit as history.densify
MAX_SPREAD = Decimal("0.10")


def _calibrator_stamp(calibrator: IsotonicCalibrator | None) -> str:
    """Folded into the version hash so a blend-alone and a blend+isotonic
    artifact with identical weights never collide, and a tampered calibrator
    fails the same hash check as tampered weights."""
    if calibrator is None:
        return "none"
    return f"{calibrator.x_thresholds}|{calibrator.y_values}"


@dataclass(frozen=True)
class BlendArtifact:
    weights: tuple[float, float, float]
    half_life_s: float
    trained_at: str
    n_windows: int
    version_hash: str
    # Learned component #1 (CLAUDE.md), kept only if `should_promote_calibrator`
    # found it improves BOTH held-out Brier and log-loss over the blend alone
    # -- see cli.py's `_run_propose_model`. None means "blend alone" (today's
    # behavior).
    calibrator: IsotonicCalibrator | None = None

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        d = {
            "weights": list(self.weights),
            "half_life_s": self.half_life_s,
            "trained_at": self.trained_at,
            "n_windows": self.n_windows,
            "version_hash": self.version_hash,
        }
        if self.calibrator is not None:
            d["calibrator"] = {
                "x_thresholds": self.calibrator.x_thresholds,
                "y_values": self.calibrator.y_values,
            }
        path.write_text(json.dumps(d, indent=2), encoding="utf-8")

    @staticmethod
    def load(path: Path) -> BlendArtifact:
        d = json.loads(path.read_text(encoding="utf-8"))
        w = d["weights"]
        cal_d = d.get("calibrator")
        calibrator = (
            IsotonicCalibrator(x_thresholds=cal_d["x_thresholds"], y_values=cal_d["y_values"])
            if cal_d is not None
            else None
        )
        return BlendArtifact((w[0], w[1], w[2]), d["half_life_s"], d["trained_at"],
                             d["n_windows"], d["version_hash"], calibrator)


def make_artifact(
    weights: tuple[float, float, float],
    half_life_s: float,
    n_windows: int,
    now: datetime,
    calibrator: IsotonicCalibrator | None = None,
) -> BlendArtifact:
    stamp = now.astimezone(UTC).isoformat()
    digest = hashlib.sha1(
        f"{weights}|{half_life_s}|{stamp}|{_calibrator_stamp(calibrator)}".encode()
    ).hexdigest()[:10]
    return BlendArtifact(weights, half_life_s, stamp, n_windows, digest, calibrator)


def _size(market: dict, key: str) -> Decimal | None:
    try:
        return Decimal(str(market[key]))
    except (KeyError, ArithmeticError, TypeError):
        return None


def parse_market_quote(market: dict, now: datetime) -> BookQuote | None:
    """Quote from Kalshi's public single-market payload; None when the book is
    empty/crossed/too wide to trade (same filter the study used). The public
    listing also carries top-of-book sizes (`yes_bid_size_fp`/
    `yes_ask_size_fp`) -- forward-collected here (Stage 4: order-flow
    imbalance) even though nothing consumes them for a live decision yet;
    missing/malformed sizes just leave the imbalance unknown, never 0."""
    try:
        yb, ya = Decimal(market["yes_bid_dollars"]), Decimal(market["yes_ask_dollars"])
    except (KeyError, ArithmeticError, TypeError):
        return None
    if not (Decimal("0") < yb < ya < Decimal("1")) or ya - yb > MAX_SPREAD:
        return None
    return BookQuote(
        yb, ya, Decimal(1) - ya, Decimal(1) - yb, now,
        yes_bid_size=_size(market, "yes_bid_size_fp"),
        yes_ask_size=_size(market, "yes_ask_size_fp"),
    )


class PriceState:
    """Last-trade PAXG price plus the minute-sampled EWMA vol the study used."""

    def __init__(self, half_life_s: float, short_horizon_s: float = 60.0) -> None:
        self._ticks: deque[tuple[datetime, float]] = deque(maxlen=5000)
        self._vol = VolatilityTracker(half_life_s, short_horizon_s, 1e-6)
        self._bars = 0
        self._last_boundary: datetime | None = None

    def on_tick(self, price: float, t: datetime) -> None:
        self._ticks.append((t, price))

    def price_at(self, boundary: datetime) -> float | None:
        """Last trade at or before `boundary`, if not older than the
        forward-fill limit."""
        for t, p in reversed(self._ticks):
            if t <= boundary:
                return p if boundary - t <= MAX_PRICE_AGE else None
        return None

    def advance(self, boundary: datetime) -> None:
        """Feed the minute-boundary price into the vol tracker. A break
        longer than 5 minutes restarts the tracker, like the study."""
        p = self.price_at(boundary)
        if p is None:
            return
        if self._last_boundary is not None and boundary - self._last_boundary > timedelta(
            minutes=5
        ):
            self._vol = VolatilityTracker(
                self._vol.half_life_s, self._vol.short_horizon_s, self._vol.min_sigma_per_minute
            )
            self._bars = 0
        self._vol.update(p, boundary)
        self._bars += 1
        self._last_boundary = boundary

    @property
    def sigma(self) -> float | None:
        return self._vol.sigma_per_minute if self._bars >= MIN_WARMUP_BARS else None


SHADOW_SCHEMA = """
CREATE TABLE IF NOT EXISTS shadow_observations (
    ticker TEXT NOT NULL, minute INTEGER NOT NULL, ts TEXT NOT NULL,
    paxg REAL, s0_proxy REAL, sigma REAL, fair_yes REAL, yes_bid REAL, yes_ask REAL,
    blend_p REAL, model_hash TEXT, yes_bid_size REAL, yes_ask_size REAL,
    PRIMARY KEY (ticker, minute)
);
CREATE TABLE IF NOT EXISTS shadow_trades (
    ticker TEXT PRIMARY KEY, minute INTEGER NOT NULL, ts TEXT NOT NULL, side TEXT NOT NULL,
    ask REAL NOT NULL, fee REAL NOT NULL, blend_p REAL NOT NULL, fair_yes REAL NOT NULL,
    gap REAL NOT NULL, model_hash TEXT, result TEXT, pnl REAL
);
"""


class ShadowStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.executescript(SHADOW_SCHEMA)
        self._migrate()

    def _migrate(self) -> None:
        """A live `gold-edge shadow` process's sqlite file predates the
        order-flow-imbalance columns -- `CREATE TABLE IF NOT EXISTS` alone
        never adds columns to an existing table, so add them here,
        idempotently, rather than losing already-collected shadow history."""
        cols = {r[1] for r in self.conn.execute("PRAGMA table_info(shadow_observations)")}
        for col in ("yes_bid_size", "yes_ask_size"):
            if col not in cols:
                self.conn.execute(f"ALTER TABLE shadow_observations ADD COLUMN {col} REAL")
        self.conn.commit()

    def record_observation(self, row: tuple) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO shadow_observations VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", row
        )
        self.conn.commit()

    def record_trade(self, row: tuple) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO shadow_trades VALUES (?,?,?,?,?,?,?,?,?,?,NULL,NULL)", row
        )
        self.conn.commit()

    def settle(self, ticker: str, result: str) -> float | None:
        """Fills in result/pnl for the window's shadow trade (if it made one)."""
        row = self.conn.execute(
            "SELECT side, ask, fee FROM shadow_trades WHERE ticker=? AND result IS NULL", (ticker,)
        ).fetchone()
        if row is None:
            return None
        side, ask, fee = row
        won = side == result
        pnl = (1.0 if won else 0.0) - ask - fee
        self.conn.execute(
            "UPDATE shadow_trades SET result=?, pnl=? WHERE ticker=?", (result, pnl, ticker)
        )
        self.conn.commit()
        return pnl

    def unsettled_tickers(self) -> list[str]:
        return [r[0] for r in self.conn.execute(
            "SELECT ticker FROM shadow_trades WHERE result IS NULL")]


@dataclass
class ShadowWindow:
    ticker: str
    open_time: datetime
    close_time: datetime
    s0_proxy: float | None = None
    traded: bool = False


class ShadowEngine:
    def __init__(
        self,
        artifact: BlendArtifact,
        model_cfg: ModelConfig,
        fees_cfg: FeesConfig,
        threshold: float,
        prices: PriceState,
        store: ShadowStore,
    ) -> None:
        self.artifact = artifact
        self.model_cfg = model_cfg
        self.fees_cfg = fees_cfg
        self.threshold = threshold
        self.prices = prices
        self.store = store

    def on_boundary(
        self, w: ShadowWindow, boundary: datetime, quote: BookQuote | None
    ) -> str | None:
        """Handle one minute boundary of one window; returns a one-line log
        message when something notable happened. The caller must already have
        called `prices.advance(boundary)`."""
        minute = int((boundary - w.open_time).total_seconds() // 60)
        price = self.prices.price_at(boundary)
        if w.s0_proxy is None:
            # Also covers joining a window mid-way: the price history still
            # holds the price known at the window's open.
            w.s0_proxy = self.prices.price_at(w.open_time)
        if minute == 0:
            return None
        if not (1 <= minute < WINDOW_MINUTES):
            return None
        sigma = self.prices.sigma
        if w.s0_proxy is None or price is None or sigma is None or quote is None:
            return None

        fair = compute_fair_value(
            price, w.s0_proxy, sigma, float(WINDOW_MINUTES - minute),
            self.model_cfg.min_fair_value, self.model_cfg.max_fair_value,
        ).yes
        p = blend_probability(self.artifact.weights, quote.yes_mid, fair)
        if self.artifact.calibrator is not None:
            p = self.artifact.calibrator.predict(p)
        ts = quote.receive_time.isoformat()
        self.store.record_observation(
            (w.ticker, minute, ts, price, w.s0_proxy, sigma, fair, float(quote.yes_bid),
             float(quote.yes_ask), p, self.artifact.version_hash,
             float(quote.yes_bid_size) if quote.yes_bid_size is not None else None,
             float(quote.yes_ask_size) if quote.yes_ask_size is not None else None)
        )
        if w.traded:
            return None
        entry = best_entry(p, quote.yes_ask, quote.no_ask, self.fees_cfg, self.threshold)
        if entry is None:
            return None
        gap, side, ask = entry
        fee = taker_fee(
            Decimal(1), ask, Decimal(str(self.fees_cfg.fee_multiplier)),
            Decimal(str(self.fees_cfg.base_rate)),
        )
        w.traded = True
        self.store.record_trade(
            (w.ticker, minute, ts, side, float(ask), float(fee), p, fair, gap,
             self.artifact.version_hash)
        )
        return f"SHADOW BUY {side.upper()} {w.ticker} min {minute} @ {ask} (gap {gap:+.3f})"
