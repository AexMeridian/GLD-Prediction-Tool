"""SQLite recorder. Always on during `live`, runnable alone via `record`.

Writes run through asyncio.to_thread so they never block the event loop;
sqlite3 itself serializes access via one connection per Recorder instance,
which is fine at the tick/book-update rates this tool operates at.
Decimal prices/sizes are stored as TEXT to avoid float rounding.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from gold_edge.learning.markouts import Markout
from gold_edge.models import BookSnapshot, Fill, Signal, Tick, Window

SCHEMA = """
CREATE TABLE IF NOT EXISTS ticks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    price REAL NOT NULL,
    conf REAL NOT NULL,
    expo INTEGER NOT NULL,
    publish_time TEXT NOT NULL,
    receive_time TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS book_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    window_ticker TEXT NOT NULL,
    yes_bid TEXT NOT NULL,
    yes_ask TEXT NOT NULL,
    yes_bid_size TEXT NOT NULL,
    yes_ask_size TEXT NOT NULL,
    no_bid TEXT NOT NULL,
    no_ask TEXT NOT NULL,
    no_bid_size TEXT NOT NULL,
    no_ask_size TEXT NOT NULL,
    receive_time TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_book_snapshots_window
    ON book_snapshots(window_ticker, receive_time);

CREATE TABLE IF NOT EXISTS windows (
    ticker TEXT PRIMARY KEY,
    event_ticker TEXT NOT NULL,
    series_ticker TEXT NOT NULL,
    open_time TEXT NOT NULL,
    close_time TEXT NOT NULL,
    s0 TEXT,
    status TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS settlements (
    ticker TEXT PRIMARY KEY,
    result TEXT,
    settled_time TEXT,
    raw_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS signals (
    id TEXT PRIMARY KEY,
    window_ticker TEXT NOT NULL,
    action TEXT NOT NULL,
    side TEXT NOT NULL,
    limit_price TEXT NOT NULL,
    size TEXT NOT NULL,
    fair REAL NOT NULL,
    market_price TEXT NOT NULL,
    edge_after_costs REAL NOT NULL,
    reason TEXT NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    status TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS fills (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id TEXT,
    window_ticker TEXT NOT NULL,
    side TEXT NOT NULL,
    action TEXT NOT NULL,
    price TEXT NOT NULL,
    size TEXT NOT NULL,
    logged_at TEXT NOT NULL,
    is_skip INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS markouts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id TEXT NOT NULL,
    horizon_s REAL NOT NULL,
    at TEXT NOT NULL,
    pyth_price REAL,
    market_mid REAL,
    market_bid REAL,
    market_ask REAL,
    fair REAL,
    edge_markout REAL
);
CREATE INDEX IF NOT EXISTS idx_markouts_signal ON markouts(signal_id);

CREATE TABLE IF NOT EXISTS grades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entry_signal_id TEXT NOT NULL,
    exit_signal_id TEXT,
    window_ticker TEXT NOT NULL,
    primary_grade TEXT NOT NULL,
    tags TEXT NOT NULL,
    net_pnl TEXT NOT NULL,
    graded_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_grades_window ON grades(window_ticker);

CREATE TABLE IF NOT EXISTS attribution (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    window_ticker TEXT NOT NULL,
    cause TEXT NOT NULL,
    dollar_impact TEXT NOT NULL,
    detail TEXT NOT NULL,
    attributed_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_attribution_window ON attribution(window_ticker);

CREATE TABLE IF NOT EXISTS opportunities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    window_ticker TEXT NOT NULL,
    side TEXT NOT NULL,
    at TEXT NOT NULL,
    reason TEXT NOT NULL,
    realistic_net_pnl TEXT NOT NULL,
    oracle_net_pnl TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_opportunities_window ON opportunities(window_ticker);

CREATE TABLE IF NOT EXISTS filter_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    window_ticker TEXT NOT NULL,
    side TEXT NOT NULL,
    at TEXT NOT NULL,
    reason TEXT NOT NULL,
    realistic_net_pnl TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_filter_events_window ON filter_events(window_ticker);

CREATE TABLE IF NOT EXISTS pattern_stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dimension TEXT NOT NULL,
    bucket TEXT NOT NULL,
    n INTEGER NOT NULL,
    mean_pnl TEXT NOT NULL,
    ci_low TEXT NOT NULL,
    ci_high TEXT NOT NULL,
    p_value REAL NOT NULL,
    significant INTEGER NOT NULL,
    computed_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_pattern_stats_dimension ON pattern_stats(dimension, bucket);

CREATE TABLE IF NOT EXISTS model_versions (
    version_hash TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    evidence_json TEXT NOT NULL,
    promoted INTEGER NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS config_versions (
    version_hash TEXT PRIMARY KEY,
    param_changes_json TEXT NOT NULL,
    evidence_json TEXT NOT NULL,
    promoted INTEGER NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS proposals (
    id TEXT PRIMARY KEY,
    param_changes_json TEXT NOT NULL,
    n_holdout_trades INTEGER NOT NULL,
    holdout_pnl_delta TEXT NOT NULL,
    holdout_pnl_delta_ci_low TEXT NOT NULL,
    holdout_pnl_delta_ci_high TEXT NOT NULL,
    holdout_drawdown_delta TEXT NOT NULL,
    rationale TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS shadow_signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_param_changes_json TEXT NOT NULL,
    window_ticker TEXT NOT NULL,
    live_net_pnl TEXT NOT NULL,
    candidate_net_pnl TEXT NOT NULL,
    recorded_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_shadow_signals_window ON shadow_signals(window_ticker);

CREATE TABLE IF NOT EXISTS drift_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    degraded_metrics_json TEXT NOT NULL,
    suggestion TEXT NOT NULL,
    detected_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS user_fills_latency (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id TEXT NOT NULL,
    action TEXT NOT NULL,
    reason TEXT NOT NULL,
    delay_s REAL,
    was_missed INTEGER NOT NULL
);
"""


class Recorder:
    def __init__(self, sqlite_path: Path) -> None:
        sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(sqlite_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    async def record_tick(self, tick: Tick) -> None:
        await asyncio.to_thread(self._insert_tick, tick)

    def _insert_tick(self, tick: Tick) -> None:
        self._conn.execute(
            "INSERT INTO ticks (symbol, price, conf, expo, publish_time, receive_time) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                tick.symbol,
                tick.price,
                tick.conf,
                tick.expo,
                tick.publish_time.isoformat(),
                tick.receive_time.isoformat(),
            ),
        )
        self._conn.commit()

    async def record_book_snapshot(self, book: BookSnapshot) -> None:
        await asyncio.to_thread(self._insert_book_snapshot, book)

    def _insert_book_snapshot(self, book: BookSnapshot) -> None:
        self._conn.execute(
            "INSERT INTO book_snapshots (window_ticker, yes_bid, yes_ask, yes_bid_size, "
            "yes_ask_size, no_bid, no_ask, no_bid_size, no_ask_size, receive_time) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                book.window_ticker,
                str(book.yes_bid),
                str(book.yes_ask),
                str(book.yes_bid_size),
                str(book.yes_ask_size),
                str(book.no_bid),
                str(book.no_ask),
                str(book.no_bid_size),
                str(book.no_ask_size),
                book.receive_time.isoformat(),
            ),
        )
        self._conn.commit()

    async def record_window(self, window: Window) -> None:
        await asyncio.to_thread(self._upsert_window, window)

    def _upsert_window(self, window: Window) -> None:
        self._conn.execute(
            "INSERT INTO windows (ticker, event_ticker, series_ticker, open_time, "
            "close_time, s0, status) VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(ticker) DO UPDATE SET s0=excluded.s0, status=excluded.status",
            (
                window.ticker,
                window.event_ticker,
                window.series_ticker,
                window.open_time.isoformat(),
                window.close_time.isoformat(),
                str(window.s0) if window.s0 is not None else None,
                window.status,
            ),
        )
        self._conn.commit()

    async def record_settlement(
        self, ticker: str, raw: dict, result: str | None, settled_time: datetime | None
    ) -> None:
        await asyncio.to_thread(self._upsert_settlement, ticker, raw, result, settled_time)

    def _upsert_settlement(
        self, ticker: str, raw: dict, result: str | None, settled_time: datetime | None
    ) -> None:
        self._conn.execute(
            "INSERT INTO settlements (ticker, result, settled_time, raw_json) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(ticker) DO UPDATE SET result=excluded.result, "
            "settled_time=excluded.settled_time, raw_json=excluded.raw_json",
            (ticker, result, settled_time.isoformat() if settled_time else None, json.dumps(raw)),
        )
        self._conn.commit()

    async def record_signal(self, signal: Signal) -> None:
        await asyncio.to_thread(self._insert_signal, signal)

    def _insert_signal(self, signal: Signal) -> None:
        self._conn.execute(
            "INSERT INTO signals (id, window_ticker, action, side, limit_price, size, fair, "
            "market_price, edge_after_costs, reason, created_at, expires_at, status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                signal.id,
                signal.window_ticker,
                signal.action,
                signal.side,
                str(signal.limit_price),
                str(signal.size),
                signal.fair,
                str(signal.market_price),
                signal.edge_after_costs,
                signal.reason,
                signal.created_at.isoformat(),
                signal.expires_at.isoformat(),
                signal.status,
            ),
        )
        self._conn.commit()

    async def update_signal_status(self, signal_id: str, status: str) -> None:
        await asyncio.to_thread(self._update_signal_status, signal_id, status)

    def _update_signal_status(self, signal_id: str, status: str) -> None:
        self._conn.execute("UPDATE signals SET status = ? WHERE id = ?", (status, signal_id))
        self._conn.commit()

    async def record_fill(self, fill: Fill) -> None:
        await asyncio.to_thread(self._insert_fill, fill)

    def _insert_fill(self, fill: Fill) -> None:
        self._conn.execute(
            "INSERT INTO fills (signal_id, window_ticker, side, action, price, size, "
            "logged_at, is_skip) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                fill.signal_id,
                fill.window_ticker,
                fill.side,
                fill.action,
                str(fill.price),
                str(fill.size),
                fill.logged_at.isoformat(),
                int(fill.is_skip),
            ),
        )
        self._conn.commit()

    async def record_markout(self, signal_id: str, markout: Markout) -> None:
        await asyncio.to_thread(self._insert_markout, signal_id, markout)

    def _insert_markout(self, signal_id: str, markout: Markout) -> None:
        self._conn.execute(
            "INSERT INTO markouts (signal_id, horizon_s, at, pyth_price, market_mid, "
            "market_bid, market_ask, fair, edge_markout) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                signal_id,
                markout.horizon_s,
                markout.at.isoformat(),
                markout.pyth_price,
                markout.market_mid,
                markout.market_bid,
                markout.market_ask,
                markout.fair,
                markout.edge_markout,
            ),
        )
        self._conn.commit()

    async def record_grade(
        self,
        entry_signal_id: str,
        exit_signal_id: str | None,
        window_ticker: str,
        primary_grade: str,
        tags: list[str],
        net_pnl: Decimal,
        graded_at: datetime,
    ) -> None:
        await asyncio.to_thread(
            self._insert_grade,
            entry_signal_id,
            exit_signal_id,
            window_ticker,
            primary_grade,
            tags,
            net_pnl,
            graded_at,
        )

    def _insert_grade(
        self,
        entry_signal_id: str,
        exit_signal_id: str | None,
        window_ticker: str,
        primary_grade: str,
        tags: list[str],
        net_pnl: Decimal,
        graded_at: datetime,
    ) -> None:
        self._conn.execute(
            "INSERT INTO grades (entry_signal_id, exit_signal_id, window_ticker, "
            "primary_grade, tags, net_pnl, graded_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                entry_signal_id,
                exit_signal_id,
                window_ticker,
                primary_grade,
                json.dumps(tags),
                str(net_pnl),
                graded_at.isoformat(),
            ),
        )
        self._conn.commit()

    async def record_attribution(
        self,
        window_ticker: str,
        cause: str,
        dollar_impact: Decimal,
        detail: str,
        attributed_at: datetime,
    ) -> None:
        # Plain primitives, not learning.attribution.AttributionResult/Cause:
        # importing that module here would cycle back through
        # backtest.replay -> server -> recorder, the same reason
        # record_grade takes primitives instead of GradeResult.
        await asyncio.to_thread(
            self._insert_attribution, window_ticker, cause, dollar_impact, detail, attributed_at
        )

    def _insert_attribution(
        self,
        window_ticker: str,
        cause: str,
        dollar_impact: Decimal,
        detail: str,
        attributed_at: datetime,
    ) -> None:
        self._conn.execute(
            "INSERT INTO attribution (window_ticker, cause, dollar_impact, detail, "
            "attributed_at) VALUES (?, ?, ?, ?, ?)",
            (window_ticker, cause, str(dollar_impact), detail, attributed_at.isoformat()),
        )
        self._conn.commit()

    async def record_opportunity(
        self,
        window_ticker: str,
        side: str,
        at: datetime,
        reason: str,
        realistic_net_pnl: Decimal,
        oracle_net_pnl: Decimal,
    ) -> None:
        await asyncio.to_thread(
            self._insert_opportunity,
            window_ticker,
            side,
            at,
            reason,
            realistic_net_pnl,
            oracle_net_pnl,
        )

    def _insert_opportunity(
        self,
        window_ticker: str,
        side: str,
        at: datetime,
        reason: str,
        realistic_net_pnl: Decimal,
        oracle_net_pnl: Decimal,
    ) -> None:
        self._conn.execute(
            "INSERT INTO opportunities (window_ticker, side, at, reason, realistic_net_pnl, "
            "oracle_net_pnl) VALUES (?, ?, ?, ?, ?, ?)",
            (
                window_ticker,
                side,
                at.isoformat(),
                reason,
                str(realistic_net_pnl),
                str(oracle_net_pnl),
            ),
        )
        self._conn.commit()

    async def record_filter_event(
        self, window_ticker: str, side: str, at: datetime, reason: str, realistic_net_pnl: Decimal
    ) -> None:
        await asyncio.to_thread(
            self._insert_filter_event, window_ticker, side, at, reason, realistic_net_pnl
        )

    def _insert_filter_event(
        self, window_ticker: str, side: str, at: datetime, reason: str, realistic_net_pnl: Decimal
    ) -> None:
        self._conn.execute(
            "INSERT INTO filter_events (window_ticker, side, at, reason, realistic_net_pnl) "
            "VALUES (?, ?, ?, ?, ?)",
            (window_ticker, side, at.isoformat(), reason, str(realistic_net_pnl)),
        )
        self._conn.commit()

    async def record_pattern_stat(
        self,
        dimension: str,
        bucket: str,
        n: int,
        mean_pnl: Decimal,
        ci_low: Decimal,
        ci_high: Decimal,
        p_value: float,
        significant: bool,
        computed_at: datetime,
    ) -> None:
        await asyncio.to_thread(
            self._insert_pattern_stat,
            dimension,
            bucket,
            n,
            mean_pnl,
            ci_low,
            ci_high,
            p_value,
            significant,
            computed_at,
        )

    def _insert_pattern_stat(
        self,
        dimension: str,
        bucket: str,
        n: int,
        mean_pnl: Decimal,
        ci_low: Decimal,
        ci_high: Decimal,
        p_value: float,
        significant: bool,
        computed_at: datetime,
    ) -> None:
        self._conn.execute(
            "INSERT INTO pattern_stats (dimension, bucket, n, mean_pnl, ci_low, ci_high, "
            "p_value, significant, computed_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                dimension,
                bucket,
                n,
                str(mean_pnl),
                str(ci_low),
                str(ci_high),
                p_value,
                int(significant),
                computed_at.isoformat(),
            ),
        )
        self._conn.commit()

    async def record_model_version(
        self,
        version_hash: str,
        kind: str,
        evidence: dict,
        promoted: bool,
        created_at: datetime,
    ) -> None:
        await asyncio.to_thread(
            self._upsert_model_version, version_hash, kind, evidence, promoted, created_at
        )

    def _upsert_model_version(
        self,
        version_hash: str,
        kind: str,
        evidence: dict,
        promoted: bool,
        created_at: datetime,
    ) -> None:
        self._conn.execute(
            "INSERT INTO model_versions (version_hash, kind, evidence_json, promoted, "
            "created_at) VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(version_hash) DO UPDATE SET promoted=excluded.promoted",
            (version_hash, kind, json.dumps(evidence), int(promoted), created_at.isoformat()),
        )
        self._conn.commit()

    async def record_config_version(
        self,
        version_hash: str,
        param_changes: dict,
        evidence: dict,
        promoted: bool,
        created_at: datetime,
    ) -> None:
        await asyncio.to_thread(
            self._upsert_config_version,
            version_hash,
            param_changes,
            evidence,
            promoted,
            created_at,
        )

    def _upsert_config_version(
        self,
        version_hash: str,
        param_changes: dict,
        evidence: dict,
        promoted: bool,
        created_at: datetime,
    ) -> None:
        self._conn.execute(
            "INSERT INTO config_versions (version_hash, param_changes_json, evidence_json, "
            "promoted, created_at) VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(version_hash) DO UPDATE SET promoted=excluded.promoted",
            (
                version_hash,
                json.dumps(param_changes),
                json.dumps(evidence),
                int(promoted),
                created_at.isoformat(),
            ),
        )
        self._conn.commit()

    async def record_proposal(
        self,
        proposal_id: str,
        param_changes: dict,
        n_holdout_trades: int,
        holdout_pnl_delta: Decimal,
        holdout_pnl_delta_ci_low: Decimal,
        holdout_pnl_delta_ci_high: Decimal,
        holdout_drawdown_delta: Decimal,
        rationale: str,
        status: str,
        created_at: datetime,
    ) -> None:
        await asyncio.to_thread(
            self._insert_proposal,
            proposal_id,
            param_changes,
            n_holdout_trades,
            holdout_pnl_delta,
            holdout_pnl_delta_ci_low,
            holdout_pnl_delta_ci_high,
            holdout_drawdown_delta,
            rationale,
            status,
            created_at,
        )

    def _insert_proposal(
        self,
        proposal_id: str,
        param_changes: dict,
        n_holdout_trades: int,
        holdout_pnl_delta: Decimal,
        holdout_pnl_delta_ci_low: Decimal,
        holdout_pnl_delta_ci_high: Decimal,
        holdout_drawdown_delta: Decimal,
        rationale: str,
        status: str,
        created_at: datetime,
    ) -> None:
        self._conn.execute(
            "INSERT INTO proposals (id, param_changes_json, n_holdout_trades, "
            "holdout_pnl_delta, holdout_pnl_delta_ci_low, holdout_pnl_delta_ci_high, "
            "holdout_drawdown_delta, rationale, status, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                proposal_id,
                json.dumps(param_changes),
                n_holdout_trades,
                str(holdout_pnl_delta),
                str(holdout_pnl_delta_ci_low),
                str(holdout_pnl_delta_ci_high),
                str(holdout_drawdown_delta),
                rationale,
                status,
                created_at.isoformat(),
            ),
        )
        self._conn.commit()

    async def update_proposal_status(self, proposal_id: str, status: str) -> None:
        await asyncio.to_thread(self._update_proposal_status, proposal_id, status)

    def _update_proposal_status(self, proposal_id: str, status: str) -> None:
        self._conn.execute("UPDATE proposals SET status = ? WHERE id = ?", (status, proposal_id))
        self._conn.commit()

    async def record_shadow_signal(
        self,
        candidate_param_changes: dict,
        window_ticker: str,
        live_net_pnl: Decimal,
        candidate_net_pnl: Decimal,
        recorded_at: datetime,
    ) -> None:
        await asyncio.to_thread(
            self._insert_shadow_signal,
            candidate_param_changes,
            window_ticker,
            live_net_pnl,
            candidate_net_pnl,
            recorded_at,
        )

    def _insert_shadow_signal(
        self,
        candidate_param_changes: dict,
        window_ticker: str,
        live_net_pnl: Decimal,
        candidate_net_pnl: Decimal,
        recorded_at: datetime,
    ) -> None:
        self._conn.execute(
            "INSERT INTO shadow_signals (candidate_param_changes_json, window_ticker, "
            "live_net_pnl, candidate_net_pnl, recorded_at) VALUES (?, ?, ?, ?, ?)",
            (
                json.dumps(candidate_param_changes),
                window_ticker,
                str(live_net_pnl),
                str(candidate_net_pnl),
                recorded_at.isoformat(),
            ),
        )
        self._conn.commit()

    async def record_drift_event(
        self, degraded_metrics: list[str], suggestion: str, detected_at: datetime
    ) -> None:
        await asyncio.to_thread(self._insert_drift_event, degraded_metrics, suggestion, detected_at)

    def _insert_drift_event(
        self, degraded_metrics: list[str], suggestion: str, detected_at: datetime
    ) -> None:
        self._conn.execute(
            "INSERT INTO drift_events (degraded_metrics_json, suggestion, detected_at) "
            "VALUES (?, ?, ?)",
            (json.dumps(degraded_metrics), suggestion, detected_at.isoformat()),
        )
        self._conn.commit()

    async def record_fill_latency(
        self, signal_id: str, action: str, reason: str, delay_s: float | None, was_missed: bool
    ) -> None:
        await asyncio.to_thread(
            self._insert_fill_latency, signal_id, action, reason, delay_s, was_missed
        )

    def _insert_fill_latency(
        self, signal_id: str, action: str, reason: str, delay_s: float | None, was_missed: bool
    ) -> None:
        self._conn.execute(
            "INSERT INTO user_fills_latency (signal_id, action, reason, delay_s, was_missed) "
            "VALUES (?, ?, ?, ?, ?)",
            (signal_id, action, reason, delay_s, int(was_missed)),
        )
        self._conn.commit()
