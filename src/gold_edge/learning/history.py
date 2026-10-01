"""Free, keyless historical backfill: months of settled KXGOLD15M windows.

Kalshi's market-data endpoints (settled markets + 1-minute candlesticks with
bid/ask) and Coinbase's PAXG candles are public and need no API key, so the
learning data no longer depends on the live recorder staying alive. Everything
lands in its OWN sqlite file (`data/history.sqlite`), never in the live
recorder's tables -- proxy prices must not blend into live/backtest data (see
backtest/replay.py's tick_source rule).

Kalshi's own fields give ground truth we can audit the PAXG proxy against:
`floor_strike` is the window's true S0 and `expiration_value` the true
settlement price, both from the Pyth index Kalshi actually settles on.
"""

from __future__ import annotations

import asyncio
import sqlite3
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

from gold_edge.learning.historical_gold import PriceBar

KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"
COINBASE_BASE = "https://api.exchange.coinbase.com"
KRAKEN_BASE = "https://api.kraken.com/0/public"
_HEADERS = {"User-Agent": "Mozilla/5.0"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS kalshi_windows (
    ticker TEXT PRIMARY KEY, open_time TEXT NOT NULL, close_time TEXT NOT NULL,
    s0 REAL, settle_value REAL, result TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS kalshi_candles (
    ticker TEXT NOT NULL, end_ts INTEGER NOT NULL,
    yes_bid_open REAL, yes_bid_close REAL, yes_ask_open REAL, yes_ask_close REAL,
    volume REAL, PRIMARY KEY (ticker, end_ts)
);
CREATE TABLE IF NOT EXISTS paxg_1m (ts INTEGER PRIMARY KEY, close REAL NOT NULL);
CREATE TABLE IF NOT EXISTS kraken_paxg_1m (ts INTEGER PRIMARY KEY, close REAL NOT NULL);
"""


@dataclass(frozen=True)
class WindowRow:
    ticker: str
    open_time: datetime
    close_time: datetime
    s0: float | None
    settle_value: float | None
    result: str


@dataclass(frozen=True)
class CandleRow:
    end_ts: int
    yes_bid_open: float | None
    yes_bid_close: float | None
    yes_ask_open: float | None
    yes_ask_close: float | None
    volume: float


def _dt(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _f(x: Any) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def parse_market(m: dict) -> WindowRow | None:
    """Only cleanly settled yes/no markets; voids and unsettled are skipped."""
    if m.get("result") not in ("yes", "no"):
        return None
    return WindowRow(
        ticker=m["ticker"],
        open_time=_dt(m["open_time"]),
        close_time=_dt(m["close_time"]),
        s0=_f(m.get("floor_strike")),
        settle_value=_f(m.get("expiration_value")),
        result=m["result"],
    )


def parse_candle(c: dict) -> CandleRow:
    bid, ask = c.get("yes_bid") or {}, c.get("yes_ask") or {}
    return CandleRow(
        end_ts=int(c["end_period_ts"]),
        yes_bid_open=_f(bid.get("open_dollars")),
        yes_bid_close=_f(bid.get("close_dollars")),
        yes_ask_open=_f(ask.get("open_dollars")),
        yes_ask_close=_f(ask.get("close_dollars")),
        volume=_f(c.get("volume_fp")) or 0.0,
    )


def parse_coinbase_candles(rows: Sequence[Sequence[float]]) -> list[PriceBar]:
    """Coinbase row = [start_epoch, low, high, open, close, volume]; a minute
    with no trades is simply absent (PAXG is thin), handled by `densify`."""
    return [PriceBar(datetime.fromtimestamp(int(r[0]), tz=UTC), float(r[4])) for r in rows]


def parse_kraken_candles(rows: Sequence[Sequence[Any]]) -> list[PriceBar]:
    """Kraken OHLC row = [time, open, high, low, close, vwap, volume, count]
    (price fields as strings). Kraken itself carries the last close forward
    into minutes with no trade (volume "0.00000000", OHLC all equal) --
    those aren't real price discovery, so they're dropped here and left to
    `densify` to fill, the same "a minute with no trades is simply absent"
    convention already used for Coinbase's PAXG candles."""
    out = []
    for r in rows:
        if float(r[6]) <= 0:
            continue
        out.append(PriceBar(datetime.fromtimestamp(int(r[0]), tz=UTC), float(r[4])))
    return out


def densify(bars: Sequence[PriceBar], max_gap_min: int = 10) -> list[PriceBar]:
    """Forward-fill minutes with no trades (the last trade IS the current
    price) for up to `max_gap_min`; longer silences stay gaps so a dead feed
    is never presented as a flat price."""
    out: list[PriceBar] = []
    for bar in sorted(bars, key=lambda b: b.timestamp):
        if out:
            prev = out[-1]
            gap = int((bar.timestamp - prev.timestamp).total_seconds() // 60)
            if 1 < gap <= max_gap_min + 1:
                out.extend(
                    PriceBar(prev.timestamp + timedelta(minutes=i), prev.close)
                    for i in range(1, gap)
                )
        out.append(bar)
    return out


def open_history(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    return conn


async def _get(client: httpx.AsyncClient, url: str, params: dict) -> Any:
    delay = 1.0
    for _ in range(6):
        resp = await client.get(url, params=params)
        if resp.status_code == 429 or resp.status_code >= 500:
            await asyncio.sleep(delay)
            delay = min(delay * 2, 30.0)
            continue
        resp.raise_for_status()
        return resp.json()
    resp.raise_for_status()
    return None


async def backfill_windows(
    conn: sqlite3.Connection, series: str, since: datetime, client: httpx.AsyncClient
) -> int:
    added, cursor = 0, ""
    while True:
        params = {"series_ticker": series, "status": "settled", "limit": 1000}
        if cursor:
            params["cursor"] = cursor
        data = await _get(client, f"{KALSHI_BASE}/markets", params)
        for m in data.get("markets", []):
            w = parse_market(m)
            if w is None or w.close_time < since:
                continue
            cur = conn.execute(
                "INSERT OR IGNORE INTO kalshi_windows VALUES (?,?,?,?,?,?)",
                (w.ticker, w.open_time.isoformat(), w.close_time.isoformat(),
                 w.s0, w.settle_value, w.result),
            )
            added += cur.rowcount
        cursor = data.get("cursor") or ""
        if not cursor or not data.get("markets"):
            break
    conn.commit()
    return added


async def backfill_candles(
    conn: sqlite3.Connection, series: str, client: httpx.AsyncClient, concurrency: int = 4
) -> int:
    todo = conn.execute(
        "SELECT ticker, open_time, close_time FROM kalshi_windows WHERE ticker NOT IN "
        "(SELECT DISTINCT ticker FROM kalshi_candles) ORDER BY open_time"
    ).fetchall()
    sem = asyncio.Semaphore(concurrency)
    done = 0

    async def one(ticker: str, o: str, c: str) -> None:
        nonlocal done
        async with sem:
            url = f"{KALSHI_BASE}/series/{series}/markets/{ticker}/candlesticks"
            params = {
                "start_ts": int(_dt(o).timestamp()),
                "end_ts": int(_dt(c).timestamp()),
                "period_interval": 1,
            }
            try:
                data = await _get(client, url, params)
            except httpx.HTTPError:
                return
            for cs in data.get("candlesticks", []):
                r = parse_candle(cs)
                conn.execute(
                    "INSERT OR IGNORE INTO kalshi_candles VALUES (?,?,?,?,?,?,?)",
                    (ticker, r.end_ts, r.yes_bid_open, r.yes_bid_close,
                     r.yes_ask_open, r.yes_ask_close, r.volume),
                )
            done += 1
            if done % 200 == 0:
                conn.commit()
                print(f"  candles: {done}/{len(todo)} windows")

    await asyncio.gather(*(one(*t) for t in todo))
    conn.commit()
    return done


async def backfill_paxg(
    conn: sqlite3.Connection, since: datetime, until: datetime, client: httpx.AsyncClient
) -> int:
    """Coinbase serves <=300 one-minute candles per call; page backwards."""
    row = conn.execute("SELECT MIN(ts), MAX(ts) FROM paxg_1m").fetchone()
    have_lo, have_hi = (row[0], row[1]) if row and row[0] is not None else (None, None)
    added = 0
    cur = since
    while cur < until:
        end = min(cur + timedelta(minutes=299), until)
        if have_lo is not None and cur.timestamp() >= have_lo and end.timestamp() <= have_hi:
            cur = end
            continue
        rows = await _get(
            client,
            f"{COINBASE_BASE}/products/PAXG-USD/candles",
            {"granularity": 60, "start": cur.isoformat(), "end": end.isoformat()},
        )
        for b in parse_coinbase_candles(rows):
            c = conn.execute(
                "INSERT OR IGNORE INTO paxg_1m VALUES (?,?)",
                (int(b.timestamp.timestamp()), b.close),
            )
            added += c.rowcount
        cur = end
        await asyncio.sleep(0.15)
    conn.commit()
    return added


async def backfill_kraken_paxg(
    conn: sqlite3.Connection, since: datetime, until: datetime, client: httpx.AsyncClient
) -> int:
    """Kraken serves up to 720 one-minute candles per call, and `since` can be
    passed to page forward using the response's own `last` cursor as the next
    `since` -- BUT in practice Kraken's free public OHLC endpoint only ever
    has roughly the trailing ~12 hours of 1-minute data available, no matter
    how far back `since` points (confirmed empirically: a `since` 60 days in
    the past still only returned the most recent ~700 candles). Unlike
    Coinbase's candles endpoint (which does serve deep history), Kraken PAXG
    is therefore NOT usable for backtesting against `kalshi_windows` history
    -- `gold-edge study-proxies` found only ~40 overlapping settled windows
    even after a 60-day backfill request. It's still collected here because
    it's free and harmless, and could become a genuine live second-exchange
    input for `gold-edge shadow` (forward-collected consensus/dislocation,
    same caveat as the plan's order-flow-imbalance idea) -- just not a
    historical feature today."""
    added = 0
    cursor = int(since.timestamp())
    until_ts = until.timestamp()
    while cursor < until_ts:
        data = await _get(
            client, f"{KRAKEN_BASE}/OHLC", {"pair": "PAXGUSD", "interval": 1, "since": cursor}
        )
        if data.get("error"):
            break
        result = data.get("result", {})
        keys = [k for k in result if k != "last"]
        if not keys:
            break
        rows = result[keys[0]]
        if not rows:
            break
        for b in parse_kraken_candles(rows):
            c = conn.execute(
                "INSERT OR IGNORE INTO kraken_paxg_1m VALUES (?,?)",
                (int(b.timestamp.timestamp()), b.close),
            )
            added += c.rowcount
        next_cursor = int(result.get("last", cursor))
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        await asyncio.sleep(0.5)
    conn.commit()
    return added


def merge_consensus(*bar_lists: Sequence[PriceBar]) -> list[PriceBar]:
    """Per-minute median across however many of the given (already densified)
    series actually have a price at that minute -- reduces single-exchange
    noise without requiring every source to be present at every minute."""
    by_ts: dict[int, list[float]] = {}
    for bars in bar_lists:
        for b in bars:
            by_ts.setdefault(int(b.timestamp.timestamp()), []).append(b.close)
    out = [
        PriceBar(datetime.fromtimestamp(ts, tz=UTC), statistics.median(prices))
        for ts, prices in by_ts.items()
    ]
    return sorted(out, key=lambda b: b.timestamp)


def load_paxg_bars(conn: sqlite3.Connection) -> list[PriceBar]:
    return [
        PriceBar(datetime.fromtimestamp(ts, tz=UTC), c)
        for ts, c in conn.execute("SELECT ts, close FROM paxg_1m ORDER BY ts")
    ]


def load_kraken_paxg_bars(conn: sqlite3.Connection) -> list[PriceBar]:
    return [
        PriceBar(datetime.fromtimestamp(ts, tz=UTC), c)
        for ts, c in conn.execute("SELECT ts, close FROM kraken_paxg_1m ORDER BY ts")
    ]
