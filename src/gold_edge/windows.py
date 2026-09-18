"""Find the current/next KXGOLD15M window and cross-check the settlement
source against config at runtime (the settlement feed is known to change —
see docs/contract_notes.md), rather than trusting a hardcoded assumption.
"""

from __future__ import annotations

import logging
from datetime import datetime
from decimal import Decimal
from typing import Any

from gold_edge.feeds.kalshi_rest import KalshiRestClient
from gold_edge.models import Window

logger = logging.getLogger(__name__)


def _parse_window(market: dict[str, Any], series_ticker: str) -> Window:
    floor_strike = market.get("floor_strike")
    return Window(
        ticker=market["ticker"],
        event_ticker=market["event_ticker"],
        series_ticker=series_ticker,
        open_time=datetime.fromisoformat(market["open_time"].replace("Z", "+00:00")),
        close_time=datetime.fromisoformat(market["close_time"].replace("Z", "+00:00")),
        s0=Decimal(str(floor_strike)) if floor_strike is not None else None,
        status=market["status"],
    )


async def get_current_window(client: KalshiRestClient, series_ticker: str) -> Window | None:
    """The single 'active' (currently trading) market, if any."""
    data = await client.get_markets(series_ticker, status="open", limit=1)
    markets = data.get("markets", [])
    if not markets:
        return None
    return _parse_window(markets[0], series_ticker)


async def get_next_window(client: KalshiRestClient, series_ticker: str) -> Window | None:
    """The soonest 'initialized' (not yet open) market. S0 (floor_strike) is
    not published until the window actually opens, so it will be None here."""
    data = await client.get_markets(series_ticker, status="unopened", limit=1)
    markets = data.get("markets", [])
    if not markets:
        return None
    return _parse_window(markets[0], series_ticker)


async def check_settlement_source(
    client: KalshiRestClient, series_ticker: str, expected_symbol: str
) -> bool:
    """Warn loudly if the series' settlement_sources no longer matches the
    Pyth symbol we're configured to track (Kalshi has changed this before
    and announced doing so again — see docs/contract_notes.md). Returns
    True if it matches."""
    series = await client.get_series(series_ticker)
    sources = series.get("settlement_sources", [])
    urls = [s.get("url", "") for s in sources]
    matched = any(expected_symbol.replace("/", "%2F") in url for url in urls)
    if not matched:
        logger.warning(
            "Settlement source mismatch for %s: expected symbol %r not found in "
            "series settlement_sources %r. The live price feed may no longer "
            "match what Kalshi settles on — check docs/contract_notes.md and "
            "update config.yaml before trusting signals.",
            series_ticker,
            expected_symbol,
            urls,
        )
    return matched
