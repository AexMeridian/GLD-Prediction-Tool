"""Pyth Hermes SSE client.

As of the Aug 2026 Pyth Core upgrade, every Hermes route (including the
price-update stream) requires `Authorization: Bearer <PYTH_API_KEY>` and
the canonical host is https://pyth.dourolabs.app/hermes. See
docs/contract_notes.md. The feed-id lookup endpoint (`/v2/price_feeds`) is
metadata-only and does not require the key.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from datetime import UTC, datetime

import httpx

from gold_edge.models import Tick

logger = logging.getLogger(__name__)


async def resolve_feed_id(hermes_base: str, symbol_query: str, asset_type: str = "metal") -> str:
    """Look up a Hermes feed id by symbol/description substring. Never
    hand-guess feed ids — always resolve them this way (or pin a value
    that was obtained this way, as config.yaml does)."""
    async with httpx.AsyncClient(base_url=hermes_base, timeout=10.0) as client:
        resp = await client.get(
            "/v2/price_feeds", params={"asset_type": asset_type, "query": symbol_query}
        )
        resp.raise_for_status()
        results = resp.json()
    if not results:
        raise ValueError(f"No Hermes feed found for query={symbol_query!r}")
    if len(results) > 1:
        symbols = [r["attributes"]["symbol"] for r in results]
        raise ValueError(f"Ambiguous Hermes feed query={symbol_query!r}: {symbols}")
    return results[0]["id"]


async def get_market_hours(
    hermes_base: str, feed_id: str, symbol_query: str, asset_type: str = "metal"
) -> bool | None:
    """Whether the underlying market for this feed is currently open.

    Standard (non-Index) feeds like XAU/USD track real trading hours and
    pause daily. Pure receive-time staleness detection (is_stale below)
    can't tell a paused market apart from a live one if Hermes keeps
    emitting periodic ticks through the pause — this is an independent,
    authoritative check for exactly that gap. Returns None if the feed
    can't be found (treat as "unknown", not "closed" — a lookup failure
    shouldn't halt trading on its own).
    """
    async with httpx.AsyncClient(base_url=hermes_base, timeout=10.0) as client:
        resp = await client.get(
            "/v2/price_feeds", params={"asset_type": asset_type, "query": symbol_query}
        )
        resp.raise_for_status()
        results = resp.json()
    for item in results:
        if item.get("id") == feed_id:
            return item.get("market_hours", {}).get("is_open")
    return None


class PythFeedClient:
    """Streams ticks for one feed id, auto-reconnecting with backoff."""

    def __init__(
        self,
        hermes_base: str,
        api_key: str,
        feed_id: str,
        max_backoff_s: float = 30.0,
        idle_timeout_s: float = 60.0,
    ) -> None:
        self._hermes_base = hermes_base.rstrip("/")
        self._api_key = api_key
        self._feed_id = feed_id
        self._max_backoff_s = max_backoff_s
        # An SSE connection has no protocol-level keepalive (unlike
        # WebSocket's ping/pong -- see kalshi_ws.py/gold_proxy.py, which
        # get this for free from the `websockets` library). A silently
        # dead TCP connection (no FIN/RST -- common after a NAT timeout or
        # a long idle stretch) leaves `timeout=None` waiting forever with
        # no exception ever raised, so reconnect-with-backoff never fires.
        # This bounds how long we'll wait for the NEXT message before
        # treating the connection as dead and reconnecting.
        self._idle_timeout_s = idle_timeout_s
        self.last_receive_time: datetime | None = None

    def is_stale(self, now: datetime, stale_s: float) -> bool:
        if self.last_receive_time is None:
            return True
        return (now - self.last_receive_time).total_seconds() > stale_s

    async def ticks(self) -> AsyncIterator[Tick]:
        backoff = 1.0
        url = f"{self._hermes_base}/v2/updates/price/stream"
        headers = {"Authorization": f"Bearer {self._api_key}"}
        params = {"ids[]": self._feed_id}
        while True:
            try:
                async with httpx.AsyncClient(timeout=None) as client:
                    async with client.stream(
                        "GET", url, headers=headers, params=params
                    ) as response:
                        response.raise_for_status()
                        backoff = 1.0
                        lines = response.aiter_lines()
                        while True:
                            try:
                                line = await asyncio.wait_for(
                                    lines.__anext__(), timeout=self._idle_timeout_s
                                )
                            except StopAsyncIteration:
                                break
                            tick = self._parse_line(line)
                            if tick is not None:
                                self.last_receive_time = tick.receive_time
                                yield tick
            except (TimeoutError, httpx.HTTPError) as exc:
                logger.warning("Pyth Hermes stream error (%s); reconnecting in %.1fs", exc, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, self._max_backoff_s)

    def _parse_line(self, line: str) -> Tick | None:
        if not line.startswith("data:"):
            return None
        payload = line[len("data:") :].strip()
        if not payload:
            return None
        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            logger.warning("Malformed Hermes SSE payload: %s", payload[:200])
            return None
        parsed = event.get("parsed") or []
        for item in parsed:
            if item.get("id") != self._feed_id:
                continue
            price_obj = item["price"]
            expo = int(price_obj["expo"])
            return Tick(
                symbol=self._feed_id,
                price=int(price_obj["price"]) * (10**expo),
                conf=int(price_obj["conf"]) * (10**expo),
                expo=expo,
                publish_time=datetime.fromtimestamp(int(price_obj["publish_time"]), tz=UTC),
                receive_time=datetime.now(UTC),
            )
        return None
