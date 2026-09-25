"""24/7 fallback gold price via PAXG (Pax Gold, redeemable 1:1 for physical
gold and closely arbitraged to spot), streamed from Coinbase's public
Exchange WebSocket feed.

Pyth's XAU/USD feed is free but tracks real commodity trading hours (daily
halt, closed weekends); Kalshi's KXGOLD15M settles on a 24/7 feed that
requires Pyth's paid `pyth-indices` entitlement. Coinbase's `ticker` channel
needs no API key or account and is 24/7, so it stands in for spot during
exactly the gap the free Pyth plan can't cover. See docs/contract_notes.md
and model/basis.py for how the two are reconciled -- this client only
streams raw proxy ticks, tagged `source="paxg_proxy"`; it makes no claim
about how close PAXG is to spot at any given moment.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

import websockets

from gold_edge.models import Tick

logger = logging.getLogger(__name__)

SOURCE = "paxg_proxy"


class ProxyFeedClient:
    """Streams last-trade ticks for one Coinbase product, auto-reconnecting
    with backoff. Mirrors PythFeedClient's/KalshiWsClient's interface so
    callers can treat it the same way."""

    def __init__(
        self,
        ws_url: str,
        product_id: str,
        max_backoff_s: float = 30.0,
    ) -> None:
        self._ws_url = ws_url
        self._product_id = product_id
        self._max_backoff_s = max_backoff_s
        self.last_receive_time: datetime | None = None

    def is_stale(self, now: datetime, stale_s: float) -> bool:
        if self.last_receive_time is None:
            return True
        return (now - self.last_receive_time).total_seconds() > stale_s

    async def ticks(self) -> AsyncIterator[Tick]:
        backoff = 1.0
        subscribe_msg = json.dumps(
            {
                "type": "subscribe",
                "channels": [{"name": "ticker", "product_ids": [self._product_id]}],
            }
        )
        while True:
            try:
                async with websockets.connect(self._ws_url) as ws:
                    await ws.send(subscribe_msg)
                    backoff = 1.0
                    async for raw in ws:
                        tick = self._parse_message(raw)
                        if tick is not None:
                            self.last_receive_time = tick.receive_time
                            yield tick
            except (websockets.WebSocketException, OSError) as exc:
                logger.warning(
                    "Gold proxy WS disconnected (%s); reconnecting in %.1fs", exc, backoff
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, self._max_backoff_s)

    def _parse_message(self, raw: str) -> Tick | None:
        try:
            msg: dict[str, Any] = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Malformed gold proxy WS payload: %s", raw[:200])
            return None
        return parse_ticker_message(msg, self._product_id)


def parse_ticker_message(msg: dict[str, Any], product_id: str) -> Tick | None:
    """Pure parser for one Coinbase `ticker` channel message, e.g.:

        {"type": "ticker", "product_id": "PAXG-USD", "price": "4373.55",
         "best_bid": "4372.41", "best_ask": "4373.55",
         "time": "2026-09-18T21:59:33.242569Z", ...}

    Returns None for any other message type (subscription acks, heartbeats,
    errors) or a message for a different product."""
    if msg.get("type") != "ticker" or msg.get("product_id") != product_id:
        return None
    price = msg.get("price")
    time_str = msg.get("time")
    if price is None or time_str is None:
        return None
    conf = 0.0
    best_bid, best_ask = msg.get("best_bid"), msg.get("best_ask")
    if best_bid is not None and best_ask is not None:
        conf = abs(float(best_ask) - float(best_bid)) / 2.0
    return Tick(
        symbol=product_id,
        price=float(price),
        conf=conf,
        expo=0,
        publish_time=datetime.fromisoformat(time_str.replace("Z", "+00:00")),
        receive_time=datetime.now(UTC),
        source=SOURCE,
    )
