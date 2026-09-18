"""Kalshi orderbook_delta WebSocket client with local book maintenance.

Kalshi's book carries bids only for YES and NO; asks are derived as
1 - opposite side's best bid (see docs/contract_notes.md). The exact
delta wire format is provisional (not confirmed against official docs)
and should be checked against a real connection early in Phase 1.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from urllib.parse import urlsplit

import websockets

from gold_edge.feeds.kalshi_auth import KalshiSigner
from gold_edge.models import BookSnapshot

logger = logging.getLogger(__name__)

WS_PATH = "/trade-api/ws/v2"


class _LocalBook:
    """One market's bid-only book for both sides, kept as price -> size."""

    def __init__(self) -> None:
        self.yes_bids: dict[Decimal, Decimal] = {}
        self.no_bids: dict[Decimal, Decimal] = {}
        self.last_seq: int | None = None

    def apply_snapshot(self, msg: dict[str, Any]) -> None:
        self.yes_bids = _levels_to_map(msg.get("yes") or msg.get("yes_dollars_fp") or [])
        self.no_bids = _levels_to_map(msg.get("no") or msg.get("no_dollars_fp") or [])
        self.last_seq = msg.get("seq")

    def apply_delta(self, msg: dict[str, Any]) -> None:
        side = msg.get("side")
        price = Decimal(str(msg.get("price_dollars", msg.get("price"))))
        delta = Decimal(str(msg.get("delta_fp", msg.get("delta", "0"))))
        book = self.yes_bids if side == "yes" else self.no_bids
        new_size = book.get(price, Decimal(0)) + delta
        if new_size <= 0:
            book.pop(price, None)
        else:
            book[price] = new_size
        self.last_seq = msg.get("seq", self.last_seq)

    def best_bid(self, book: dict[Decimal, Decimal]) -> Decimal:
        return max(book) if book else Decimal(0)

    def best_bid_size(self, book: dict[Decimal, Decimal]) -> Decimal:
        price = self.best_bid(book)
        return book.get(price, Decimal(0))

    def to_snapshot(self, window_ticker: str) -> BookSnapshot:
        yes_bid = self.best_bid(self.yes_bids)
        no_bid = self.best_bid(self.no_bids)
        return BookSnapshot(
            window_ticker=window_ticker,
            yes_bid=yes_bid,
            yes_ask=Decimal(1) - no_bid if no_bid else Decimal(1),
            yes_bid_size=self.best_bid_size(self.yes_bids),
            yes_ask_size=self.best_bid_size(self.no_bids),
            no_bid=no_bid,
            no_ask=Decimal(1) - yes_bid if yes_bid else Decimal(1),
            no_bid_size=self.best_bid_size(self.no_bids),
            no_ask_size=self.best_bid_size(self.yes_bids),
            receive_time=datetime.now(UTC),
        )


def _levels_to_map(levels: list[Any]) -> dict[Decimal, Decimal]:
    return {Decimal(str(price)): Decimal(str(size)) for price, size in levels}


class KalshiWsClient:
    """Subscribes to orderbook_delta for one market and yields BookSnapshots.

    Auto-reconnects with exponential backoff on disconnect or sequence gap.
    """

    def __init__(
        self,
        ws_url: str,
        signer: KalshiSigner,
        market_ticker: str,
        max_backoff_s: float = 30.0,
    ) -> None:
        self._ws_url = ws_url
        self._signer = signer
        self._market_ticker = market_ticker
        self._max_backoff_s = max_backoff_s
        self._book = _LocalBook()
        self._stale = True

    @property
    def is_stale(self) -> bool:
        return self._stale

    async def snapshots(self) -> AsyncIterator[BookSnapshot]:
        backoff = 1.0
        while True:
            try:
                sign_path = urlsplit(self._ws_url).path
                headers = self._signer.headers("GET", sign_path)
                async with websockets.connect(
                    self._ws_url, additional_headers=headers
                ) as ws:
                    await ws.send(
                        json.dumps(
                            {
                                "id": 1,
                                "cmd": "subscribe",
                                "params": {
                                    "channels": ["orderbook_delta"],
                                    "market_ticker": self._market_ticker,
                                },
                            }
                        )
                    )
                    self._stale = False
                    backoff = 1.0
                    async for raw in ws:
                        msg = json.loads(raw)
                        snapshot = self._handle_message(msg)
                        if snapshot is not None:
                            yield snapshot
            except (websockets.WebSocketException, OSError) as exc:
                # Covers ConnectionClosed as well as handshake/auth failures
                # (e.g. InvalidStatus on a bad or expired key) — any of
                # these should retry with backoff, not kill the stream.
                logger.warning("Kalshi WS disconnected (%s); reconnecting in %.1fs", exc, backoff)
                self._stale = True
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, self._max_backoff_s)

    def _handle_message(self, msg: dict[str, Any]) -> BookSnapshot | None:
        msg_type = msg.get("type")
        body = msg.get("msg", msg)
        if msg_type == "orderbook_snapshot":
            self._book.apply_snapshot(body)
            return self._book.to_snapshot(self._market_ticker)
        if msg_type == "orderbook_delta":
            seq = body.get("seq")
            if (
                self._book.last_seq is not None
                and seq is not None
                and seq != self._book.last_seq + 1
            ):
                logger.warning(
                    "Sequence gap on %s: expected %s, got %s; forcing reconnect",
                    self._market_ticker,
                    self._book.last_seq + 1,
                    seq,
                )
                self._stale = True
                raise websockets.ConnectionClosed(None, None)
            self._book.apply_delta(body)
            return self._book.to_snapshot(self._market_ticker)
        return None
