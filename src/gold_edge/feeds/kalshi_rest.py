"""Kalshi Trade API v2 REST client.

Market discovery, series/rules lookup, and settlements/fills use REST.
Live orderbooks come from the WebSocket client (kalshi_ws.py), not here.
Market discovery and series/rules lookups are public and need no auth;
portfolio endpoints (settlements, fills) require signed requests.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit

import httpx

from gold_edge.feeds.kalshi_auth import KalshiSigner

API_PREFIX = "/trade-api/v2"


class KalshiRestClient:
    def __init__(
        self,
        rest_base: str,
        signer: KalshiSigner | None = None,
        timeout: float = 10.0,
    ) -> None:
        self._rest_base = rest_base.rstrip("/")
        self._signer = signer
        self._client = httpx.AsyncClient(base_url=self._rest_base, timeout=timeout)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> KalshiRestClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def _get(
        self, path: str, params: dict[str, Any] | None = None, auth: bool = False
    ) -> dict[str, Any]:
        headers = {}
        if auth:
            if self._signer is None:
                raise RuntimeError(f"GET {path} requires auth but no signer was configured")
            # Signature covers the path only (no query string), with the
            # /trade-api/v2 prefix from the base URL.
            sign_path = urlsplit(self._rest_base).path + path
            headers = self._signer.headers("GET", sign_path)
        response = await self._client.get(path, params=params, headers=headers)
        response.raise_for_status()
        return response.json()

    async def get_series(self, series_ticker: str) -> dict[str, Any]:
        data = await self._get(f"/series/{series_ticker}")
        return data["series"]

    async def get_markets(
        self,
        series_ticker: str,
        status: str | None = None,
        limit: int = 10,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"series_ticker": series_ticker, "limit": limit}
        if status is not None:
            params["status"] = status
        if cursor:
            params["cursor"] = cursor
        return await self._get("/markets", params=params)

    async def get_market(self, ticker: str) -> dict[str, Any]:
        data = await self._get(f"/markets/{ticker}")
        return data["market"]

    async def get_settlements(
        self, limit: int = 100, cursor: str | None = None
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"limit": limit}
        if cursor:
            params["cursor"] = cursor
        return await self._get("/portfolio/settlements", params=params, auth=True)

    async def get_fills(
        self,
        ticker: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"limit": limit}
        if ticker:
            params["ticker"] = ticker
        if cursor:
            params["cursor"] = cursor
        return await self._get("/portfolio/fills", params=params, auth=True)
