"""Tracks the live premium/discount between the true Pyth spot price and a
24/7 proxy price (PAXG tokenized gold), so the proxy can stand in for spot
during Pyth XAU/USD's daily halt and weekend closures without introducing a
silent bias.

PAXG is a token redeemable 1:1 for physical gold and is closely arbitraged
to spot, but it isn't identical to it -- there's a small, slowly-drifting
premium/discount. We track that offset with a time-decayed EWMA (same
convention as model/volatility.py) whenever BOTH prices are observed close
together in time, so `estimate_spot` can correct for it once the primary
feed goes stale.
"""

from __future__ import annotations

import math
from datetime import datetime

_LN2 = math.log(2)


class BasisTracker:
    def __init__(self, half_life_s: float, max_pair_age_s: float = 30.0) -> None:
        self.half_life_s = half_life_s
        self.max_pair_age_s = max_pair_age_s

        self._basis: float | None = None
        self._last_update: datetime | None = None

    def update(
        self,
        primary_price: float,
        proxy_price: float,
        primary_time: datetime,
        proxy_time: datetime,
    ) -> None:
        """Fold in one (primary, proxy) observation pair, as long as they're
        close enough in time to be comparable. Call this whenever both the
        real spot feed and the proxy feed are fresh -- not just while the
        primary is confirmed open, since a few extra samples right at the
        open/close boundary only sharpen the estimate."""
        if abs((primary_time - proxy_time).total_seconds()) > self.max_pair_age_s:
            return
        sample = primary_price - proxy_price
        if self._basis is None or self._last_update is None:
            self._basis = sample
            self._last_update = primary_time
            return
        dt = (primary_time - self._last_update).total_seconds()
        if dt <= 0:
            return
        decay = math.exp(-_LN2 / self.half_life_s * dt)
        self._basis = (1 - decay) * sample + decay * self._basis
        self._last_update = primary_time

    @property
    def has_estimate(self) -> bool:
        return self._basis is not None

    def estimate_spot(self, proxy_price: float) -> float | None:
        """None (not an assumed-zero offset) until at least one real
        (primary, proxy) pair has been observed -- an honest "we don't know
        the offset yet" rather than pretending the proxy tracks 1:1."""
        if self._basis is None:
            return None
        return proxy_price + self._basis
