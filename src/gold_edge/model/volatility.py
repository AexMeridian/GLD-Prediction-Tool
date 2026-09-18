"""Realized volatility of the underlying Pyth price.

Two estimates, both expressed as sigma per minute (the unit fair_value.py
needs), maintained together since both are driven by the same tick stream:

- `sigma_per_minute`: an EWMA of per-second log-return variance, decayed by
  actual elapsed time rather than tick count so irregular tick spacing
  (Pyth publishes on a fixed real-time cadence, not fixed tick count)
  doesn't distort the estimate. This is mathematically equivalent to
  resampling onto a 1-second grid and EWMA-ing that (variance of a
  diffusion process is linear in elapsed time regardless of sampling
  grid), but needs no separate resampling step.
- `short_horizon_sigma_per_minute`: plain realized volatility over the
  trailing `short_horizon_s` window (no decay) — used to detect sudden
  vol spikes against a threshold, where a smoothed EWMA would react too
  slowly.

Floors at `min_sigma_per_minute` so a perfectly quiet feed never produces a
zero denominator downstream in fair_value.py.
"""

from __future__ import annotations

import math
from collections import deque
from datetime import datetime, timedelta

_LN2 = math.log(2)


class VolatilityTracker:
    def __init__(
        self, half_life_s: float, short_horizon_s: float, min_sigma_per_minute: float
    ) -> None:
        self.half_life_s = half_life_s
        self.short_horizon_s = short_horizon_s
        self.min_sigma_per_minute = min_sigma_per_minute

        self._last_price: float | None = None
        self._last_time: datetime | None = None
        self._ewma_variance_per_second: float = 0.0
        self._ewma_initialized = False

        self._short_samples: deque[tuple[datetime, float]] = deque()

    def update(self, price: float, now: datetime) -> None:
        self._update_short_horizon(price, now)

        if self._last_price is None or self._last_time is None:
            self._last_price = price
            self._last_time = now
            return

        dt = (now - self._last_time).total_seconds()
        if dt <= 0:
            # Duplicate or out-of-order timestamp; skip rather than divide
            # by zero or move time backwards.
            return

        log_return = math.log(price / self._last_price)
        sq_return_per_second = (log_return * log_return) / dt
        decay = math.exp(-_LN2 / self.half_life_s * dt)
        if not self._ewma_initialized:
            self._ewma_variance_per_second = sq_return_per_second
            self._ewma_initialized = True
        else:
            self._ewma_variance_per_second = (
                1 - decay
            ) * sq_return_per_second + decay * self._ewma_variance_per_second
        self._last_price = price
        self._last_time = now

    def _update_short_horizon(self, price: float, now: datetime) -> None:
        self._short_samples.append((now, price))
        cutoff = now - timedelta(seconds=self.short_horizon_s)
        while self._short_samples and self._short_samples[0][0] < cutoff:
            self._short_samples.popleft()

    @property
    def sigma_per_minute(self) -> float:
        variance_per_minute = self._ewma_variance_per_second * 60.0
        sigma = math.sqrt(variance_per_minute) if variance_per_minute > 0 else 0.0
        return max(sigma, self.min_sigma_per_minute)

    @property
    def short_horizon_sigma_per_minute(self) -> float | None:
        samples = list(self._short_samples)
        if len(samples) < 2:
            return None
        total_sq_return = 0.0
        total_dt = 0.0
        for (t0, p0), (t1, p1) in zip(samples, samples[1:], strict=False):
            dt = (t1 - t0).total_seconds()
            if dt <= 0:
                continue
            r = math.log(p1 / p0)
            total_sq_return += r * r
            total_dt += dt
        if total_dt <= 0:
            return None
        variance_per_second = total_sq_return / total_dt
        return math.sqrt(variance_per_second * 60.0)
