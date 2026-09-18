import math
from datetime import UTC, datetime, timedelta

from gold_edge.model.volatility import VolatilityTracker


def _times(n: int, step_s: float, start: datetime | None = None):
    start = start or datetime(2026, 1, 1, tzinfo=UTC)
    return [start + timedelta(seconds=i * step_s) for i in range(n)]


def test_floors_at_min_sigma_with_no_movement():
    tracker = VolatilityTracker(half_life_s=60, short_horizon_s=60, min_sigma_per_minute=0.001)
    ts = _times(20, 1.0)
    for t in ts:
        tracker.update(price=2000.0, now=t)
    assert tracker.sigma_per_minute == 0.001


def test_single_update_does_not_crash_and_stays_at_floor():
    tracker = VolatilityTracker(half_life_s=60, short_horizon_s=60, min_sigma_per_minute=0.0007)
    tracker.update(price=2000.0, now=datetime(2026, 1, 1, tzinfo=UTC))
    assert tracker.sigma_per_minute == 0.0007
    assert tracker.short_horizon_sigma_per_minute is None


def test_sigma_rises_with_larger_returns():
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    ts = _times(30, 1.0, start=t0)

    calm = VolatilityTracker(half_life_s=30, short_horizon_s=60, min_sigma_per_minute=1e-6)
    price = 2000.0
    for t in ts:
        calm.update(price=price, now=t)
        price *= 1.00001  # tiny drift

    volatile = VolatilityTracker(half_life_s=30, short_horizon_s=60, min_sigma_per_minute=1e-6)
    price = 2000.0
    sign = 1
    for t in ts:
        volatile.update(price=price, now=t)
        price *= 1 + sign * 0.002
        sign *= -1

    assert volatile.sigma_per_minute > calm.sigma_per_minute


def test_short_horizon_ignores_samples_outside_window():
    tracker = VolatilityTracker(half_life_s=60, short_horizon_s=10, min_sigma_per_minute=1e-6)
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    # Big jump far in the past, then quiet for the last 10s.
    tracker.update(price=2000.0, now=t0)
    tracker.update(price=2100.0, now=t0 + timedelta(seconds=1))
    for i in range(10):
        tracker.update(price=2100.0, now=t0 + timedelta(seconds=30 + i))

    # short horizon (last 10s) saw no movement -> should be at/near zero,
    # not inflated by the earlier jump.
    assert tracker.short_horizon_sigma_per_minute is not None
    assert tracker.short_horizon_sigma_per_minute < 0.01


def test_zero_or_negative_dt_is_ignored_not_crashed():
    tracker = VolatilityTracker(half_life_s=60, short_horizon_s=60, min_sigma_per_minute=0.001)
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    tracker.update(price=2000.0, now=t0)
    tracker.update(price=2050.0, now=t0)  # same timestamp, dt=0
    tracker.update(price=2010.0, now=t0 - timedelta(seconds=1))  # out of order
    # Should not raise, and should not divide by zero.
    assert math.isfinite(tracker.sigma_per_minute)
