"""Pure risk/eligibility checks used to gate entries. Each function takes
plain values (no config object, no I/O) so they're trivial to unit test and
compose; state_machine.py wires them up to config and live inputs.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal


def is_stale(pyth_age_s: float, kalshi_age_s: float, stale_s: float) -> bool:
    return pyth_age_s > stale_s or kalshi_age_s > stale_s


def spread_ok(spread: Decimal, max_spread: Decimal) -> bool:
    return spread <= max_spread


def vol_ok(short_horizon_sigma_per_minute: float | None, vol_spike_limit: float) -> bool:
    """No short-horizon reading yet means not enough data to call a spike —
    that's not a reason to block, since the EWMA floor already guards
    fair_value.py against a zero denominator."""
    if short_horizon_sigma_per_minute is None:
        return True
    return short_horizon_sigma_per_minute <= vol_spike_limit


def entry_cutoff_ok(time_left_s: float, entry_cutoff_s: float) -> bool:
    return time_left_s > entry_cutoff_s


def exit_cutoff_reached(time_left_s: float, exit_cutoff_s: float) -> bool:
    return time_left_s < exit_cutoff_s


def round_trips_ok(round_trips: int, max_round_trips: int) -> bool:
    return round_trips < max_round_trips


def daily_loss_ok(realized_pnl_today: Decimal, daily_loss_stop: Decimal) -> bool:
    return realized_pnl_today > -daily_loss_stop


def cooldown_active(now: datetime, cooldown_until: datetime | None) -> bool:
    if cooldown_until is None:
        return False
    return now < cooldown_until
