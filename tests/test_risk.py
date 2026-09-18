from datetime import UTC, datetime, timedelta
from decimal import Decimal

from gold_edge.engine import risk


def test_is_stale_false_when_within_threshold():
    assert not risk.is_stale(pyth_age_s=1.0, kalshi_age_s=1.0, stale_s=3.0)


def test_is_stale_true_when_pyth_too_old():
    assert risk.is_stale(pyth_age_s=3.1, kalshi_age_s=0.5, stale_s=3.0)


def test_is_stale_true_when_kalshi_too_old():
    assert risk.is_stale(pyth_age_s=0.5, kalshi_age_s=3.1, stale_s=3.0)


def test_is_stale_boundary_is_not_stale():
    assert not risk.is_stale(pyth_age_s=3.0, kalshi_age_s=3.0, stale_s=3.0)


def test_spread_ok():
    assert risk.spread_ok(Decimal("0.04"), Decimal("0.04"))
    assert risk.spread_ok(Decimal("0.03"), Decimal("0.04"))
    assert not risk.spread_ok(Decimal("0.05"), Decimal("0.04"))


def test_vol_ok_passes_when_no_short_horizon_reading_yet():
    # Not enough data to judge a spike -> don't block on it.
    assert risk.vol_ok(None, vol_spike_limit=0.01)


def test_vol_ok_blocks_on_spike():
    assert risk.vol_ok(0.005, vol_spike_limit=0.01)
    assert not risk.vol_ok(0.02, vol_spike_limit=0.01)


def test_entry_cutoff_ok():
    assert risk.entry_cutoff_ok(time_left_s=91, entry_cutoff_s=90)
    assert not risk.entry_cutoff_ok(time_left_s=90, entry_cutoff_s=90)
    assert not risk.entry_cutoff_ok(time_left_s=10, entry_cutoff_s=90)


def test_exit_cutoff_reached():
    assert risk.exit_cutoff_reached(time_left_s=74, exit_cutoff_s=75)
    assert not risk.exit_cutoff_reached(time_left_s=75, exit_cutoff_s=75)
    assert not risk.exit_cutoff_reached(time_left_s=200, exit_cutoff_s=75)


def test_round_trips_ok():
    assert risk.round_trips_ok(5, max_round_trips=6)
    assert not risk.round_trips_ok(6, max_round_trips=6)


def test_daily_loss_ok():
    assert risk.daily_loss_ok(Decimal("-24.99"), daily_loss_stop=Decimal("25.00"))
    assert risk.daily_loss_ok(Decimal("10"), daily_loss_stop=Decimal("25.00"))
    assert not risk.daily_loss_ok(Decimal("-25.00"), daily_loss_stop=Decimal("25.00"))
    assert not risk.daily_loss_ok(Decimal("-30"), daily_loss_stop=Decimal("25.00"))


def test_cooldown_active():
    now = datetime(2026, 1, 1, 0, 0, 20, tzinfo=UTC)
    assert risk.cooldown_active(now, cooldown_until=now + timedelta(seconds=1))
    assert not risk.cooldown_active(now, cooldown_until=now)
    assert not risk.cooldown_active(now, cooldown_until=None)
    assert not risk.cooldown_active(now, cooldown_until=now - timedelta(seconds=1))
