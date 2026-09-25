from datetime import UTC, datetime, timedelta

from gold_edge.model.basis import BasisTracker

T0 = datetime(2026, 1, 1, tzinfo=UTC)


def test_no_estimate_before_any_update():
    tracker = BasisTracker(half_life_s=300.0)
    assert tracker.has_estimate is False
    assert tracker.estimate_spot(4370.0) is None


def test_first_update_sets_basis_exactly():
    tracker = BasisTracker(half_life_s=300.0)
    tracker.update(primary_price=4380.0, proxy_price=4375.0, primary_time=T0, proxy_time=T0)
    assert tracker.has_estimate is True
    assert tracker.estimate_spot(4375.0) == 4380.0


def test_ignores_pair_too_far_apart_in_time():
    tracker = BasisTracker(half_life_s=300.0, max_pair_age_s=30.0)
    tracker.update(
        primary_price=4380.0,
        proxy_price=4375.0,
        primary_time=T0,
        proxy_time=T0 - timedelta(seconds=60),
    )
    assert tracker.has_estimate is False


def test_ewma_decays_toward_new_samples_over_time():
    tracker = BasisTracker(half_life_s=100.0)
    tracker.update(4380.0, 4375.0, T0, T0)  # basis = 5.0
    t1 = T0 + timedelta(seconds=100)  # exactly one half-life later
    tracker.update(4380.0, 4370.0, t1, t1)  # new sample basis = 10.0
    # After one half-life, the estimate should be halfway between 5 and 10.
    basis = tracker.estimate_spot(0.0)
    assert 7.4 < basis < 7.6


def test_estimate_spot_applies_current_basis_to_new_proxy_price():
    tracker = BasisTracker(half_life_s=300.0)
    tracker.update(4380.0, 4375.0, T0, T0)  # basis = 5.0
    assert tracker.estimate_spot(4360.0) == 4365.0


def test_non_positive_dt_between_updates_is_ignored():
    tracker = BasisTracker(half_life_s=300.0)
    tracker.update(4380.0, 4375.0, T0, T0)  # basis = 5.0
    tracker.update(4390.0, 4375.0, T0, T0)  # same primary_time -> dt == 0, skipped
    assert tracker.estimate_spot(0.0) == 5.0
