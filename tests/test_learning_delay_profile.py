import random

import pytest

from gold_edge.learning.delay_profile import (
    MIN_FILLS_TO_LEARN,
    DelayProfile,
    FillLatencySample,
    build_delay_profile,
    sample_delay_s,
)


def sample(delay=1.5, reason="enter_edge", missed=False, action="BUY") -> FillLatencySample:
    return FillLatencySample(
        signal_id="s1", action=action, reason=reason, delay_s=delay, was_missed=missed
    )


class TestBuildDelayProfile:
    def test_not_learned_below_min_fills(self):
        profile = build_delay_profile([sample()] * (MIN_FILLS_TO_LEARN - 1))
        assert profile.is_learned is False

    def test_learned_at_min_fills(self):
        profile = build_delay_profile([sample()] * MIN_FILLS_TO_LEARN)
        assert profile.is_learned is True
        assert profile.n == MIN_FILLS_TO_LEARN

    def test_missed_signals_excluded_from_delay_samples_but_counted_in_n(self):
        fills = [sample(delay=1.0)] * 5 + [sample(missed=True)] * 5
        profile = build_delay_profile(fills)
        assert profile.n == 5
        assert len(profile.samples) == 5

    def test_miss_rate_computed_per_reason(self):
        fills = (
            [sample(reason="enter_edge", missed=False)] * 8
            + [sample(reason="enter_edge", missed=True)] * 2
            + [sample(reason="stop", missed=False)] * 5
        )
        profile = build_delay_profile(fills)
        assert profile.miss_rate_by_reason["enter_edge"] == pytest.approx(0.2)
        assert profile.miss_rate_by_reason["stop"] == pytest.approx(0.0)

    def test_empty_input(self):
        profile = build_delay_profile([])
        assert profile.n == 0
        assert profile.is_learned is False


class TestDelayProfileMethods:
    def test_sample_delay_raises_without_data(self):
        profile = DelayProfile()
        with pytest.raises(ValueError):
            profile.sample_delay_s(random.Random(0))

    def test_sample_delay_draws_from_observed_samples(self):
        profile = DelayProfile(samples=[1.0, 2.0, 3.0], n=3)
        rng = random.Random(0)
        draws = {profile.sample_delay_s(rng) for _ in range(50)}
        assert draws <= {1.0, 2.0, 3.0}

    def test_mean_delay(self):
        profile = DelayProfile(samples=[1.0, 2.0, 3.0], n=3)
        assert profile.mean_delay_s() == pytest.approx(2.0)

    def test_percentile_delay(self):
        profile = DelayProfile(samples=[1.0, 2.0, 3.0, 4.0, 5.0], n=5)
        assert profile.percentile_delay_s(50) == pytest.approx(3.0)
        assert profile.percentile_delay_s(0) == pytest.approx(1.0)
        assert profile.percentile_delay_s(100) == pytest.approx(5.0)


class TestSampleDelayS:
    def test_uses_fallback_when_not_learned(self):
        profile = DelayProfile()
        rng = random.Random(0)
        for _ in range(20):
            d = sample_delay_s(profile, rng, fallback_min_s=1.0, fallback_max_s=2.0)
            assert 1.0 <= d <= 2.0

    def test_uses_profile_samples_once_learned(self):
        profile = DelayProfile(samples=[9.0] * MIN_FILLS_TO_LEARN, n=MIN_FILLS_TO_LEARN)
        rng = random.Random(0)
        d = sample_delay_s(profile, rng, fallback_min_s=1.0, fallback_max_s=2.0)
        assert d == 9.0
