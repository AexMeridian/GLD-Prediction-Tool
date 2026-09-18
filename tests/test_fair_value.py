import math

import pytest

from gold_edge.model.fair_value import FairValue, compute_fair_value

MIN_FV = 0.01
MAX_FV = 0.99


def _fv(s, s0, sigma, tau) -> FairValue:
    return compute_fair_value(s, s0, sigma, tau, MIN_FV, MAX_FV)


def test_price_above_s0_favors_yes():
    fv = _fv(s=2010.0, s0=2000.0, sigma=0.01, tau=5.0)
    assert fv.yes > 0.5
    assert fv.no < 0.5


def test_price_below_s0_favors_no():
    fv = _fv(s=1990.0, s0=2000.0, sigma=0.01, tau=5.0)
    assert fv.yes < 0.5
    assert fv.no > 0.5


def test_yes_and_no_sum_to_one():
    fv = _fv(s=2005.0, s0=2000.0, sigma=0.01, tau=3.0)
    assert fv.yes + fv.no == pytest.approx(1.0)


def test_s_equals_s0_with_time_left_is_fifty_fifty():
    fv = _fv(s=2000.0, s0=2000.0, sigma=0.01, tau=5.0)
    assert fv.yes == pytest.approx(0.5, abs=1e-9)


def test_tau_zero_with_s_above_s0_resolves_yes():
    fv = _fv(s=2001.0, s0=2000.0, sigma=0.01, tau=0.0)
    assert fv.yes == MAX_FV  # clamped ceiling, deterministic YES


def test_tau_zero_with_s_below_s0_resolves_no():
    fv = _fv(s=1999.0, s0=2000.0, sigma=0.01, tau=0.0)
    assert fv.yes == MIN_FV  # clamped floor, deterministic NO


def test_tau_zero_with_tie_resolves_yes():
    # Contract rule: ties resolve YES.
    fv = _fv(s=2000.0, s0=2000.0, sigma=0.01, tau=0.0)
    assert fv.yes == MAX_FV


def test_negative_tau_treated_like_zero():
    fv = _fv(s=2001.0, s0=2000.0, sigma=0.01, tau=-1.0)
    assert fv.yes == MAX_FV


def test_zero_sigma_does_not_crash_and_behaves_like_tau_zero():
    fv = _fv(s=2001.0, s0=2000.0, sigma=0.0, tau=5.0)
    assert fv.yes == MAX_FV
    fv_below = _fv(s=1999.0, s0=2000.0, sigma=0.0, tau=5.0)
    assert fv_below.yes == MIN_FV


def test_clamped_to_min_and_max():
    # Enormous move relative to vol/time should saturate, not go to 0 or 1.
    fv_high = _fv(s=3000.0, s0=2000.0, sigma=0.001, tau=1.0)
    assert fv_high.yes == MAX_FV
    fv_low = _fv(s=1000.0, s0=2000.0, sigma=0.001, tau=1.0)
    assert fv_low.yes == MIN_FV


def test_larger_sigma_pulls_fair_value_toward_half():
    tight = _fv(s=2010.0, s0=2000.0, sigma=0.005, tau=5.0)
    wide = _fv(s=2010.0, s0=2000.0, sigma=0.05, tau=5.0)
    assert wide.yes < tight.yes
    assert wide.yes > 0.5


def test_more_time_left_pulls_fair_value_toward_half():
    soon = _fv(s=2010.0, s0=2000.0, sigma=0.01, tau=1.0)
    later = _fv(s=2010.0, s0=2000.0, sigma=0.01, tau=14.0)
    assert later.yes < soon.yes
    assert later.yes > 0.5


def test_rejects_non_positive_prices():
    with pytest.raises(ValueError):
        compute_fair_value(0.0, 2000.0, 0.01, 5.0, MIN_FV, MAX_FV)
    with pytest.raises(ValueError):
        compute_fair_value(2000.0, -1.0, 0.01, 5.0, MIN_FV, MAX_FV)


def test_result_is_finite():
    fv = _fv(s=2500.0, s0=2000.0, sigma=0.02, tau=0.01)
    assert math.isfinite(fv.yes)
    assert math.isfinite(fv.no)
