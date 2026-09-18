import random

import numpy as np
import pytest

from gold_edge.learning.calibrator import (
    brier_score,
    fit_isotonic_calibrator,
    fit_vol_multipliers,
    log_loss,
    should_promote_calibrator,
)


class TestFitIsotonicCalibrator:
    def test_raises_on_mismatched_lengths(self):
        with pytest.raises(ValueError):
            fit_isotonic_calibrator([0.1, 0.2], [1.0])

    def test_raises_on_empty_input(self):
        with pytest.raises(ValueError):
            fit_isotonic_calibrator([], [])

    def test_fitted_curve_is_monotonic(self):
        rng = random.Random(0)
        raw = sorted(rng.uniform(0, 1) for _ in range(200))
        # Ground truth: raw is systematically overconfident (true prob is
        # compressed toward 0.5) -- isotonic should learn a flatter curve.
        targets = [0.5 + 0.5 * (r - 0.5) for r in raw]
        cal = fit_isotonic_calibrator(raw, targets)
        preds = cal.predict_many(sorted(cal.x_thresholds))
        assert all(a <= b + 1e-9 for a, b in zip(preds, preds[1:], strict=False))

    def test_predict_interpolates_between_training_points(self):
        cal = fit_isotonic_calibrator([0.0, 1.0], [0.0, 1.0])
        assert cal.predict(0.5) == pytest.approx(0.5, abs=0.01)

    def test_predict_clamps_outside_training_range(self):
        cal = fit_isotonic_calibrator([0.2, 0.8], [0.3, 0.7])
        assert cal.predict(0.0) == pytest.approx(0.3)
        assert cal.predict(1.0) == pytest.approx(0.7)

    def test_recovers_a_known_miscalibration(self):
        # Raw fair_yes of 0.9 is actually only right 60% of the time --
        # a classic overconfidence pattern the calibrator should correct.
        # Targets are shuffled within each block: PAVA respects the ORDER
        # of individual points, not just each block's mean, so an
        # unshuffled run of 1s-then-0s would create spurious local
        # violations that have nothing to do with the miscalibration.
        rng = random.Random(5)
        high_targets = [1.0] * 60 + [0.0] * 40
        rng.shuffle(high_targets)
        low_targets = [0.0] * 60 + [1.0] * 40
        rng.shuffle(low_targets)
        raw = [0.9] * 100 + [0.1] * 100
        targets = high_targets + low_targets
        cal = fit_isotonic_calibrator(raw, targets)
        assert cal.predict(0.9) == pytest.approx(0.6, abs=0.05)
        assert cal.predict(0.1) == pytest.approx(0.4, abs=0.05)


class TestMetrics:
    def test_brier_score_zero_for_perfect_predictions(self):
        assert brier_score([1.0, 0.0], [1.0, 0.0]) == 0.0

    def test_brier_score_positive_for_wrong_predictions(self):
        assert brier_score([0.0, 1.0], [1.0, 0.0]) == pytest.approx(1.0)

    def test_log_loss_low_for_confident_correct_predictions(self):
        assert log_loss([0.99, 0.01], [1.0, 0.0]) < 0.05

    def test_log_loss_high_for_confident_wrong_predictions(self):
        assert log_loss([0.01, 0.99], [1.0, 0.0]) > 3.0


class TestShouldPromoteCalibrator:
    def test_promotes_when_calibrated_improves_both_metrics(self):
        raw = [0.9] * 50 + [0.1] * 50
        outcomes = [1.0] * 30 + [0.0] * 20 + [0.0] * 30 + [1.0] * 20
        calibrated = [0.6] * 50 + [0.4] * 50  # closer to the true 60/40 split
        assert should_promote_calibrator(raw, calibrated, outcomes) is True

    def test_does_not_promote_when_raw_is_already_well_calibrated(self):
        raw = [0.9] * 50 + [0.1] * 50
        outcomes = [1.0] * 45 + [0.0] * 5 + [0.0] * 45 + [1.0] * 5
        worse_calibrated = [0.6] * 50 + [0.4] * 50
        assert should_promote_calibrator(raw, worse_calibrated, outcomes) is False


class TestFitVolMultipliers:
    def test_falls_back_to_one_for_unseen_bucket(self):
        table = fit_vol_multipliers({}, min_bucket_n=30)
        assert table.get("low", "asia") == 1.0

    def test_skips_buckets_below_min_n(self):
        triples = [(0.002, 1.0, 1.0)] * 10
        table = fit_vol_multipliers({("low", "asia"): triples}, min_bucket_n=30)
        assert table.get("low", "asia") == 1.0

    def test_picks_a_factor_that_widens_an_overconfident_z(self):
        # z is systematically too extreme for the true outcome rate implied
        # -- scaling sigma UP (dividing z by a factor > 1) should fit better
        # than leaving it alone.
        rng = random.Random(3)
        n = 200
        true_z = np.array([rng.uniform(-1.0, 1.0) for _ in range(n)])
        from scipy.stats import norm

        outcomes = (norm.cdf(true_z) > np.array([rng.uniform(0, 1) for _ in range(n)])).astype(
            float
        )
        overconfident_z = true_z * 2.0  # what the (mis-specified) model actually used
        triples = list(zip([0.002] * n, overconfident_z.tolist(), outcomes.tolist(), strict=True))
        table = fit_vol_multipliers({("high", "ny"): triples}, min_bucket_n=30)
        assert table.get("high", "ny") > 1.0
