import random

import pytest

from gold_edge.learning.patterns import RoundTripFeatures
from gold_edge.learning.trade_filter import (
    encode_features,
    filter_beats_baseline,
    fit_trade_filter,
)


def feat(minute=5, side="YES", gap="0.05-0.08", **overrides) -> RoundTripFeatures:
    defaults = dict(
        minute_of_window=minute,
        session="london",
        gap_bucket=gap,
        side=side,
        spread_bucket="0.01-0.02",
        vol_regime="normal",
        sigma_distance_bucket="<0.5σ",
        time_since_last_trade_bucket="first_trade",
        day_of_week="Monday",
        macro_proximity_bucket="none_nearby",
    )
    defaults.update(overrides)
    return RoundTripFeatures(**defaults)


class TestEncodeFeatures:
    def test_reuses_training_vocabulary_for_unseen_bucket_values(self):
        train = [feat(side="YES"), feat(side="NO")]
        matrix, vocab = encode_features(train)
        unseen = [feat(side="YES", day_of_week="Sunday")]  # "Sunday" not in vocab
        matrix2, _ = encode_features(unseen, vocab)
        assert matrix2.shape[1] == matrix.shape[1]

    def test_minute_of_window_is_numeric_not_one_hot(self):
        matrix, _ = encode_features([feat(minute=3), feat(minute=10)])
        assert matrix[0, 0] == 3.0
        assert matrix[1, 0] == 10.0


class TestFitTradeFilter:
    def test_raises_on_mismatched_lengths(self):
        with pytest.raises(ValueError):
            fit_trade_filter([feat()], [True, False])

    def test_raises_with_too_few_examples(self):
        with pytest.raises(ValueError):
            fit_trade_filter([feat()], [True])

    def test_learns_a_clear_separating_pattern(self):
        rng = random.Random(0)
        features = []
        profitable = []
        for _ in range(100):
            gap = rng.choice(["0.03-0.05", "0.08+"])
            features.append(feat(gap=gap))
            # 0.08+ gap almost always profitable; 0.03-0.05 almost never --
            # a clean, learnable pattern the filter should pick up.
            is_profitable = (gap == "0.08+") == (rng.random() > 0.05)
            profitable.append(is_profitable)
        model = fit_trade_filter(features, profitable, l2=0.1)

        high_gap_prob = model.predict_proba(feat(gap="0.08+"))
        low_gap_prob = model.predict_proba(feat(gap="0.03-0.05"))
        assert high_gap_prob > low_gap_prob
        assert high_gap_prob > 0.6
        assert low_gap_prob < 0.4

    def test_predict_proba_many_matches_predict_proba(self):
        features = [feat(gap="0.03-0.05"), feat(gap="0.08+")] * 10
        profitable = [False, True] * 10
        model = fit_trade_filter(features, profitable)
        single = [model.predict_proba(f) for f in features]
        batch = model.predict_proba_many(features)
        for a, b in zip(single, batch, strict=True):
            assert a == pytest.approx(b, abs=1e-9)

    def test_unseen_bucket_value_still_predicts_without_error(self):
        features = [feat(gap="0.03-0.05"), feat(gap="0.08+")] * 10
        profitable = [False, True] * 10
        model = fit_trade_filter(features, profitable)
        prob = model.predict_proba(feat(gap="0.08+", day_of_week="Sunday"))
        assert 0.0 <= prob <= 1.0


class TestFilterBeatsBaseline:
    def test_beats_when_margin_cleared(self):
        assert filter_beats_baseline(10.0, 12.0, promotion_margin=1.0) is True

    def test_does_not_beat_when_margin_not_cleared(self):
        assert filter_beats_baseline(10.0, 10.5, promotion_margin=1.0) is False

    def test_exact_margin_counts_as_beating(self):
        assert filter_beats_baseline(10.0, 11.0, promotion_margin=1.0) is True
