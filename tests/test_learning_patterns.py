import random
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from gold_edge.engine.state_machine import EngineState, MarketSnapshot
from gold_edge.learning.patterns import (
    BucketStat,
    MacroEvent,
    aggregate_patterns,
    apply_benjamini_hochberg,
    build_round_trip_features,
    compute_bucket_stats,
    load_macro_events,
)
from gold_edge.model.fair_value import FairValue
from tests.test_backtest_replay import make_window
from tests.test_learning_grader import rt
from tests.test_state_machine import book, fees_cfg

T0 = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)  # noon UTC -> "london" session


def snap(now, fair_yes=0.55, s0="2000.00", underlying=2001.0, sigma=0.002):
    win = make_window().model_copy(
        update={
            "open_time": T0 - timedelta(minutes=1),
            "close_time": T0 + timedelta(minutes=14),
            "s0": Decimal(s0),
        }
    )
    return (
        EngineState(),
        MarketSnapshot(
            now=now,
            window=win,
            book=book(yes_bid="0.50", yes_ask="0.52", now=now),
            fair=FairValue(yes=fair_yes, no=1 - fair_yes),
            pyth_age_s=0.1,
            kalshi_age_s=0.1,
            short_horizon_sigma_per_minute=sigma,
            underlying_price=underlying,
        ),
    )


class TestBuildRoundTripFeatures:
    def test_extracts_features_from_nearest_snapshot(self):
        trace = [snap(T0), snap(T0 + timedelta(seconds=10))]
        times = [s.now for _, s in trace]
        rtr = rt(pnl="0.30", fees="0.05", entry_time=T0 + timedelta(seconds=5))
        features = build_round_trip_features(rtr, times, trace, fees_cfg(), 0.01, None)
        assert features is not None
        assert features.side == "YES"
        assert features.session == "london"
        assert features.day_of_week == rtr.entry_time.strftime("%A")

    def test_returns_none_when_no_snapshot_at_or_before_entry(self):
        trace = [snap(T0 + timedelta(seconds=10))]
        times = [s.now for _, s in trace]
        rtr = rt(pnl="0.30", fees="0.05", entry_time=T0)
        assert build_round_trip_features(rtr, times, trace, fees_cfg(), 0.01, None) is None

    def test_time_since_last_trade_first_trade_bucket(self):
        trace = [snap(T0)]
        times = [s.now for _, s in trace]
        rtr = rt(pnl="0.30", fees="0.05", entry_time=T0)
        features = build_round_trip_features(rtr, times, trace, fees_cfg(), 0.01, None)
        assert features.time_since_last_trade_bucket == "first_trade"

    def test_time_since_last_trade_buckets_by_gap(self):
        trace = [snap(T0)]
        times = [s.now for _, s in trace]
        rtr = rt(pnl="0.30", fees="0.05", entry_time=T0)
        near = build_round_trip_features(
            rtr, times, trace, fees_cfg(), 0.01, T0 - timedelta(seconds=30)
        )
        far = build_round_trip_features(
            rtr, times, trace, fees_cfg(), 0.01, T0 - timedelta(seconds=600)
        )
        assert near.time_since_last_trade_bucket == "<60s"
        assert far.time_since_last_trade_bucket == "300s+"

    def test_sigma_distance_unknown_without_underlying_price(self):
        trace = [
            (
                EngineState(),
                MarketSnapshot(
                    now=T0,
                    window=make_window().model_copy(
                        update={
                            "open_time": T0 - timedelta(minutes=1),
                            "close_time": T0 + timedelta(minutes=14),
                            "s0": Decimal("2000.00"),
                        }
                    ),
                    book=book(yes_bid="0.50", yes_ask="0.52", now=T0),
                    fair=FairValue(yes=0.55, no=0.45),
                    pyth_age_s=0.1,
                    kalshi_age_s=0.1,
                    short_horizon_sigma_per_minute=0.002,
                    underlying_price=None,
                ),
            )
        ]
        times = [T0]
        rtr = rt(pnl="0.30", fees="0.05", entry_time=T0)
        features = build_round_trip_features(rtr, times, trace, fees_cfg(), 0.01, None)
        assert features.sigma_distance_bucket == "unknown"

    def test_macro_proximity_within_5_minutes(self):
        trace = [snap(T0)]
        times = [s.now for _, s in trace]
        rtr = rt(pnl="0.30", fees="0.05", entry_time=T0)
        events = [MacroEvent(name="CPI", at=T0 + timedelta(minutes=2))]
        features = build_round_trip_features(rtr, times, trace, fees_cfg(), 0.01, None, events)
        assert features.macro_proximity_bucket == "<5m"

    def test_macro_proximity_none_nearby_without_events(self):
        trace = [snap(T0)]
        times = [s.now for _, s in trace]
        rtr = rt(pnl="0.30", fees="0.05", entry_time=T0)
        features = build_round_trip_features(rtr, times, trace, fees_cfg(), 0.01, None, [])
        assert features.macro_proximity_bucket == "none_nearby"


class TestLoadMacroEvents:
    def test_missing_file_returns_empty(self, tmp_path):
        assert load_macro_events(tmp_path / "nope.json") == []

    def test_loads_events_from_json(self, tmp_path):
        p = tmp_path / "events.json"
        p.write_text('[{"name": "CPI", "at": "2026-01-01T12:00:00+00:00"}]', encoding="utf-8")
        events = load_macro_events(p)
        assert len(events) == 1
        assert events[0].name == "CPI"


class TestComputeBucketStats:
    def test_drops_buckets_below_min_n(self):
        by_bucket = {"a": [Decimal("0.1")] * 5, "b": [Decimal("0.1")] * 40}
        stats = compute_bucket_stats("dim", by_bucket, min_bucket_n=30, rng=random.Random(0))
        assert {s.bucket for s in stats} == {"b"}

    def test_mean_pnl_matches_sample_mean(self):
        by_bucket = {"a": [Decimal("1.0")] * 40}
        stats = compute_bucket_stats("dim", by_bucket, min_bucket_n=30, rng=random.Random(0))
        assert stats[0].mean_pnl == Decimal("1.0")
        assert stats[0].n == 40

    def test_ci_excludes_zero_and_low_p_value_for_clearly_positive_bucket(self):
        rng = random.Random(42)
        pnls = [Decimal(str(1.0 + rng.uniform(-0.05, 0.05))) for _ in range(50)]
        stats = compute_bucket_stats("dim", {"a": pnls}, min_bucket_n=30, rng=random.Random(1))
        assert stats[0].ci_low > Decimal("0")
        assert stats[0].p_value < 0.05


class TestBenjaminiHochberg:
    def test_empty_input(self):
        assert apply_benjamini_hochberg([]) == []

    def test_marks_clearly_significant_bucket_and_not_a_noisy_one(self):
        stats = [
            BucketStat("dim", "sig", 40, Decimal("1.0"), Decimal("0.9"), Decimal("1.1"), 0.001),
            BucketStat("dim", "noise", 40, Decimal("0.0"), Decimal("-0.5"), Decimal("0.5"), 0.90),
        ]
        result = apply_benjamini_hochberg(stats, alpha=0.05)
        by_bucket = {s.bucket: s for s in result}
        assert by_bucket["sig"].significant is True
        assert by_bucket["noise"].significant is False

    def test_all_high_p_values_yield_no_significant_buckets(self):
        stats = [
            BucketStat("dim", "a", 40, Decimal("0"), Decimal("-1"), Decimal("1"), 0.5),
            BucketStat("dim", "b", 40, Decimal("0"), Decimal("-1"), Decimal("1"), 0.6),
        ]
        result = apply_benjamini_hochberg(stats, alpha=0.05)
        assert all(not s.significant for s in result)


class TestAggregatePatterns:
    def test_aggregates_across_dimensions_and_respects_min_n(self):
        trace = [snap(T0)]
        times = [s.now for _, s in trace]
        rng = random.Random(7)
        features_and_pnl = []
        for _ in range(40):
            rtr = rt(pnl=str(round(1.0 + rng.uniform(-0.05, 0.05), 4)), fees="0.05", entry_time=T0)
            features = build_round_trip_features(rtr, times, trace, fees_cfg(), 0.01, None)
            features_and_pnl.append((features, rtr.pnl))
        stats = aggregate_patterns(features_and_pnl, min_bucket_n=30, rng=random.Random(2))
        side_stats = [s for s in stats if s.dimension == "side"]
        assert len(side_stats) == 1
        assert side_stats[0].bucket == "YES"
        assert side_stats[0].significant is True

    def test_too_few_trades_produces_no_buckets(self):
        trace = [snap(T0)]
        times = [s.now for _, s in trace]
        rtr = rt(pnl="0.30", fees="0.05", entry_time=T0)
        features = build_round_trip_features(rtr, times, trace, fees_cfg(), 0.01, None)
        stats = aggregate_patterns([(features, rtr.pnl)], min_bucket_n=30)
        assert stats == []
