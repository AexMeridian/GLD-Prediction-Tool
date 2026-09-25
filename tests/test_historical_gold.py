from datetime import UTC, datetime, timedelta

from gold_edge.config import ModelConfig, VolatilityConfig
from gold_edge.learning.historical_gold import (
    PriceBar,
    build_calibration_points,
    evaluate_historical_calibration,
    fit_gld_session_vol_multipliers,
    format_historical_report,
    format_vol_multiplier_report,
    format_walkforward_report,
    parse_yahoo_chart_json,
    run_walkforward_rounds,
    summarize_realized_vol,
)

T0 = datetime(2026, 1, 1, tzinfo=UTC)


def vol_cfg(**overrides) -> VolatilityConfig:
    base = dict(
        ewma_half_life_s=60.0,
        short_horizon_s=60.0,
        min_sigma_per_minute=0.0005,
        vol_spike_limit=0.5,
    )
    base.update(overrides)
    return VolatilityConfig(**base)


def model_cfg(**overrides) -> ModelConfig:
    base = dict(min_fair_value=0.01, max_fair_value=0.99)
    base.update(overrides)
    return ModelConfig(**base)


def bars(prices: list[float], step_s: float = 60.0) -> list[PriceBar]:
    return [
        PriceBar(timestamp=T0 + timedelta(seconds=i * step_s), close=p)
        for i, p in enumerate(prices)
    ]


def multi_day_bars(prices_per_day: list[list[float]], step_s: float = 60.0) -> list[PriceBar]:
    out: list[PriceBar] = []
    for day_idx, day_prices in enumerate(prices_per_day):
        day_start = T0 + timedelta(days=day_idx)
        out.extend(
            PriceBar(timestamp=day_start + timedelta(seconds=i * step_s), close=p)
            for i, p in enumerate(day_prices)
        )
    return out


class TestParseYahooChartJson:
    def test_parses_timestamps_and_closes(self):
        payload = {
            "chart": {
                "result": [
                    {
                        "timestamp": [1735689600, 1735776000],
                        "indicators": {"quote": [{"close": [2000.5, 2010.25]}]},
                    }
                ]
            }
        }
        result = parse_yahoo_chart_json(payload)
        assert len(result) == 2
        assert result[0].close == 2000.5
        assert result[1].close == 2010.25

    def test_drops_null_closes(self):
        payload = {
            "chart": {
                "result": [
                    {
                        "timestamp": [1, 2, 3],
                        "indicators": {"quote": [{"close": [1.0, None, 3.0]}]},
                    }
                ]
            }
        }
        result = parse_yahoo_chart_json(payload)
        assert [b.close for b in result] == [1.0, 3.0]

    def test_malformed_payload_returns_empty(self):
        assert parse_yahoo_chart_json({}) == []
        assert parse_yahoo_chart_json({"chart": {"result": []}}) == []


class TestSummarizeRealizedVol:
    def test_too_few_bars_returns_zero(self):
        result = summarize_realized_vol(bars([2000.0]))
        assert result.n_bars == 1
        assert result.overall_sigma == 0.0

    def test_computes_nonzero_sigma_for_moving_prices(self):
        prices = [2000.0 + (i % 3) * 5.0 for i in range(50)]
        result = summarize_realized_vol(bars(prices))
        assert result.n_returns == 49
        assert result.overall_sigma > 0.0

    def test_weekday_bucket_omitted_below_min_n(self):
        prices = [2000.0 + i for i in range(10)]
        result = summarize_realized_vol(bars(prices, step_s=86400.0), min_bucket_n=30)
        assert result.by_weekday == {}


class TestBuildCalibrationPoints:
    def test_strong_uptrend_predicts_high_fair_yes_and_wins(self):
        # A steady, low-noise uptrend: the model should be confidently right.
        prices = [2000.0 * (1.0005**i) for i in range(120)]
        points = build_calibration_points(
            bars(prices), vol_cfg(), model_cfg(), window_bars=5, warmup_bars=20
        )
        assert len(points) > 0
        assert all(p.outcome == 1.0 for p in points)
        assert all(p.predicted_yes > 0.5 for p in points)

    def test_flat_series_predicts_near_half(self):
        prices = [2000.0 for _ in range(60)]
        points = build_calibration_points(
            bars(prices), vol_cfg(), model_cfg(), window_bars=5, warmup_bars=20
        )
        assert len(points) > 0
        for p in points:
            assert p.outcome == 1.0  # ties resolve YES
            assert abs(p.predicted_yes - 0.5) < 0.2

    def test_never_uses_the_window_close_as_the_current_price(self):
        # Regression test for a real bug: if the window's own close bar were
        # ever used as "current", predicted_yes and outcome would be
        # perfectly (tautologically) correlated -- every bucket showing
        # exactly 0% or 100% actual win rate. A noisy, mean-reverting series
        # makes that tautology visibly wrong if it ever creeps back in.
        prices = [2000.0 + 10.0 * ((-1) ** i) for i in range(200)]
        points = build_calibration_points(
            bars(prices), vol_cfg(), model_cfg(), window_bars=10, warmup_bars=20
        )
        assert len(points) > 0
        # Some interior points must disagree with the eventual outcome --
        # a perfect split at 0.5 would mean the close leaked into "current".
        disagreements = [p for p in points if (p.predicted_yes > 0.5) != (p.outcome == 1.0)]
        assert disagreements

    def test_max_window_minutes_excludes_wide_windows(self):
        wide_bars = [
            PriceBar(timestamp=T0, close=2000.0),
            PriceBar(timestamp=T0 + timedelta(minutes=1), close=2001.0),
            PriceBar(timestamp=T0 + timedelta(minutes=2), close=2002.0),
            PriceBar(timestamp=T0 + timedelta(hours=20), close=2003.0),
            PriceBar(timestamp=T0 + timedelta(hours=20, minutes=1), close=2004.0),
            PriceBar(timestamp=T0 + timedelta(hours=20, minutes=2), close=2005.0),
        ]
        points = build_calibration_points(
            wide_bars, vol_cfg(), model_cfg(), window_bars=2, warmup_bars=0, max_window_minutes=5.0
        )
        assert len(points) > 0
        assert all(p.tau_minutes <= 5.0 for p in points)

    def test_single_bar_window_has_no_interior_point(self):
        # window_bars=1 has no bar strictly between open and close, so
        # there is nothing to evaluate without leaking the close in as
        # "current" -- see the tautology regression test above.
        prices = [2000.0, 2001.0, 2002.0, 2003.0, 2004.0]
        points = build_calibration_points(
            bars(prices), vol_cfg(), model_cfg(), window_bars=1, warmup_bars=0
        )
        assert points == []

    def test_no_points_when_dataset_smaller_than_warmup_plus_window(self):
        prices = [2000.0, 2001.0, 2002.0]
        points = build_calibration_points(
            bars(prices), vol_cfg(), model_cfg(), window_bars=2, warmup_bars=20
        )
        assert points == []


class TestEvaluateHistoricalCalibration:
    def test_empty_points_returns_zeroed_result(self):
        result = evaluate_historical_calibration([])
        assert result.n_pairs == 0
        assert result.calibrator_would_help is None
        assert result.bucket_table == []

    def test_small_sample_skips_holdout_comparison(self):
        prices = [2000.0 * (1.0005**i) for i in range(60)]
        points = build_calibration_points(bars(prices), vol_cfg(), model_cfg(), window_bars=3)
        result = evaluate_historical_calibration(points, min_holdout_n=1000)
        assert result.n_pairs > 0
        assert result.calibrator_would_help is None
        assert result.holdout_raw_brier is None

    def test_large_sample_produces_holdout_comparison(self):
        prices = [2000.0 * (1.0002 ** (i % 40)) for i in range(400)]
        points = build_calibration_points(
            bars(prices), vol_cfg(), model_cfg(), window_bars=3, warmup_bars=10
        )
        result = evaluate_historical_calibration(points, min_holdout_n=30)
        assert result.n_pairs >= 60
        assert result.calibrator_would_help is not None
        assert result.holdout_n > 0
        assert result.holdout_raw_brier is not None
        assert result.holdout_calibrated_brier is not None

    def test_bucket_table_ns_sum_to_total_pairs(self):
        prices = [2000.0 + (i % 7) - 3 for i in range(200)]
        points = build_calibration_points(bars(prices), vol_cfg(), model_cfg(), window_bars=2)
        result = evaluate_historical_calibration(points)
        assert sum(b["n"] for b in result.bucket_table) == result.n_pairs


class TestFormatHistoricalReport:
    def test_handles_zero_pairs_without_crashing(self):
        vol_summary = summarize_realized_vol([])
        calibration = evaluate_historical_calibration([])
        report = format_historical_report("Empty", vol_summary, calibration)
        assert "not enough data" in report

    def test_includes_bucket_lines_for_real_data(self):
        prices = [2000.0 * (1.0002 ** (i % 40)) for i in range(300)]
        b = bars(prices)
        vol_summary = summarize_realized_vol(b)
        points = build_calibration_points(b, vol_cfg(), model_cfg(), window_bars=3)
        calibration = evaluate_historical_calibration(points)
        report = format_historical_report("Sample", vol_summary, calibration)
        assert "bucket: mean predicted fair_yes vs. actual win rate" in report
        assert "calibration pairs" in report


def _trending_days(n_days: int, bars_per_day: int = 150) -> list[PriceBar]:
    prices_per_day = [
        [2000.0 * (1.0003 ** (i % 30)) for i in range(bars_per_day)] for _ in range(n_days)
    ]
    return multi_day_bars(prices_per_day)


class TestRunWalkforwardRounds:
    def test_no_rounds_scored_when_too_few_points_per_round(self):
        day_bars = multi_day_bars([[2000.0 + i for i in range(10)] for _ in range(3)])
        points = build_calibration_points(
            day_bars, vol_cfg(), model_cfg(), window_bars=2, warmup_bars=0, round_by="day"
        )
        results = run_walkforward_rounds(points, min_test_n=1000)
        assert results == []

    def test_first_scored_round_has_an_empty_training_pool(self):
        points = build_calibration_points(
            _trending_days(4), vol_cfg(), model_cfg(), window_bars=5, warmup_bars=5, round_by="day"
        )
        results = run_walkforward_rounds(points, min_test_n=30, min_train_n=100)
        assert len(results) >= 1
        assert results[0].n_train == 0
        assert results[0].calibrator_would_help is None

    def test_later_rounds_get_an_expanding_training_pool_and_a_verdict(self):
        points = build_calibration_points(
            _trending_days(4), vol_cfg(), model_cfg(), window_bars=5, warmup_bars=5, round_by="day"
        )
        results = run_walkforward_rounds(points, min_test_n=30, min_train_n=100)
        assert len(results) >= 2
        for earlier, later in zip(results, results[1:], strict=False):
            assert later.n_train > earlier.n_train
        assert results[-1].calibrator_would_help is not None

    def test_never_trains_on_the_round_it_scores(self):
        points = build_calibration_points(
            _trending_days(3), vol_cfg(), model_cfg(), window_bars=5, warmup_bars=5, round_by="day"
        )
        results = run_walkforward_rounds(points, min_test_n=1, min_train_n=0)
        cumulative = 0
        for r in results:
            assert r.n_train == cumulative
            cumulative += r.n_test


class TestFitGldSessionVolMultipliers:
    def test_returns_empty_table_when_no_bucket_has_enough_data(self):
        points = build_calibration_points(
            bars([2000.0 * (1.0003 ** (i % 30)) for i in range(120)]),
            vol_cfg(),
            model_cfg(),
            window_bars=5,
            warmup_bars=5,
        )
        table = fit_gld_session_vol_multipliers(points, vol_spike_limit=0.5, min_bucket_n=10_000)
        assert table.multipliers == {}
        assert table.get("normal", "ny") == 1.0

    def test_fits_a_multiplier_for_a_well_populated_bucket(self):
        points = build_calibration_points(
            bars([2000.0 * (1.0003 ** (i % 30)) for i in range(500)]),
            vol_cfg(),
            model_cfg(),
            window_bars=5,
            warmup_bars=5,
        )
        table = fit_gld_session_vol_multipliers(points, vol_spike_limit=0.5, min_bucket_n=10)
        assert table.multipliers


class TestFormatWalkforwardReport:
    def test_handles_no_rounds(self):
        report = format_walkforward_report("Test", [])
        assert "not enough data" in report

    def test_includes_round_lines_and_a_summary(self):
        points = build_calibration_points(
            _trending_days(4), vol_cfg(), model_cfg(), window_bars=5, warmup_bars=5, round_by="day"
        )
        results = run_walkforward_rounds(points, min_test_n=30, min_train_n=100)
        report = format_walkforward_report("Test", results)
        assert "walk-forward" in report
        assert "rounds scored" in report
        assert "recalibration helped in" in report


class TestFormatVolMultiplierReport:
    def test_handles_empty_table(self):
        from gold_edge.learning.calibrator import VolMultiplierTable

        report = format_vol_multiplier_report(VolMultiplierTable(multipliers={}), min_bucket_n=30)
        assert "not enough data" in report

    def test_includes_fitted_buckets(self):
        points = build_calibration_points(
            bars([2000.0 * (1.0003 ** (i % 30)) for i in range(500)]),
            vol_cfg(),
            model_cfg(),
            window_bars=5,
            warmup_bars=5,
        )
        table = fit_gld_session_vol_multipliers(points, vol_spike_limit=0.5, min_bucket_n=10)
        report = format_vol_multiplier_report(table, min_bucket_n=10)
        assert "multiplier=" in report
