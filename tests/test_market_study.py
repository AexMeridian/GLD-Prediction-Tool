from datetime import UTC, datetime, timedelta
from decimal import Decimal

from gold_edge.config import FeesConfig, ModelConfig
from gold_edge.learning.historical_gold import PriceBar
from gold_edge.learning.market_study import (
    BookQuote,
    GldSeries,
    StudyRow,
    WindowRef,
    build_study_rows,
    compare_brier,
    format_study,
    hold_to_settlement,
    load_gld_cache,
    proxy_agreement,
    save_gld_cache,
)

T0 = datetime(2026, 9, 21, 14, 0, tzinfo=UTC)
MODEL = ModelConfig(min_fair_value=0.01, max_fair_value=0.99)
FEES = FeesConfig(base_rate=0.07, fee_multiplier=1.0, maker_fees_enabled=True)


def minute_bars(prices: list[float], start: datetime = T0) -> list[PriceBar]:
    return [PriceBar(start + timedelta(minutes=i), p) for i, p in enumerate(prices)]


def quote(yes_bid="0.50", yes_ask="0.52", no_bid="0.48", no_ask="0.50", t=T0) -> BookQuote:
    return BookQuote(Decimal(yes_bid), Decimal(yes_ask), Decimal(no_bid), Decimal(no_ask), t)


def window(result="yes", open_time=T0 + timedelta(minutes=20)) -> WindowRef:
    return WindowRef("W1", open_time, open_time + timedelta(minutes=15), result)


class TestGldSeries:
    def test_price_known_at_boundary_is_the_bar_that_just_closed(self):
        gld = GldSeries(minute_bars([100.0, 101.0]), 900.0, 60.0)
        assert gld.price_at(T0 + timedelta(minutes=1)) == 100.0
        assert gld.price_at(T0 + timedelta(minutes=2)) == 101.0
        assert gld.price_at(T0) is None

    def test_sigma_only_after_warmup_and_reset_after_a_gap(self):
        prices = [100.0 + (i % 3) for i in range(30)]
        gld = GldSeries(minute_bars(prices), 900.0, 60.0)
        assert gld.sigma_at(T0 + timedelta(minutes=5)) is None
        assert gld.sigma_at(T0 + timedelta(minutes=25)) is not None
        next_day = minute_bars(prices, T0 + timedelta(days=1))
        gld2 = GldSeries(minute_bars(prices) + next_day, 900.0, 60.0)
        assert gld2.sigma_at(T0 + timedelta(days=1, minutes=3)) is None


class TestProxyAgreement:
    def test_counts_matches_and_ties_resolve_yes(self):
        w = window("yes")
        prices = [100.0] * 60
        gld = GldSeries(minute_bars(prices), 900.0, 60.0)
        assert proxy_agreement([w], gld) == (1, 1)
        assert proxy_agreement([window("no")], gld) == (1, 0)

    def test_skips_windows_without_gld_coverage(self):
        gld = GldSeries(minute_bars([100.0] * 5), 900.0, 60.0)
        assert proxy_agreement([window()], gld) == (0, 0)


class TestBuildStudyRows:
    def test_builds_interior_rows_using_only_prices_known_at_the_boundary(self):
        prices = [100.0 + 0.05 * (i % 4) for i in range(80)]
        gld = GldSeries(minute_bars(prices), 900.0, 60.0)
        seen = []

        def quote_at(ticker, t):
            seen.append(t)
            return quote(t=t)

        rows = build_study_rows([window()], gld, quote_at, MODEL, delay_s=1.5)
        assert 1 <= len(rows) <= 14
        assert all(r.outcome == 1.0 for r in rows)
        w = window()
        assert all((t - w.open_time).total_seconds() % 60 == 1.5 for t in seen)

    def test_rows_dropped_when_no_quote(self):
        prices = [100.0 + 0.05 * (i % 4) for i in range(80)]
        gld = GldSeries(minute_bars(prices), 900.0, 60.0)
        assert build_study_rows([window()], gld, lambda *_: None, MODEL) == []


def rows_for(outcomes: list[float], fair: float, mid_pair=("0.50", "0.52")) -> list[StudyRow]:
    return [
        StudyRow(f"W{i}", 5, fair, quote(*mid_pair), o) for i, o in enumerate(outcomes)
    ]


class TestCompareBrier:
    def test_empty(self):
        assert compare_brier([]) is None

    def test_model_that_knows_the_answer_beats_a_coin_flip_market(self):
        rows = [
            StudyRow(f"W{i}", 5, 0.95 if i % 2 else 0.05, quote(), 1.0 if i % 2 else 0.0)
            for i in range(40)
        ]
        result = compare_brier(rows, n_boot=200)
        assert result is not None
        assert result.diff > 0 and result.ci_low > 0
        assert result.n_windows == 40

    def test_report_names_a_verdict(self):
        rows = rows_for([1.0, 0.0] * 10, 0.5)
        text = format_study("t", 20, 18, compare_brier(rows, n_boot=100), [])
        assert "Brier" in text and "90.0%" in text


class TestHoldToSettlement:
    def test_pnl_arithmetic_matches_fee_formula(self):
        # fair 0.90 vs yes ask 0.52: fee = ceil(0.07*.52*.48*100)/100 = 0.02
        rows = [StudyRow("W1", 3, 0.90, quote(), 1.0)]
        res = hold_to_settlement(rows, FEES, 0.03, n_boot=50)
        assert res.n_trades == 1
        assert abs(res.mean_pnl - (1.0 - 0.52 - 0.02)) < 1e-9

    def test_loser_and_one_trade_per_window(self):
        rows = [
            StudyRow("W1", 3, 0.90, quote(), 0.0),
            StudyRow("W1", 4, 0.90, quote(), 0.0),
        ]
        res = hold_to_settlement(rows, FEES, 0.03, n_boot=50)
        assert res.n_trades == 1
        assert res.mean_pnl < 0 and res.win_rate == 0.0

    def test_no_trade_below_threshold(self):
        rows = [StudyRow("W1", 3, 0.53, quote(), 1.0)]
        assert hold_to_settlement(rows, FEES, 0.03).n_trades == 0

    def test_buys_no_side_when_that_is_the_edge(self):
        rows = [StudyRow("W1", 3, 0.05, quote(), 0.0)]
        res = hold_to_settlement(rows, FEES, 0.03, n_boot=50)
        assert res.n_trades == 1 and res.win_rate == 1.0


def test_gld_cache_round_trip_and_merge(tmp_path):
    path = tmp_path / "gld.json"
    save_gld_cache(path, minute_bars([1.0, 2.0]))
    save_gld_cache(path, minute_bars([2.5, 3.0], T0 + timedelta(minutes=1)))
    loaded = load_gld_cache(path)
    assert [b.close for b in loaded] == [1.0, 2.5, 3.0]
