import random
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from gold_edge.backtest.fills import BookHistory, sample_delay_s, simulate_fill
from gold_edge.config import BacktestConfig
from gold_edge.engine.signals import build_signal
from gold_edge.models import Action, Side
from tests.test_state_machine import book

T0 = datetime(2026, 1, 1, tzinfo=UTC)


def buy_signal(ttl_s=6.0, limit="0.50", now=T0, side=Side.YES):
    return build_signal(
        action=Action.BUY,
        side=side,
        window_ticker="KXGOLD15M-TEST",
        book=book(yes_ask=limit, no_ask=limit, now=now),
        fair=0.60,
        size=Decimal(1),
        edge_after_costs=0.04,
        reason="enter_edge",
        now=now,
        ttl_s=ttl_s,
    )


def sell_signal(ttl_s=6.0, limit="0.55", now=T0, side=Side.YES):
    return build_signal(
        action=Action.SELL,
        side=side,
        window_ticker="KXGOLD15M-TEST",
        book=book(yes_bid=limit, no_bid=limit, now=now),
        fair=0.55,
        size=Decimal(1),
        edge_after_costs=0.01,
        reason="converged",
        now=now,
        ttl_s=ttl_s,
    )


def bt_cfg(**overrides):
    defaults = dict(human_delay_min_s=1.0, human_delay_max_s=2.0)
    defaults.update(overrides)
    return BacktestConfig(**defaults)


class TestSampleDelay:
    def test_within_configured_bounds(self):
        rng = random.Random(0)
        cfg = bt_cfg(human_delay_min_s=1.0, human_delay_max_s=2.0)
        for _ in range(200):
            d = sample_delay_s(cfg, rng)
            assert 1.0 <= d <= 2.0

    def test_deterministic_given_seeded_rng(self):
        cfg = bt_cfg()
        a = sample_delay_s(cfg, random.Random(42))
        b = sample_delay_s(cfg, random.Random(42))
        assert a == b


class TestBookHistory:
    def test_at_or_before_returns_none_when_no_snapshots_yet(self):
        history = BookHistory([])
        assert history.at_or_before(T0) is None

    def test_at_or_before_picks_latest_not_future(self):
        b1 = book(yes_bid="0.40", yes_ask="0.42", now=T0)
        b2 = book(yes_bid="0.45", yes_ask="0.47", now=T0 + timedelta(seconds=5))
        history = BookHistory([b1, b2])
        assert history.at_or_before(T0 + timedelta(seconds=1)) is b1
        assert history.at_or_before(T0 + timedelta(seconds=10)) is b2
        assert history.at_or_before(T0 - timedelta(seconds=1)) is None

    def test_between_is_exclusive_start_inclusive_end(self):
        b1 = book(now=T0)
        b2 = book(now=T0 + timedelta(seconds=5))
        b3 = book(now=T0 + timedelta(seconds=10))
        history = BookHistory([b1, b2, b3])
        result = history.between(T0, T0 + timedelta(seconds=10))
        assert result == [b2, b3]


class TestSimulateFill:
    def test_fills_when_book_still_honors_limit_price(self):
        signal = buy_signal(limit="0.50", now=T0)
        # Book unchanged through the delay window: still offers 0.50 or better.
        history = BookHistory([book(yes_ask="0.50", now=T0 + timedelta(seconds=1))])
        fill = simulate_fill(signal, history, delay_s=1.5)
        assert fill is not None
        assert fill.price == Decimal("0.50")
        assert fill.size == signal.size
        assert fill.side == Side.YES
        assert fill.action == Action.BUY
        assert fill.logged_at == T0 + timedelta(seconds=1.5)
        assert fill.is_skip is False

    def test_misses_when_ask_moves_above_limit_before_delay_elapses(self):
        signal = buy_signal(limit="0.50", now=T0)
        # Ask jumps to 0.53 shortly after signal, before the 1.5s delay elapses.
        history = BookHistory([book(yes_ask="0.53", now=T0 + timedelta(seconds=0.5))])
        fill = simulate_fill(signal, history, delay_s=1.5)
        assert fill is None

    def test_does_not_chase_even_if_price_reverts_before_delay_elapses(self):
        """CLAUDE.md: expired/invalidated signals are never re-issued at a
        worse price. If the ask moved past the limit at any point before the
        human could have clicked, it's a miss — even if the book happens to
        recover to the original price by the time the delay elapses."""
        signal = buy_signal(limit="0.50", now=T0)
        moved = book(yes_ask="0.53", now=T0 + timedelta(seconds=0.5))
        reverted = book(yes_ask="0.50", now=T0 + timedelta(seconds=1.4))
        history = BookHistory([moved, reverted])
        fill = simulate_fill(signal, history, delay_s=1.5)
        assert fill is None

    def test_misses_when_delay_pushes_past_ttl_expiry(self):
        signal = buy_signal(limit="0.50", now=T0, ttl_s=1.0)
        history = BookHistory([book(yes_ask="0.50", now=T0 + timedelta(seconds=1))])
        fill = simulate_fill(signal, history, delay_s=1.5)
        assert fill is None

    def test_fills_sell_signal_when_bid_still_at_or_above_limit(self):
        signal = sell_signal(limit="0.55", now=T0)
        history = BookHistory([book(yes_bid="0.55", now=T0 + timedelta(seconds=1))])
        fill = simulate_fill(signal, history, delay_s=1.5)
        assert fill is not None
        assert fill.action == Action.SELL
        assert fill.price == Decimal("0.55")

    def test_misses_sell_signal_when_bid_drops_below_limit(self):
        signal = sell_signal(limit="0.55", now=T0)
        history = BookHistory([book(yes_bid="0.52", now=T0 + timedelta(seconds=0.5))])
        fill = simulate_fill(signal, history, delay_s=1.5)
        assert fill is None

    def test_no_book_data_at_all_is_a_miss(self):
        signal = buy_signal(limit="0.50", now=T0)
        history = BookHistory([])
        fill = simulate_fill(signal, history, delay_s=1.5)
        assert fill is None

    def test_book_holding_steady_below_limit_improvement_still_fills(self):
        """A better-or-equal price never invalidates — only a strictly worse
        one does (mirrors engine.signals.price_moved_past_limit)."""
        signal = buy_signal(limit="0.50", now=T0)
        better = book(yes_ask="0.48", now=T0 + timedelta(seconds=1))
        history = BookHistory([better])
        fill = simulate_fill(signal, history, delay_s=1.5)
        assert fill is not None
