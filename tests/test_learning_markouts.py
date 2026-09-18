from datetime import UTC, datetime, timedelta
from decimal import Decimal

from gold_edge.backtest.fills import BookHistory
from gold_edge.engine.signals import build_signal
from gold_edge.learning.markouts import compute_markouts
from gold_edge.models import Action, Side, Tick
from tests.test_backtest_replay import model_cfg, vol_cfg
from tests.test_state_machine import book, window

T0 = datetime(2026, 1, 1, tzinfo=UTC)


def tick(price: float, t: datetime) -> Tick:
    return Tick(
        symbol="Metal.XAU/USD", price=price, conf=0.1, expo=-2, publish_time=t, receive_time=t
    )


def buy_signal(now=T0, side=Side.YES, limit="0.50", fair=0.55):
    return build_signal(
        action=Action.BUY,
        side=side,
        window_ticker="KXGOLD15M-TEST",
        book=book(yes_ask=limit, no_ask=limit, now=now),
        fair=fair,
        size=Decimal(1),
        edge_after_costs=0.02,
        reason="enter_edge",
        now=now,
        ttl_s=6.0,
    )


def sell_signal(now=T0, side=Side.YES, limit="0.55", fair=0.55):
    return build_signal(
        action=Action.SELL,
        side=side,
        window_ticker="KXGOLD15M-TEST",
        book=book(yes_bid=limit, no_bid=limit, now=now),
        fair=fair,
        size=Decimal(1),
        edge_after_costs=0.01,
        reason="converged",
        now=now,
        ttl_s=6.0,
    )


class TestComputeMarkouts:
    def test_returns_one_markout_per_horizon(self):
        signal = buy_signal(now=T0)
        win = window(s0="2000.00")
        ticks = [tick(2000.0, T0 + timedelta(seconds=i)) for i in range(0, 90)]
        books = BookHistory([book(now=T0 + timedelta(seconds=i)) for i in range(0, 90)])
        markouts = compute_markouts(
            signal, win, ticks, books, [5.0, 15.0, 30.0, 60.0], model_cfg(), vol_cfg()
        )
        assert [m.horizon_s for m in markouts] == [5.0, 15.0, 30.0, 60.0]
        for m in markouts:
            assert m.at == T0 + timedelta(seconds=m.horizon_s)

    def test_price_rising_after_buy_gives_positive_edge_markout(self):
        """Price rises well past S0 after a YES buy at 0.50 -> the market
        moving toward our fair-value view should read as a positive markout."""
        signal = buy_signal(now=T0, side=Side.YES, limit="0.50")
        win = window(s0="2000.00")
        ticks = [tick(2000.0, T0 + timedelta(seconds=i)) for i in range(0, 10)]
        ticks += [tick(2015.0, T0 + timedelta(seconds=i)) for i in range(10, 90)]
        rising_book = book(yes_bid="0.60", yes_ask="0.62")
        books = BookHistory(
            [
                rising_book.model_copy(update={"receive_time": T0 + timedelta(seconds=i)})
                for i in range(0, 90)
            ]
        )
        markouts = compute_markouts(signal, win, ticks, books, [30.0], model_cfg(), vol_cfg())
        assert markouts[0].edge_markout is not None
        assert markouts[0].edge_markout > 0

    def test_price_falling_after_buy_gives_negative_edge_markout(self):
        signal = buy_signal(now=T0, side=Side.YES, limit="0.50")
        win = window(s0="2000.00")
        ticks = [tick(2000.0, T0 + timedelta(seconds=i)) for i in range(0, 90)]
        falling_book = book(yes_bid="0.30", yes_ask="0.32")
        books = BookHistory(
            [
                falling_book.model_copy(update={"receive_time": T0 + timedelta(seconds=i)})
                for i in range(0, 90)
            ]
        )
        markouts = compute_markouts(signal, win, ticks, books, [30.0], model_cfg(), vol_cfg())
        assert markouts[0].edge_markout is not None
        assert markouts[0].edge_markout < 0

    def test_sell_signal_edge_markout_is_direction_flipped(self):
        """For a SELL, a positive markout should mean price kept moving in
        the direction that validates the sell (falling, for a YES sell) —
        the opposite sign convention from a BUY on the same book move."""
        buy = buy_signal(now=T0, side=Side.YES, limit="0.50")
        sell = sell_signal(now=T0, side=Side.YES, limit="0.50")
        win = window(s0="2000.00")
        ticks = [tick(2000.0, T0 + timedelta(seconds=i)) for i in range(0, 90)]
        falling_book = book(yes_bid="0.30", yes_ask="0.32")
        books = BookHistory(
            [
                falling_book.model_copy(update={"receive_time": T0 + timedelta(seconds=i)})
                for i in range(0, 90)
            ]
        )
        buy_markouts = compute_markouts(buy, win, ticks, books, [30.0], model_cfg(), vol_cfg())
        sell_markouts = compute_markouts(sell, win, ticks, books, [30.0], model_cfg(), vol_cfg())
        assert buy_markouts[0].edge_markout < 0
        assert sell_markouts[0].edge_markout > 0

    def test_no_data_beyond_window_close_gives_none_fields(self):
        signal = buy_signal(now=T0)
        win = window(close_in_s=10.0, s0="2000.00")
        ticks = [tick(2000.0, T0 + timedelta(seconds=i)) for i in range(0, 10)]
        books = BookHistory([book(now=T0 + timedelta(seconds=i)) for i in range(0, 10)])
        # 60s horizon is long after window close and after all recorded data.
        markouts = compute_markouts(signal, win, ticks, books, [60.0], model_cfg(), vol_cfg())
        assert markouts[0].market_mid is None
        assert markouts[0].edge_markout is None

    def test_market_mid_is_average_of_bid_ask_for_signal_side(self):
        signal = buy_signal(now=T0, side=Side.YES)
        win = window(s0="2000.00")
        ticks = [tick(2000.0, T0 + timedelta(seconds=i)) for i in range(0, 40)]
        bk = book(yes_bid="0.40", yes_ask="0.44")
        books = BookHistory(
            [
                bk.model_copy(update={"receive_time": T0 + timedelta(seconds=i)})
                for i in range(0, 40)
            ]
        )
        markouts = compute_markouts(signal, win, ticks, books, [10.0], model_cfg(), vol_cfg())
        assert markouts[0].market_mid == 0.42
        assert markouts[0].market_bid == 0.40
        assert markouts[0].market_ask == 0.44
