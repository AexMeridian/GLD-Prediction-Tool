from datetime import UTC, datetime, timedelta
from decimal import Decimal

from gold_edge.engine.signals import build_signal, is_expired, price_moved_past_limit
from gold_edge.models import Action, BookSnapshot, Side


def _book(**overrides) -> BookSnapshot:
    defaults = dict(
        window_ticker="KXGOLD15M-TEST",
        yes_bid=Decimal("0.37"),
        yes_ask=Decimal("0.39"),
        yes_bid_size=Decimal("100"),
        yes_ask_size=Decimal("100"),
        no_bid=Decimal("0.61"),
        no_ask=Decimal("0.63"),
        no_bid_size=Decimal("100"),
        no_ask_size=Decimal("100"),
        receive_time=datetime(2026, 1, 1, tzinfo=UTC),
    )
    defaults.update(overrides)
    return BookSnapshot(**defaults)


def test_buy_signal_limit_is_the_ask():
    now = datetime(2026, 1, 1, tzinfo=UTC)
    signal = build_signal(
        action=Action.BUY,
        side=Side.YES,
        window_ticker="KXGOLD15M-TEST",
        book=_book(),
        fair=0.45,
        size=Decimal(10),
        edge_after_costs=0.06,
        reason="enter_edge",
        now=now,
        ttl_s=6.0,
    )
    assert signal.limit_price == Decimal("0.39")
    assert signal.market_price == Decimal("0.39")
    assert signal.expires_at == now + timedelta(seconds=6.0)


def test_sell_signal_limit_is_the_bid():
    now = datetime(2026, 1, 1, tzinfo=UTC)
    signal = build_signal(
        action=Action.SELL,
        side=Side.NO,
        window_ticker="KXGOLD15M-TEST",
        book=_book(),
        fair=0.55,
        size=Decimal(10),
        edge_after_costs=0.02,
        reason="converged",
        now=now,
        ttl_s=6.0,
    )
    assert signal.limit_price == Decimal("0.61")


def test_signal_not_expired_before_ttl():
    now = datetime(2026, 1, 1, tzinfo=UTC)
    signal = build_signal(
        action=Action.BUY,
        side=Side.YES,
        window_ticker="t",
        book=_book(),
        fair=0.5,
        size=Decimal(1),
        edge_after_costs=0.01,
        reason="x",
        now=now,
        ttl_s=6.0,
    )
    assert not is_expired(signal, now + timedelta(seconds=5.9))


def test_signal_expired_after_ttl():
    now = datetime(2026, 1, 1, tzinfo=UTC)
    signal = build_signal(
        action=Action.BUY,
        side=Side.YES,
        window_ticker="t",
        book=_book(),
        fair=0.5,
        size=Decimal(1),
        edge_after_costs=0.01,
        reason="x",
        now=now,
        ttl_s=6.0,
    )
    assert is_expired(signal, now + timedelta(seconds=6.0))
    assert is_expired(signal, now + timedelta(seconds=100))


def test_buy_signal_price_moved_past_limit_when_ask_rises():
    now = datetime(2026, 1, 1, tzinfo=UTC)
    signal = build_signal(
        action=Action.BUY,
        side=Side.YES,
        window_ticker="t",
        book=_book(yes_ask=Decimal("0.39")),
        fair=0.5,
        size=Decimal(1),
        edge_after_costs=0.05,
        reason="x",
        now=now,
        ttl_s=6.0,
    )
    worse_book = _book(yes_ask=Decimal("0.42"))
    same_book = _book(yes_ask=Decimal("0.39"))
    better_book = _book(yes_ask=Decimal("0.36"))
    assert price_moved_past_limit(signal, worse_book)
    assert not price_moved_past_limit(signal, same_book)
    assert not price_moved_past_limit(signal, better_book)


def test_sell_signal_price_moved_past_limit_when_bid_falls():
    now = datetime(2026, 1, 1, tzinfo=UTC)
    signal = build_signal(
        action=Action.SELL,
        side=Side.NO,
        window_ticker="t",
        book=_book(no_bid=Decimal("0.61")),
        fair=0.5,
        size=Decimal(1),
        edge_after_costs=0.02,
        reason="x",
        now=now,
        ttl_s=6.0,
    )
    worse_book = _book(no_bid=Decimal("0.58"))
    better_book = _book(no_bid=Decimal("0.64"))
    assert price_moved_past_limit(signal, worse_book)
    assert not price_moved_past_limit(signal, better_book)
