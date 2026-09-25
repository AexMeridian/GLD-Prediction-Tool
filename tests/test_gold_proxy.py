from datetime import UTC, datetime

from gold_edge.feeds.gold_proxy import SOURCE, parse_ticker_message


def _msg(**overrides):
    base = {
        "type": "ticker",
        "sequence": 123,
        "product_id": "PAXG-USD",
        "price": "4373.55",
        "best_bid": "4372.41",
        "best_ask": "4373.55",
        "time": "2026-09-18T21:59:33.242569Z",
        "trade_id": 999,
    }
    base.update(overrides)
    return base


def test_parses_a_ticker_message():
    tick = parse_ticker_message(_msg(), "PAXG-USD")
    assert tick is not None
    assert tick.symbol == "PAXG-USD"
    assert tick.price == 4373.55
    assert tick.source == SOURCE
    assert tick.publish_time == datetime(2026, 9, 18, 21, 59, 33, 242569, tzinfo=UTC)
    assert tick.conf == abs(4373.55 - 4372.41) / 2.0


def test_ignores_message_for_a_different_product():
    assert parse_ticker_message(_msg(product_id="BTC-USD"), "PAXG-USD") is None


def test_ignores_non_ticker_message_types():
    assert parse_ticker_message(_msg(type="subscriptions"), "PAXG-USD") is None
    assert parse_ticker_message(_msg(type="heartbeat"), "PAXG-USD") is None


def test_missing_price_or_time_returns_none():
    msg = _msg()
    del msg["price"]
    assert parse_ticker_message(msg, "PAXG-USD") is None

    msg2 = _msg()
    del msg2["time"]
    assert parse_ticker_message(msg2, "PAXG-USD") is None


def test_missing_bid_ask_falls_back_to_zero_confidence():
    msg = _msg()
    del msg["best_bid"]
    del msg["best_ask"]
    tick = parse_ticker_message(msg, "PAXG-USD")
    assert tick is not None
    assert tick.conf == 0.0
