from datetime import UTC, datetime
from decimal import Decimal

from gold_edge.models import BookSnapshot, Side, Window


def _book() -> BookSnapshot:
    return BookSnapshot(
        window_ticker="KXGOLD15M-TEST",
        yes_bid=Decimal("0.37"),
        yes_ask=Decimal("0.39"),
        yes_bid_size=Decimal("100"),
        yes_ask_size=Decimal("50"),
        no_bid=Decimal("0.61"),
        no_ask=Decimal("0.63"),
        no_bid_size=Decimal("80"),
        no_ask_size=Decimal("20"),
        receive_time=datetime.now(UTC),
    )


def test_book_bid_ask_and_spread():
    book = _book()
    assert book.bid(Side.YES) == Decimal("0.37")
    assert book.ask(Side.YES) == Decimal("0.39")
    assert book.bid(Side.NO) == Decimal("0.61")
    assert book.ask(Side.NO) == Decimal("0.63")
    assert book.spread(Side.YES) == Decimal("0.02")
    assert book.spread(Side.NO) == Decimal("0.02")


def test_yes_no_reciprocal_relationship_holds_in_fixture():
    # Kalshi's binary book only stores bids; asks are 1 - opposite bid.
    book = _book()
    assert book.yes_ask == Decimal("1") - book.no_bid
    assert book.no_ask == Decimal("1") - book.yes_bid


def test_window_seconds_left_floors_at_zero():
    window = Window(
        ticker="KXGOLD15M-TEST",
        event_ticker="KXGOLD15M-TESTEVT",
        series_ticker="KXGOLD15M",
        open_time=datetime(2026, 1, 1, tzinfo=UTC),
        close_time=datetime(2026, 1, 1, 0, 15, tzinfo=UTC),
        status="active",
    )
    past_close = datetime(2026, 1, 1, 0, 20, tzinfo=UTC)
    assert window.seconds_left(past_close) == 0.0
