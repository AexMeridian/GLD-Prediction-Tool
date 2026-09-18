from decimal import Decimal

from gold_edge.model.fees import maker_fee, taker_fee


def test_fee_zero_at_price_zero():
    assert taker_fee(Decimal(100), Decimal("0.00"), Decimal(1)) == Decimal("0.00")


def test_fee_zero_at_price_one():
    assert taker_fee(Decimal(100), Decimal("1.00"), Decimal(1)) == Decimal("0.00")


def test_fee_matches_known_kalshi_example():
    # 100 contracts @ 50c, M=1: 1 * 0.07 * 100 * 0.5 * 0.5 = 1.75 -> already a
    # whole cent, ceiling should not bump it up.
    assert taker_fee(Decimal(100), Decimal("0.50"), Decimal(1)) == Decimal("1.75")


def test_fee_rounds_up_to_next_cent():
    # 1 contract @ 50c: 1 * 0.07 * 1 * 0.5 * 0.5 = 0.0175 -> ceil to 0.02
    assert taker_fee(Decimal(1), Decimal("0.50"), Decimal(1)) == Decimal("0.02")


def test_fee_rounds_up_near_one_cent_price():
    # 1 contract @ 1c: 1 * 0.07 * 1 * 0.01 * 0.99 = 0.000693 -> ceil to 0.01
    fee = taker_fee(Decimal(1), Decimal("0.01"), Decimal(1))
    assert fee == Decimal("0.01")


def test_fee_rounds_up_near_ninety_nine_cent_price():
    # symmetric with the 1c case: P(1-P) is the same at 0.01 and 0.99
    fee_low = taker_fee(Decimal(1), Decimal("0.01"), Decimal(1))
    fee_high = taker_fee(Decimal(1), Decimal("0.99"), Decimal(1))
    assert fee_low == fee_high == Decimal("0.01")


def test_fee_scales_with_multiplier():
    base = taker_fee(Decimal(100), Decimal("0.50"), Decimal(1))
    doubled = taker_fee(Decimal(100), Decimal("0.50"), Decimal(2))
    assert doubled == base * 2


def test_fee_never_negative_for_valid_prices():
    for cents in range(0, 101):
        price = Decimal(cents) / Decimal(100)
        assert taker_fee(Decimal(10), price, Decimal(1)) >= Decimal("0.00")


def test_maker_fee_equals_taker_fee_when_enabled():
    contracts, price, multiplier = Decimal(50), Decimal("0.42"), Decimal(1)
    assert maker_fee(contracts, price, multiplier, maker_fees_enabled=True) == taker_fee(
        contracts, price, multiplier
    )


def test_maker_fee_is_zero_when_disabled():
    fee = maker_fee(Decimal(50), Decimal("0.42"), Decimal(1), maker_fees_enabled=False)
    assert fee == Decimal("0.00")


def test_fee_rejects_out_of_range_price():
    import pytest

    with pytest.raises(ValueError):
        taker_fee(Decimal(1), Decimal("1.01"), Decimal(1))
    with pytest.raises(ValueError):
        taker_fee(Decimal(1), Decimal("-0.01"), Decimal(1))
