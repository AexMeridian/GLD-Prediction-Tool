from datetime import UTC, datetime, timedelta

from gold_edge.learning.historical_gold import PriceBar
from gold_edge.learning.history import (
    densify,
    merge_consensus,
    open_history,
    parse_candle,
    parse_coinbase_candles,
    parse_kraken_candles,
    parse_market,
)

T0 = datetime(2026, 9, 21, 14, 0, tzinfo=UTC)


def market(**over):
    base = {
        "ticker": "KXGOLD15M-X",
        "open_time": "2026-09-25T03:00:00Z",
        "close_time": "2026-09-25T03:15:00Z",
        "floor_strike": 4283.79,
        "expiration_value": "4282.89",
        "result": "no",
    }
    base.update(over)
    return base


def test_parse_market_reads_true_s0_and_settlement_value():
    w = parse_market(market())
    assert w is not None
    assert w.s0 == 4283.79 and w.settle_value == 4282.89 and w.result == "no"
    assert w.close_time - w.open_time == timedelta(minutes=15)


def test_unsettled_or_void_markets_are_skipped():
    assert parse_market(market(result="")) is None
    assert parse_market(market(result="void")) is None


def test_parse_candle_reads_bid_ask_and_tolerates_missing_sides():
    c = parse_candle(
        {
            "end_period_ts": 1790305260,
            "volume_fp": "19051.45",
            "yes_bid": {"open_dollars": "0.0110", "close_dollars": "0.4800"},
            "yes_ask": {"open_dollars": "1.0000", "close_dollars": "0.4900"},
        }
    )
    assert (c.yes_bid_close, c.yes_ask_close, c.yes_ask_open) == (0.48, 0.49, 1.0)
    empty = parse_candle({"end_period_ts": 5})
    assert empty.yes_bid_close is None and empty.volume == 0.0


def test_parse_coinbase_candles_uses_close_column():
    bars = parse_coinbase_candles([[1789999800, 4335.4, 4335.6, 4335.5, 4335.65, 0.007]])
    assert bars[0].close == 4335.65
    assert bars[0].timestamp == datetime.fromtimestamp(1789999800, tz=UTC)


def test_parse_kraken_candles_drops_zero_volume_carry_forward_rows():
    rows = [
        # [time, open, high, low, close, vwap, volume, count]
        [1789999800, "4335.4", "4335.6", "4335.5", "4335.65", "4335.5", "0.00930819", 4],
        [1790000400, "4335.65", "4335.65", "4335.65", "4335.65", "0.00", "0.00000000", 0],
    ]
    bars = parse_kraken_candles(rows)
    assert len(bars) == 1
    assert bars[0].close == 4335.65
    assert bars[0].timestamp == datetime.fromtimestamp(1789999800, tz=UTC)


def test_merge_consensus_takes_the_median_per_minute():
    a = [PriceBar(T0, 100.0), PriceBar(T0 + timedelta(minutes=1), 101.0)]
    b = [PriceBar(T0, 102.0)]  # only covers minute 0
    out = merge_consensus(a, b)
    by_ts = {b.timestamp: b.close for b in out}
    assert by_ts[T0] == 101.0  # median(100, 102)
    assert by_ts[T0 + timedelta(minutes=1)] == 101.0  # only source -> itself


def test_densify_forward_fills_short_gaps_only():
    bars = [
        PriceBar(T0, 100.0),
        PriceBar(T0 + timedelta(minutes=3), 101.0),
        PriceBar(T0 + timedelta(minutes=30), 102.0),
    ]
    out = densify(bars, max_gap_min=10)
    minutes = [int((b.timestamp - T0).total_seconds() // 60) for b in out]
    assert minutes == [0, 1, 2, 3, 30]
    assert [b.close for b in out[:4]] == [100.0, 100.0, 100.0, 101.0]


def test_open_history_creates_schema(tmp_path):
    conn = open_history(tmp_path / "sub" / "h.sqlite")
    names = {r[0] for r in conn.execute("select name from sqlite_master where type='table'")}
    assert {"kalshi_windows", "kalshi_candles", "paxg_1m", "kraken_paxg_1m"} <= names
