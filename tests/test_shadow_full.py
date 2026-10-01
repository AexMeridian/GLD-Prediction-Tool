from datetime import UTC, datetime, timedelta
from decimal import Decimal

from gold_edge.config import FeesConfig, ModelConfig
from gold_edge.learning.blend_shadow import PriceState, make_artifact
from gold_edge.learning.shadow_full import (
    FullShadowEngine,
    FullShadowStore,
    FullShadowWindow,
    market_to_book_snapshot,
)
from tests.test_state_machine import engine_cfg

T0 = datetime(2026, 9, 25, 5, 0, tzinfo=UTC)
MODEL = ModelConfig(min_fair_value=0.01, max_fair_value=0.99)
FEES = FeesConfig(base_rate=0.07, fee_multiplier=1.0, maker_fees_enabled=True)


def artifact(weights=(0.0, 1.0, 5.0)):
    return make_artifact(weights, 900.0, 100, T0)


def warmed_prices(start_price: float = 4000.0, minutes: int = 40) -> PriceState:
    ps = PriceState(900.0)
    for i in range(minutes):
        t = T0 - timedelta(minutes=minutes - i)
        ps.on_tick(start_price, t)
        ps.advance(t)
    return ps


def aggressive_cfg(**overrides):
    # persist_s=0/cooldown_s=0 let entries/exits fire on the very step that
    # qualifies (no multi-minute warmup needed in a test); exit_cutoff_s
    # bigger than the window forces a time-cutoff exit the moment a position
    # is no longer deep ITM (hold_to_settlement_when_itm only suppresses it
    # above fair=0.90), without needing the market quote itself to move.
    base = dict(persist_s=0.0, cooldown_s=0.0, max_round_trips=10, exit_cutoff_s=1000.0)
    base.update(overrides)
    return engine_cfg(**base)


class TestMarketToBookSnapshot:
    def test_parses_quote_and_mirrors_no_sizes(self):
        m = {
            "ticker": "W", "yes_bid_dollars": "0.49", "yes_ask_dollars": "0.51",
            "yes_bid_size_fp": "80", "yes_ask_size_fp": "20",
        }
        q = market_to_book_snapshot(m, T0)
        assert q is not None
        assert q.yes_bid_size == Decimal("80") and q.yes_ask_size == Decimal("20")
        # Buying NO at p is selling YES at 1-p -> sizes mirror.
        assert q.no_bid_size == Decimal("20") and q.no_ask_size == Decimal("80")
        assert q.no_bid == Decimal("0.49") and q.no_ask == Decimal("0.51")

    def test_missing_sizes_default_to_zero_not_none(self):
        m = {"ticker": "W", "yes_bid_dollars": "0.49", "yes_ask_dollars": "0.51"}
        q = market_to_book_snapshot(m, T0)
        assert q is not None
        assert q.yes_bid_size == Decimal("0") and q.no_ask_size == Decimal("0")

    def test_rejects_crossed_or_too_wide_books(self):
        assert market_to_book_snapshot(
            {"yes_bid_dollars": "0.60", "yes_ask_dollars": "0.30"}, T0
        ) is None
        assert market_to_book_snapshot(
            {"yes_bid_dollars": "0.10", "yes_ask_dollars": "0.80"}, T0
        ) is None
        assert market_to_book_snapshot({}, T0) is None


def quote(yes_bid="0.49", yes_ask="0.51"):
    return market_to_book_snapshot(
        {"yes_bid_dollars": yes_bid, "yes_ask_dollars": yes_ask}, T0
    )


class TestFullShadowEngine:
    def make(self, tmp_path, weights=(0.0, 1.0, 5.0), cfg=None):
        ps = warmed_prices()
        store = FullShadowStore(tmp_path / "f.sqlite")
        eng = FullShadowEngine(
            artifact(weights), MODEL, FEES, cfg or aggressive_cfg(), 0.5, ps, store
        )
        return ps, store, eng

    def test_buy_then_sell_records_one_closed_round_trip(self, tmp_path):
        ps, store, eng = self.make(tmp_path)
        w = FullShadowWindow("W", T0, T0 + timedelta(minutes=15))
        eng.on_boundary(w, T0, quote())  # minute 0: just sets s0_proxy

        # Price jumps hard above S0 -> model says YES strongly -> should enter.
        b1 = T0 + timedelta(minutes=1)
        ps.on_tick(4020.0, b1 - timedelta(seconds=5))
        ps.advance(b1)
        msgs1 = eng.on_boundary(w, b1, quote())
        assert any("ENTER YES" in m for m in msgs1), msgs1
        assert "W" in eng.open_rows

        # Price reverts back toward S0 -> fair YES drops back near 0.5 (no
        # longer deep ITM) -> the aggressive cfg's time-cutoff exit fires.
        b2 = T0 + timedelta(minutes=2)
        ps.on_tick(4000.0, b2 - timedelta(seconds=5))
        ps.advance(b2)
        msgs2 = eng.on_boundary(w, b2, quote())
        assert any("EXIT YES" in m for m in msgs2), msgs2
        assert "W" not in eng.open_rows

        rows = store.conn.execute(
            "SELECT side, entry_minute, exit_minute, pnl FROM shadow_full_round_trips"
        ).fetchall()
        assert len(rows) == 1
        side, entry_minute, exit_minute, pnl = rows[0]
        assert side == "YES" and entry_minute == 1 and exit_minute == 2
        assert pnl is not None

    def test_multiple_round_trips_in_one_window(self, tmp_path):
        ps, store, eng = self.make(tmp_path)
        w = FullShadowWindow("W", T0, T0 + timedelta(minutes=15))
        eng.on_boundary(w, T0, quote())

        prices_by_minute = {1: 4020.0, 2: 4000.0, 3: 4020.0, 4: 4000.0}
        for minute, price in prices_by_minute.items():
            b = T0 + timedelta(minutes=minute)
            ps.on_tick(price, b - timedelta(seconds=5))
            ps.advance(b)
            eng.on_boundary(w, b, quote())

        n_round_trips = store.conn.execute(
            "SELECT COUNT(*) FROM shadow_full_round_trips WHERE pnl IS NOT NULL"
        ).fetchone()[0]
        assert n_round_trips >= 2, "aggressive cfg + an oscillating price should yield >1 trip"

    def test_settle_if_still_open_reconciles_an_open_position(self, tmp_path):
        ps, store, eng = self.make(tmp_path)
        w = FullShadowWindow("W", T0, T0 + timedelta(minutes=15))
        eng.on_boundary(w, T0, quote())
        b1 = T0 + timedelta(minutes=1)
        ps.on_tick(4020.0, b1 - timedelta(seconds=5))
        ps.advance(b1)
        eng.on_boundary(w, b1, quote())
        assert "W" in eng.open_rows

        msg = eng.settle_if_still_open("W", "yes")
        assert msg is not None and "SETTLE" in msg
        assert "W" not in eng.open_rows
        row = store.conn.execute(
            "SELECT exit_reason, pnl FROM shadow_full_round_trips WHERE ticker='W'"
        ).fetchone()
        assert row[0] == "settled:yes" and row[1] is not None

    def test_settle_if_still_open_is_a_noop_when_nothing_is_open(self, tmp_path):
        _, _, eng = self.make(tmp_path)
        assert eng.settle_if_still_open("NOPE", "yes") is None


class TestFullShadowStore:
    def test_open_then_close_round_trip(self, tmp_path):
        store = FullShadowStore(tmp_path / "s.sqlite")
        row_id = store.open_round_trip("W", "YES", 1, 0.55, T0.isoformat(), "hash")
        assert store.open_row_ids() == {"W": row_id}
        store.close_round_trip(row_id, 3, 0.60, T0.isoformat(), "converged", 0.05)
        assert store.open_row_ids() == {}
        row = store.conn.execute(
            "SELECT exit_minute, exit_price, exit_reason, pnl FROM shadow_full_round_trips "
            "WHERE id=?",
            (row_id,),
        ).fetchone()
        assert row == (3, 0.60, "converged", 0.05)
