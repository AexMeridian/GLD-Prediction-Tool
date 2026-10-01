from datetime import UTC, datetime, timedelta
from decimal import Decimal

from gold_edge.config import FeesConfig, ModelConfig
from gold_edge.learning.blend import blend_probability
from gold_edge.learning.blend_shadow import (
    BlendArtifact,
    PriceState,
    ShadowEngine,
    ShadowStore,
    ShadowWindow,
    make_artifact,
    parse_market_quote,
)
from gold_edge.learning.calibrator import IsotonicCalibrator
from gold_edge.learning.market_study import BookQuote, best_entry

T0 = datetime(2026, 9, 25, 5, 0, tzinfo=UTC)
MODEL = ModelConfig(min_fair_value=0.01, max_fair_value=0.99)
FEES = FeesConfig(base_rate=0.07, fee_multiplier=1.0, maker_fees_enabled=True)


def warmed_prices(start_price: float = 4000.0, minutes: int = 40) -> PriceState:
    ps = PriceState(900.0)
    for i in range(minutes):
        t = T0 - timedelta(minutes=minutes - i)
        ps.on_tick(start_price + (i % 3) * 0.5, t)
        ps.advance(t)
    return ps


def quote(yes_bid="0.50", yes_ask="0.52", t=T0, yes_bid_size=None, yes_ask_size=None) -> BookQuote:
    yb, ya = Decimal(yes_bid), Decimal(yes_ask)
    return BookQuote(
        yb, ya, 1 - ya, 1 - yb, t,
        yes_bid_size=Decimal(yes_bid_size) if yes_bid_size is not None else None,
        yes_ask_size=Decimal(yes_ask_size) if yes_ask_size is not None else None,
    )


def artifact(weights=(0.0, 1.0, 0.0)) -> BlendArtifact:
    return make_artifact(weights, 900.0, 100, T0)


class TestBestEntry:
    def test_buys_yes_when_fair_far_above_ask(self):
        got = best_entry(0.90, Decimal("0.52"), Decimal("0.50"), FEES, 0.05)
        assert got is not None and got[1] == "yes"

    def test_buys_no_when_fair_far_below(self):
        got = best_entry(0.10, Decimal("0.52"), Decimal("0.50"), FEES, 0.05)
        assert got is not None and got[1] == "no"

    def test_nothing_below_threshold(self):
        assert best_entry(0.55, Decimal("0.52"), Decimal("0.50"), FEES, 0.05) is None


class TestPriceState:
    def test_price_at_uses_last_trade_at_or_before_boundary(self):
        ps = PriceState(900.0)
        ps.on_tick(100.0, T0)
        ps.on_tick(101.0, T0 + timedelta(seconds=90))
        assert ps.price_at(T0 + timedelta(seconds=60)) == 100.0
        assert ps.price_at(T0 + timedelta(seconds=120)) == 101.0
        assert ps.price_at(T0 - timedelta(seconds=1)) is None

    def test_stale_price_is_refused(self):
        ps = PriceState(900.0)
        ps.on_tick(100.0, T0)
        assert ps.price_at(T0 + timedelta(minutes=11)) is None

    def test_sigma_needs_warmup(self):
        ps = PriceState(900.0)
        assert ps.sigma is None
        assert warmed_prices().sigma is not None

    def test_long_break_restarts_the_tracker(self):
        ps = warmed_prices()
        ps.on_tick(4001.0, T0 + timedelta(minutes=20))
        ps.advance(T0 + timedelta(minutes=20))
        assert ps.sigma is None


def test_parse_market_quote_validates():
    good = {"yes_bid_dollars": "0.4900", "yes_ask_dollars": "0.5100"}
    q = parse_market_quote(good, T0)
    assert q is not None and q.no_ask == Decimal("0.5100")
    assert q.yes_bid_size is None and q.yes_ask_size is None
    empty = {"yes_bid_dollars": "0.0000", "yes_ask_dollars": "1.0000"}
    assert parse_market_quote(empty, T0) is None
    assert parse_market_quote({"yes_bid_dollars": "0.30", "yes_ask_dollars": "0.60"}, T0) is None
    assert parse_market_quote({}, T0) is None


def test_parse_market_quote_captures_top_of_book_sizes():
    m = {
        "yes_bid_dollars": "0.49", "yes_ask_dollars": "0.51",
        "yes_bid_size_fp": "120.0", "yes_ask_size_fp": "40.0",
    }
    q = parse_market_quote(m, T0)
    assert q is not None
    assert q.yes_bid_size == Decimal("120.0") and q.yes_ask_size == Decimal("40.0")
    assert abs(q.size_imbalance - 0.5) < 1e-9  # (120-40)/(120+40)


def test_size_imbalance_is_none_without_sizes():
    q = BookQuote(Decimal("0.49"), Decimal("0.51"), Decimal("0.49"), Decimal("0.51"), T0)
    assert q.size_imbalance is None


def test_artifact_round_trip(tmp_path):
    a = artifact((0.1, 0.9, 0.2))
    a.save(tmp_path / "m.json")
    b = BlendArtifact.load(tmp_path / "m.json")
    assert b == a and len(a.version_hash) == 10


def test_artifact_round_trip_with_calibrator(tmp_path):
    cal = IsotonicCalibrator(x_thresholds=[0.1, 0.5, 0.9], y_values=[0.2, 0.5, 0.8])
    a = make_artifact((0.1, 0.9, 0.2), 900.0, 100, T0, calibrator=cal)
    a.save(tmp_path / "m.json")
    b = BlendArtifact.load(tmp_path / "m.json")
    assert b == a
    assert b.calibrator == cal


def test_calibrator_changes_the_version_hash():
    cal = IsotonicCalibrator(x_thresholds=[0.1, 0.9], y_values=[0.2, 0.8])
    without = make_artifact((0.0, 1.0, 0.0), 900.0, 100, T0)
    with_cal = make_artifact((0.0, 1.0, 0.0), 900.0, 100, T0, calibrator=cal)
    assert without.version_hash != with_cal.version_hash


class TestShadowEngine:
    def make(self, tmp_path, weights=(0.0, 1.0, 5.0)):
        ps = warmed_prices()
        store = ShadowStore(tmp_path / "s.sqlite")
        return ps, store, ShadowEngine(artifact(weights), MODEL, FEES, 0.05, ps, store)

    def test_open_boundary_only_records_s0(self, tmp_path):
        ps, store, eng = self.make(tmp_path)
        ps.on_tick(4000.0, T0)
        w = ShadowWindow("W", T0, T0 + timedelta(minutes=15))
        assert eng.on_boundary(w, T0, quote()) is None
        assert w.s0_proxy == 4000.0
        assert store.conn.execute("select count(*) from shadow_observations").fetchone()[0] == 0

    def test_trades_once_per_window_when_model_disagrees_strongly(self, tmp_path):
        ps, store, eng = self.make(tmp_path)
        ps.on_tick(4000.0, T0)
        w = ShadowWindow("W", T0, T0 + timedelta(minutes=15))
        eng.on_boundary(w, T0, quote())
        msgs = []
        for m in range(1, 13):  # advance every minute like the live loop does
            b = T0 + timedelta(minutes=m)
            # price jumps well above S0 from minute 10 while the market sits at 50/52
            ps.on_tick(4010.0 if m >= 10 else 4000.0, b - timedelta(seconds=5))
            ps.advance(b)
            msgs.append(eng.on_boundary(w, b, quote(t=b)))
        assert sum(1 for x in msgs if x) == 1
        assert store.conn.execute("select count(*) from shadow_trades").fetchone()[0] == 1
        assert store.conn.execute("select count(*) from shadow_observations").fetchone()[0] == 12
        side = store.conn.execute("select side from shadow_trades").fetchone()[0]
        assert side == "yes"

    def test_no_trade_when_blend_just_trusts_the_market(self, tmp_path):
        ps, store, eng = self.make(tmp_path, weights=(0.0, 1.0, 0.0))
        ps.on_tick(4000.0, T0)
        w = ShadowWindow("W", T0, T0 + timedelta(minutes=15))
        eng.on_boundary(w, T0, quote())
        b = T0 + timedelta(minutes=8)
        ps.on_tick(4010.0, b - timedelta(seconds=5))
        ps.advance(b)
        assert eng.on_boundary(w, b, quote(t=b)) is None

    def test_calibrator_is_applied_to_the_recorded_blend_probability(self, tmp_path):
        # Market-trusting weights (0, 1, 0) alone would record blend_p == the
        # market mid (0.50); a calibrator that maps everything near 0.5 up to
        # ~0.9 should show up in what gets recorded, proving the shadow test
        # exercises the exact rule a promoted artifact would run live.
        ps = warmed_prices()
        store = ShadowStore(tmp_path / "s.sqlite")
        cal = IsotonicCalibrator(x_thresholds=[0.0, 1.0], y_values=[0.9, 0.9])
        art = make_artifact((0.0, 1.0, 0.0), 900.0, 100, T0, calibrator=cal)
        eng = ShadowEngine(art, MODEL, FEES, 0.05, ps, store)
        ps.on_tick(4000.0, T0)
        w = ShadowWindow("W", T0, T0 + timedelta(minutes=15))
        eng.on_boundary(w, T0, quote())
        b = T0 + timedelta(minutes=1)
        ps.on_tick(4000.0, b - timedelta(seconds=5))
        ps.advance(b)
        eng.on_boundary(w, b, quote(t=b))
        blend_p = store.conn.execute(
            "select blend_p from shadow_observations where minute=1"
        ).fetchone()[0]
        assert abs(blend_p - 0.9) < 1e-6

    def test_book_sizes_are_recorded_when_the_quote_carries_them(self, tmp_path):
        ps, store, eng = self.make(tmp_path, weights=(0.0, 1.0, 0.0))
        ps.on_tick(4000.0, T0)
        w = ShadowWindow("W", T0, T0 + timedelta(minutes=15))
        eng.on_boundary(w, T0, quote())
        b = T0 + timedelta(minutes=1)
        ps.on_tick(4000.0, b - timedelta(seconds=5))
        ps.advance(b)
        eng.on_boundary(w, b, quote(t=b, yes_bid_size="80", yes_ask_size="20"))
        row = store.conn.execute(
            "select yes_bid_size, yes_ask_size from shadow_observations where minute=1"
        ).fetchone()
        assert row == (80.0, 20.0)

    def test_skips_when_quote_or_price_missing(self, tmp_path):
        ps, store, eng = self.make(tmp_path)
        ps.on_tick(4000.0, T0)
        w = ShadowWindow("W", T0, T0 + timedelta(minutes=15))
        eng.on_boundary(w, T0, quote())
        b = T0 + timedelta(minutes=5)
        ps.advance(b)
        assert eng.on_boundary(w, b, None) is None
        assert store.conn.execute("select count(*) from shadow_observations").fetchone()[0] == 0


def test_shadow_store_migrates_a_pre_existing_db_missing_size_columns(tmp_path):
    import sqlite3

    path = tmp_path / "old.sqlite"
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE shadow_observations (ticker TEXT, minute INTEGER, ts TEXT, paxg REAL, "
        "s0_proxy REAL, sigma REAL, fair_yes REAL, yes_bid REAL, yes_ask REAL, blend_p REAL, "
        "model_hash TEXT, PRIMARY KEY (ticker, minute))"
    )
    conn.execute(
        "INSERT INTO shadow_observations VALUES ('W',1,'t',1,2,3,4,5,6,7,'h')"
    )
    conn.commit()
    conn.close()

    store = ShadowStore(path)  # must not raise, and must not lose the existing row
    cols = {r[1] for r in store.conn.execute("PRAGMA table_info(shadow_observations)")}
    assert {"yes_bid_size", "yes_ask_size"} <= cols
    row = store.conn.execute("SELECT ticker, yes_bid_size FROM shadow_observations").fetchone()
    assert row == ("W", None)


class TestShadowStoreSettle:
    def row(self, side="yes", ask=0.52, fee=0.02):
        return ("W", 5, T0.isoformat(), side, ask, fee, 0.7, 0.8, 0.1, "h")

    def test_winner_and_loser_pnl(self, tmp_path):
        s = ShadowStore(tmp_path / "s.sqlite")
        s.record_trade(self.row("yes"))
        assert abs(s.settle("W", "yes") - (1 - 0.52 - 0.02)) < 1e-9
        s2 = ShadowStore(tmp_path / "s2.sqlite")
        s2.record_trade(self.row("no", ask=0.5, fee=0.02))
        assert abs(s2.settle("W", "yes") - (0 - 0.5 - 0.02)) < 1e-9

    def test_settle_is_idempotent_and_ignores_unknown(self, tmp_path):
        s = ShadowStore(tmp_path / "s.sqlite")
        s.record_trade(self.row())
        assert s.settle("W", "yes") is not None
        assert s.settle("W", "no") is None
        assert s.settle("nope", "yes") is None
        assert s.unsettled_tickers() == []


def test_blend_probability_identity_weights_return_the_market():
    assert abs(blend_probability((0.0, 1.0, 0.0), 0.7, 0.2) - 0.7) < 1e-9
