import random
import sqlite3
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from gold_edge.backtest.replay import RecordedWindow, load_recorded_data, replay_all, run_sweep
from gold_edge.config import BacktestConfig, ModelConfig, VolatilityConfig
from gold_edge.models import Tick
from gold_edge.recorder import Recorder
from tests.test_state_machine import book, engine_cfg, fees_cfg, window

T0 = datetime(2026, 1, 1, tzinfo=UTC)


def bt_cfg(**overrides):
    defaults = dict(human_delay_min_s=1.0, human_delay_max_s=1.0)  # fixed delay for determinism
    defaults.update(overrides)
    return BacktestConfig(**defaults)


def vol_cfg(**overrides):
    defaults = dict(
        ewma_half_life_s=60.0,
        short_horizon_s=60.0,
        min_sigma_per_minute=0.0005,
        vol_spike_limit=0.5,
    )
    defaults.update(overrides)
    return VolatilityConfig(**defaults)


def model_cfg(**overrides):
    defaults = dict(min_fair_value=0.01, max_fair_value=0.99)
    defaults.update(overrides)
    return ModelConfig(**defaults)


def tick(price: float, t: datetime) -> Tick:
    return Tick(
        symbol="Metal.XAU/USD", price=price, conf=0.1, expo=-2, publish_time=t, receive_time=t
    )


def make_window(ticker="KXGOLD15M-A", open_offset=0.0, close_offset=600.0, s0="2000.00"):
    return window(close_in_s=close_offset, s0=s0).model_copy(
        update={
            "ticker": ticker,
            "open_time": T0 + timedelta(seconds=open_offset),
            "close_time": T0 + timedelta(seconds=close_offset),
        }
    )


class TestReplayNoOpportunity:
    def test_flat_market_generates_no_signals_and_no_pnl(self):
        win = make_window()
        ticks = [tick(2000.0, T0 + timedelta(seconds=i)) for i in range(0, 20)]
        books = [book(now=T0 + timedelta(seconds=i)) for i in range(0, 20)]
        rw = RecordedWindow(window=win, books=books, settlement_result="yes")
        result = replay_all(
            ticks=ticks,
            windows=[rw],
            engine_cfg=engine_cfg(),
            fees_cfg=fees_cfg(),
            vol_cfg=vol_cfg(),
            model_cfg=model_cfg(),
            backtest_cfg=bt_cfg(),
            rng=random.Random(0),
        )
        assert len(result.windows) == 1
        wr = result.windows[0]
        assert wr.round_trips == []
        assert all(not sr.filled for sr in wr.signal_records) or wr.signal_records == []


class TestReplayEntryAndExit:
    def test_clear_mispricing_produces_a_filled_round_trip(self):
        """Price has jumped well above S0 and stays there — YES should be
        deeply underpriced at 0.50 relative to a fair value pushed high by
        the price move, triggering entry, then converge/overshoot exit."""
        win = make_window(close_offset=600.0, s0="2000.00")
        now_ticks = [tick(2000.0 + i * 0.5, T0 + timedelta(seconds=i)) for i in range(0, 10)]
        # Price holds well above S0 for the rest of the window.
        now_ticks += [tick(2010.0, T0 + timedelta(seconds=i)) for i in range(10, 200)]
        cheap_book = book(yes_bid="0.48", yes_ask="0.50", no_bid="0.48", no_ask="0.50")
        books = [
            cheap_book.model_copy(update={"receive_time": T0 + timedelta(seconds=i)})
            for i in range(0, 200)
        ]
        rw = RecordedWindow(window=win, books=books, settlement_result="yes")
        result = replay_all(
            ticks=now_ticks,
            windows=[rw],
            engine_cfg=engine_cfg(persist_s=0.5),
            fees_cfg=fees_cfg(),
            vol_cfg=vol_cfg(),
            model_cfg=model_cfg(),
            backtest_cfg=bt_cfg(),
            rng=random.Random(0),
        )
        wr = result.windows[0]
        assert any(sr.filled for sr in wr.signal_records), "expected at least one filled signal"


class TestReplaySettlement:
    def test_unresolved_settlement_produces_a_note_not_a_crash(self):
        win = make_window(close_offset=5.0, s0="2000.00")
        ticks = [tick(2000.0, T0 + timedelta(seconds=i)) for i in range(0, 6)]
        books = [book(now=T0 + timedelta(seconds=i)) for i in range(0, 6)]
        rw = RecordedWindow(window=win, books=books, settlement_result=None)
        result = replay_all(
            ticks=ticks,
            windows=[rw],
            engine_cfg=engine_cfg(),
            fees_cfg=fees_cfg(),
            vol_cfg=vol_cfg(),
            model_cfg=model_cfg(),
            backtest_cfg=bt_cfg(),
            rng=random.Random(0),
        )
        assert result.windows[0] is not None  # no crash


class TestRealizedPnlCarriesForward:
    def test_realized_pnl_today_persists_across_windows(self):
        win1 = make_window(ticker="KXGOLD15M-A", open_offset=0.0, close_offset=300.0)
        win2 = make_window(ticker="KXGOLD15M-B", open_offset=300.0, close_offset=600.0)
        ticks = [tick(2000.0, T0 + timedelta(seconds=i)) for i in range(0, 600)]
        books1 = [book(now=T0 + timedelta(seconds=i)) for i in range(0, 300)]
        books2 = [
            book(now=T0 + timedelta(seconds=i)).model_copy(update={"window_ticker": "KXGOLD15M-B"})
            for i in range(300, 600)
        ]
        rw1 = RecordedWindow(window=win1, books=books1, settlement_result="yes")
        rw2 = RecordedWindow(window=win2, books=books2, settlement_result="no")
        result = replay_all(
            ticks=ticks,
            windows=[rw1, rw2],
            engine_cfg=engine_cfg(),
            fees_cfg=fees_cfg(),
            vol_cfg=vol_cfg(),
            model_cfg=model_cfg(),
            backtest_cfg=bt_cfg(),
            rng=random.Random(0),
        )
        assert len(result.windows) == 2
        assert isinstance(result.windows[1].ending_realized_pnl_today, Decimal)


class TestDailyPnlReset:
    def test_realized_pnl_resets_at_utc_day_boundary(self):
        """Window 1 books a nonzero realized P&L via a real filled round
        trip. Window 2 opens ~24h later (a new UTC day) in a flat market
        that produces zero trades of its own — if the daily reset works,
        its ending total is exactly 0, not window 1's carried-forward P&L."""
        win1 = make_window(ticker="KXGOLD15M-A", close_offset=600.0, s0="2000.00")
        win1_ticks = [tick(2000.0 + i * 0.5, T0 + timedelta(seconds=i)) for i in range(0, 10)]
        win1_ticks += [tick(2010.0, T0 + timedelta(seconds=i)) for i in range(10, 200)]
        cheap_book = book(yes_bid="0.48", yes_ask="0.50", no_bid="0.48", no_ask="0.50")
        books1 = [
            cheap_book.model_copy(update={"receive_time": T0 + timedelta(seconds=i)})
            for i in range(0, 200)
        ]
        rw1 = RecordedWindow(window=win1, books=books1, settlement_result="yes")

        day_gap = timedelta(hours=24).total_seconds()
        win2 = make_window(
            ticker="KXGOLD15M-B",
            open_offset=day_gap,
            close_offset=day_gap + 300.0,
            s0="2010.00",
        )
        win2_ticks = [tick(2010.0, T0 + timedelta(seconds=day_gap + i)) for i in range(0, 300)]
        flat_book = book(yes_bid="0.48", yes_ask="0.50", no_bid="0.48", no_ask="0.50")
        books2 = [
            flat_book.model_copy(
                update={
                    "receive_time": T0 + timedelta(seconds=day_gap + i),
                    "window_ticker": "KXGOLD15M-B",
                }
            )
            for i in range(0, 300)
        ]
        rw2 = RecordedWindow(window=win2, books=books2, settlement_result="no")

        result = replay_all(
            ticks=win1_ticks + win2_ticks,
            windows=[rw1, rw2],
            engine_cfg=engine_cfg(persist_s=0.5),
            fees_cfg=fees_cfg(),
            vol_cfg=vol_cfg(),
            model_cfg=model_cfg(),
            backtest_cfg=bt_cfg(),
            rng=random.Random(0),
        )
        assert len(result.windows) == 2
        assert result.windows[0].round_trips, "expected window 1 to book a real round trip"
        assert result.windows[0].ending_realized_pnl_today != Decimal("0")
        assert result.windows[1].round_trips == []
        assert result.windows[1].ending_realized_pnl_today == Decimal("0")

    def test_same_day_windows_still_carry_pnl_forward(self):
        """Sanity check that the reset is date-gated, not unconditional —
        the existing same-day carry-forward behavior must still hold."""
        win1 = make_window(ticker="KXGOLD15M-A", close_offset=600.0, s0="2000.00")
        win1_ticks = [tick(2000.0 + i * 0.5, T0 + timedelta(seconds=i)) for i in range(0, 10)]
        win1_ticks += [tick(2010.0, T0 + timedelta(seconds=i)) for i in range(10, 200)]
        cheap_book = book(yes_bid="0.48", yes_ask="0.50", no_bid="0.48", no_ask="0.50")
        books1 = [
            cheap_book.model_copy(update={"receive_time": T0 + timedelta(seconds=i)})
            for i in range(0, 200)
        ]
        rw1 = RecordedWindow(window=win1, books=books1, settlement_result="yes")

        win2 = make_window(
            ticker="KXGOLD15M-B", open_offset=600.0, close_offset=900.0, s0="2010.00"
        )
        win2_ticks = [tick(2010.0, T0 + timedelta(seconds=i)) for i in range(600, 900)]
        flat_book = book(yes_bid="0.48", yes_ask="0.50", no_bid="0.48", no_ask="0.50")
        books2 = [
            flat_book.model_copy(
                update={"receive_time": T0 + timedelta(seconds=i), "window_ticker": "KXGOLD15M-B"}
            )
            for i in range(600, 900)
        ]
        rw2 = RecordedWindow(window=win2, books=books2, settlement_result="no")

        result = replay_all(
            ticks=win1_ticks + win2_ticks,
            windows=[rw1, rw2],
            engine_cfg=engine_cfg(persist_s=0.5),
            fees_cfg=fees_cfg(),
            vol_cfg=vol_cfg(),
            model_cfg=model_cfg(),
            backtest_cfg=bt_cfg(),
            rng=random.Random(0),
        )
        assert result.windows[0].ending_realized_pnl_today != Decimal("0")
        assert result.windows[1].round_trips == []
        assert (
            result.windows[1].ending_realized_pnl_today
            == result.windows[0].ending_realized_pnl_today
        )


class TestOnStepHook:
    def test_fires_once_per_event_with_pre_step_state(self):
        win = make_window(close_offset=20.0)
        ticks = [tick(2000.0, T0 + timedelta(seconds=i)) for i in range(0, 20)]
        books = [book(now=T0 + timedelta(seconds=i)) for i in range(0, 20)]
        rw = RecordedWindow(window=win, books=books, settlement_result="yes")

        calls = []

        def on_step(state, snap):
            calls.append((state, snap.now))

        replay_all(
            ticks=ticks,
            windows=[rw],
            engine_cfg=engine_cfg(),
            fees_cfg=fees_cfg(),
            vol_cfg=vol_cfg(),
            model_cfg=model_cfg(),
            backtest_cfg=bt_cfg(),
            rng=random.Random(0),
            on_step=on_step,
        )
        # 20 ticks + 20 books, minus the very first event (only one of
        # price/book is known yet, so step() isn't called for it).
        assert len(calls) == 39
        # Times must be non-decreasing and pre-step state starts FLAT.
        assert [t for _, t in calls] == sorted(t for _, t in calls)
        assert calls[0][0].position_state.value == "FLAT"


class TestRunSweep:
    def test_rejects_grid_larger_than_max_combinations(self):
        win = make_window()
        ticks = [tick(2000.0, T0 + timedelta(seconds=i)) for i in range(0, 5)]
        books = [book(now=T0 + timedelta(seconds=i)) for i in range(0, 5)]
        rw = RecordedWindow(window=win, books=books, settlement_result="yes")
        with pytest.raises(ValueError, match="too large"):
            run_sweep(
                train_ticks=ticks,
                train_windows=[rw],
                test_ticks=ticks,
                test_windows=[rw],
                base_engine_cfg=engine_cfg(),
                fees_cfg=fees_cfg(),
                vol_cfg=vol_cfg(),
                model_cfg=model_cfg(),
                backtest_cfg=bt_cfg(),
                param_grid={"enter_edge": [0.01, 0.02, 0.03], "stop": [0.05, 0.08]},
                max_combinations=2,
            )

    def test_evaluates_each_combination_on_both_splits(self):
        win = make_window()
        ticks = [tick(2000.0, T0 + timedelta(seconds=i)) for i in range(0, 20)]
        books = [book(now=T0 + timedelta(seconds=i)) for i in range(0, 20)]
        rw = RecordedWindow(window=win, books=books, settlement_result="yes")
        candidates = run_sweep(
            train_ticks=ticks,
            train_windows=[rw],
            test_ticks=ticks,
            test_windows=[rw],
            base_engine_cfg=engine_cfg(),
            fees_cfg=fees_cfg(),
            vol_cfg=vol_cfg(),
            model_cfg=model_cfg(),
            backtest_cfg=bt_cfg(),
            param_grid={"enter_edge": [0.02, 0.03]},
            seed=0,
        )
        assert len(candidates) == 2
        for c in candidates:
            assert "train_net_pnl" in c
            assert "test_net_pnl" in c
        # sorted by train performance, descending
        assert candidates[0]["train_net_pnl"] >= candidates[1]["train_net_pnl"]


class TestLoadRecordedData:
    def test_round_trips_through_sqlite(self, tmp_path):
        sqlite_path = tmp_path / "test.sqlite"
        recorder = Recorder(sqlite_path)
        win = make_window()
        t = tick(2000.0, T0)
        b = book(now=T0).model_copy(update={"window_ticker": win.ticker})

        import asyncio

        asyncio.run(recorder.record_window(win))
        asyncio.run(recorder.record_tick(t))
        asyncio.run(recorder.record_book_snapshot(b))
        asyncio.run(recorder.record_settlement(win.ticker, {"result": "yes"}, "yes", None))
        recorder.close()

        ticks, windows = load_recorded_data(sqlite_path)
        assert len(ticks) == 1
        assert len(windows) == 1
        assert windows[0].window.ticker == win.ticker
        assert windows[0].settlement_result == "yes"
        assert len(windows[0].books) == 1

    def test_excludes_proxy_ticks_by_default(self, tmp_path):
        """Replay must never silently blend un-basis-adjusted paxg_proxy
        ticks into the same series as real spot -- see the tick_source note
        on load_recorded_data. Only the live path (server.py) knows how to
        reconcile the two."""
        sqlite_path = tmp_path / "test.sqlite"
        recorder = Recorder(sqlite_path)
        real = tick(2000.0, T0)
        proxy = Tick(
            symbol="PAXG-USD",
            price=1995.0,
            conf=0.5,
            expo=0,
            publish_time=T0,
            receive_time=T0,
            source="paxg_proxy",
        )

        import asyncio

        asyncio.run(recorder.record_tick(real))
        asyncio.run(recorder.record_tick(proxy))
        recorder.close()

        ticks, _ = load_recorded_data(sqlite_path)
        assert len(ticks) == 1
        assert ticks[0].source == "pyth_xau"
        assert ticks[0].price == 2000.0

    def test_excludes_windows_without_known_s0(self, tmp_path):
        sqlite_path = tmp_path / "test.sqlite"
        conn = sqlite3.connect(sqlite_path)
        from gold_edge.recorder import SCHEMA

        conn.executescript(SCHEMA)
        conn.execute(
            "INSERT INTO windows (ticker, event_ticker, series_ticker, open_time, "
            "close_time, s0, status) VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("T1", "E1", "KXGOLD15M", T0.isoformat(), T0.isoformat(), None, "unopened"),
        )
        conn.commit()
        conn.close()

        ticks, windows = load_recorded_data(sqlite_path)
        assert windows == []
