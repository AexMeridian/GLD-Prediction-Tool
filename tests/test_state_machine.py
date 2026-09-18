from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from gold_edge.config import EngineConfig, FeesConfig
from gold_edge.engine.state_machine import (
    EngineState,
    MarketSnapshot,
    apply_fill,
    apply_skip,
    step,
)
from gold_edge.model.fair_value import FairValue
from gold_edge.models import (
    Action,
    BookSnapshot,
    Fill,
    PositionState,
    Side,
    SignalStatus,
    Window,
)

T0 = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)


def engine_cfg(**overrides) -> EngineConfig:
    defaults = dict(
        enter_edge=0.03,
        persist_s=1.5,
        cooldown_s=20.0,
        entry_cutoff_s=90.0,
        exit_cutoff_s=75.0,
        stale_s=3.0,
        max_spread=0.04,
        max_round_trips=6,
        daily_loss_stop=25.0,
        converge_band=0.01,
        stop=0.08,
        hold_to_settlement_when_itm=True,
        hold_to_settlement_fair_threshold=0.90,
        signal_ttl_s=6.0,
    )
    defaults.update(overrides)
    return EngineConfig(**defaults)


def fees_cfg(**overrides) -> FeesConfig:
    defaults = dict(base_rate=0.07, fee_multiplier=1.0, maker_fees_enabled=True)
    defaults.update(overrides)
    return FeesConfig(**defaults)


def window(close_in_s: float = 600.0, s0: str = "2000.00") -> Window:
    return Window(
        ticker="KXGOLD15M-TEST",
        event_ticker="KXGOLD15M-TESTEVT",
        series_ticker="KXGOLD15M",
        open_time=T0 - timedelta(seconds=300),
        close_time=T0 + timedelta(seconds=close_in_s),
        s0=Decimal(s0),
        status="active",
    )


def book(
    yes_bid="0.48", yes_ask="0.50", no_bid="0.48", no_ask="0.50", now: datetime = T0
) -> BookSnapshot:
    return BookSnapshot(
        window_ticker="KXGOLD15M-TEST",
        yes_bid=Decimal(yes_bid),
        yes_ask=Decimal(yes_ask),
        yes_bid_size=Decimal(100),
        yes_ask_size=Decimal(100),
        no_bid=Decimal(no_bid),
        no_ask=Decimal(no_ask),
        no_bid_size=Decimal(100),
        no_ask_size=Decimal(100),
        receive_time=now,
    )


def snap(
    now=T0,
    win=None,
    bk=None,
    fair=None,
    pyth_age_s=0.5,
    kalshi_age_s=0.5,
    short_horizon_sigma_per_minute=None,
) -> MarketSnapshot:
    return MarketSnapshot(
        now=now,
        window=win or window(),
        book=bk or book(),
        fair=fair or FairValue(yes=0.5, no=0.5),
        pyth_age_s=pyth_age_s,
        kalshi_age_s=kalshi_age_s,
        short_horizon_sigma_per_minute=short_horizon_sigma_per_minute,
    )


# A book/fair combo with a clean, hand-checked gap of 0.04 on YES:
# fee(0.50)=0.02, fee(0.48)=0.02, spread=0.02 -> round_trip_cost=0.06
# gap = 0.60 - 0.50 - 0.06 = 0.04 > enter_edge(0.03)
EDGE_BOOK = book(yes_bid="0.48", yes_ask="0.50", no_bid="0.30", no_ask="0.32")
EDGE_FAIR_YES = FairValue(yes=0.60, no=0.40)


def _persist_to_entry(
    state: EngineState, cfg: EngineConfig, fc: FeesConfig, win: Window
) -> EngineState:
    """Run enough steps with a qualifying gap for persistence to clear,
    without yet producing a signal (asserted along the way)."""
    result = step(
        state,
        snap(now=T0, win=win, bk=EDGE_BOOK, fair=EDGE_FAIR_YES),
        cfg,
        fc,
        vol_spike_limit=0.01,
    )
    assert result.signal is None  # not persisted yet
    return result.state


class TestEntry:
    def test_signal_fires_once_persistence_satisfied(self):
        cfg, fc = engine_cfg(), fees_cfg()
        state = EngineState()
        state = _persist_to_entry(state, cfg, fc, window())
        result = step(
            state,
            snap(now=T0 + timedelta(seconds=1.5), bk=EDGE_BOOK, fair=EDGE_FAIR_YES),
            cfg,
            fc,
            vol_spike_limit=0.01,
        )
        assert result.signal is not None
        assert result.signal.action is Action.BUY
        assert result.signal.side is Side.YES
        assert result.signal.limit_price == Decimal("0.50")
        assert result.state.pending_signal is result.signal

    def test_gap_dip_resets_persistence_timer(self):
        cfg, fc = engine_cfg(), fees_cfg()
        state = EngineState()
        state = _persist_to_entry(state, cfg, fc, window())
        # Gap disappears for one tick (flat/no-edge book).
        result = step(state, snap(now=T0 + timedelta(seconds=1.0)), cfg, fc, vol_spike_limit=0.01)
        assert result.signal is None
        assert result.state.gap_exceeded_since[Side.YES] is None
        # Edge returns, but persistence must restart from here.
        result = step(
            result.state,
            snap(now=T0 + timedelta(seconds=1.4), bk=EDGE_BOOK, fair=EDGE_FAIR_YES),
            cfg,
            fc,
            vol_spike_limit=0.01,
        )
        assert result.signal is None  # only 0s of the new persistence window elapsed

    def test_both_sides_qualifying_takes_neither(self):
        cfg, fc = engine_cfg(), fees_cfg()
        wide_book = book(yes_bid="0.30", yes_ask="0.32", no_bid="0.30", no_ask="0.32")
        both_fair = FairValue(yes=0.55, no=0.55)
        state = EngineState()
        result = step(state, snap(bk=wide_book, fair=both_fair), cfg, fc, vol_spike_limit=0.01)
        state = result.state
        result = step(
            state,
            snap(now=T0 + timedelta(seconds=1.5), bk=wide_book, fair=both_fair),
            cfg,
            fc,
            vol_spike_limit=0.01,
        )
        assert result.signal is None
        assert result.note == "both_sides_qualified_took_neither"

    def test_cooldown_blocks_entry(self):
        cfg, fc = engine_cfg(), fees_cfg()
        state = EngineState(cooldown_until=T0 + timedelta(seconds=100))
        state = _persist_to_entry(state, cfg, fc, window())
        result = step(
            state,
            snap(now=T0 + timedelta(seconds=1.5), bk=EDGE_BOOK, fair=EDGE_FAIR_YES),
            cfg,
            fc,
            vol_spike_limit=0.01,
        )
        assert result.signal is None

    def test_entry_cutoff_blocks_entry_near_close(self):
        cfg, fc = engine_cfg(), fees_cfg()
        win = window(close_in_s=80.0)  # < entry_cutoff_s(90)
        state = EngineState()
        result = step(
            state, snap(win=win, bk=EDGE_BOOK, fair=EDGE_FAIR_YES), cfg, fc, vol_spike_limit=0.01
        )
        state = result.state
        result = step(
            state,
            snap(now=T0 + timedelta(seconds=1.5), win=win, bk=EDGE_BOOK, fair=EDGE_FAIR_YES),
            cfg,
            fc,
            vol_spike_limit=0.01,
        )
        assert result.signal is None

    def test_stale_data_blocks_entry(self):
        cfg, fc = engine_cfg(), fees_cfg()
        state = EngineState()
        result = step(
            state,
            snap(bk=EDGE_BOOK, fair=EDGE_FAIR_YES, pyth_age_s=10.0),
            cfg,
            fc,
            vol_spike_limit=0.01,
        )
        assert result.signal is None
        assert result.state.gap_exceeded_since[Side.YES] is None  # frozen, not started

    def test_wide_spread_blocks_entry(self):
        cfg, fc = engine_cfg(max_spread=0.01), fees_cfg()
        state = EngineState()
        state = _persist_to_entry(state, cfg, fc, window())
        result = step(
            state,
            snap(now=T0 + timedelta(seconds=1.5), bk=EDGE_BOOK, fair=EDGE_FAIR_YES),
            cfg,
            fc,
            vol_spike_limit=0.01,
        )
        assert result.signal is None

    def test_vol_spike_blocks_entry(self):
        cfg, fc = engine_cfg(), fees_cfg()
        state = EngineState()
        state = _persist_to_entry(state, cfg, fc, window())
        result = step(
            state,
            snap(
                now=T0 + timedelta(seconds=1.5),
                bk=EDGE_BOOK,
                fair=EDGE_FAIR_YES,
                short_horizon_sigma_per_minute=0.05,
            ),
            cfg,
            fc,
            vol_spike_limit=0.01,
        )
        assert result.signal is None

    def test_max_round_trips_blocks_entry(self):
        cfg, fc = engine_cfg(max_round_trips=1), fees_cfg()
        state = EngineState(round_trips=1)
        state = _persist_to_entry(state, cfg, fc, window())
        result = step(
            state,
            snap(now=T0 + timedelta(seconds=1.5), bk=EDGE_BOOK, fair=EDGE_FAIR_YES),
            cfg,
            fc,
            vol_spike_limit=0.01,
        )
        assert result.signal is None

    def test_daily_loss_stop_blocks_entry(self):
        cfg, fc = engine_cfg(daily_loss_stop=10.0), fees_cfg()
        state = EngineState(realized_pnl_today=Decimal("-10.00"))
        state = _persist_to_entry(state, cfg, fc, window())
        result = step(
            state,
            snap(now=T0 + timedelta(seconds=1.5), bk=EDGE_BOOK, fair=EDGE_FAIR_YES),
            cfg,
            fc,
            vol_spike_limit=0.01,
        )
        assert result.signal is None


def _long_yes_state(entry_price="0.50", size="1") -> EngineState:
    from gold_edge.models import Position

    position = Position(
        window_ticker="KXGOLD15M-TEST",
        side=Side.YES,
        size=Decimal(size),
        entry_price=Decimal(entry_price),
        entered_at=T0,
        state=PositionState.LONG_YES,
    )
    return EngineState(position_state=PositionState.LONG_YES, position=position)


class TestExit:
    def test_overshoot_triggers_sell(self):
        cfg, fc = engine_cfg(), fees_cfg()
        state = _long_yes_state(entry_price="0.50")
        # exit_value = bid - fee(bid); fair=0.50 -> bid=0.60 gives exit_value
        # well above fair -> overshoot.
        bk = book(yes_bid="0.60", yes_ask="0.62")
        result = step(
            state, snap(bk=bk, fair=FairValue(yes=0.50, no=0.50)), cfg, fc, vol_spike_limit=0.01
        )
        assert result.signal is not None
        assert result.signal.action is Action.SELL
        assert result.signal.reason.startswith("overshoot")
        assert result.signal.limit_price == Decimal("0.60")

    def test_converged_triggers_sell(self):
        cfg, fc = engine_cfg(converge_band=0.01), fees_cfg()
        state = _long_yes_state(entry_price="0.50")
        # fair=0.60; bid=0.615 -> fee(0.615)=0.02 -> exit_value=0.595, which
        # is inside [fair-band, fair) = [0.59, 0.60): converged, not overshoot.
        bk = book(yes_bid="0.615", yes_ask="0.63")
        result = step(
            state, snap(bk=bk, fair=FairValue(yes=0.60, no=0.40)), cfg, fc, vol_spike_limit=0.01
        )
        assert result.signal is not None
        assert result.signal.reason.startswith("converged")

    def test_stop_triggers_sell(self):
        cfg, fc = engine_cfg(stop=0.08), fees_cfg()
        state = _long_yes_state(entry_price="0.50")
        bk = book(yes_bid="0.41", yes_ask="0.43")  # 0.50 - 0.08 = 0.42 threshold
        result = step(
            state, snap(bk=bk, fair=FairValue(yes=0.55, no=0.45)), cfg, fc, vol_spike_limit=0.01
        )
        assert result.signal is not None
        assert result.signal.reason.startswith("stop")

    def test_time_cutoff_triggers_sell(self):
        cfg, fc = engine_cfg(exit_cutoff_s=75.0, hold_to_settlement_when_itm=False), fees_cfg()
        state = _long_yes_state(entry_price="0.50")
        win = window(close_in_s=50.0)
        bk = book(yes_bid="0.50", yes_ask="0.52")
        result = step(
            state,
            snap(win=win, bk=bk, fair=FairValue(yes=0.55, no=0.45)),
            cfg,
            fc,
            vol_spike_limit=0.01,
        )
        assert result.signal is not None
        assert result.signal.reason.startswith("time_cutoff")

    def test_hold_to_settlement_suppresses_time_cutoff_when_itm(self):
        cfg = engine_cfg(
            exit_cutoff_s=75.0,
            hold_to_settlement_when_itm=True,
            hold_to_settlement_fair_threshold=0.90,
        )
        fc = fees_cfg()
        state = _long_yes_state(entry_price="0.50")
        win = window(close_in_s=50.0)
        bk = book(yes_bid="0.93", yes_ask="0.95")
        result = step(
            state,
            snap(win=win, bk=bk, fair=FairValue(yes=0.95, no=0.05)),
            cfg,
            fc,
            vol_spike_limit=0.01,
        )
        # exit_value(0.93-fee) is not >= fair(0.95)-band, no overshoot/converge;
        # not stopped; time cutoff reached but suppressed by ITM hold.
        assert result.signal is None

    def test_stale_triggers_advisory_sell_when_nothing_else_fires(self):
        cfg, fc = engine_cfg(), fees_cfg()
        state = _long_yes_state(entry_price="0.50")
        bk = book(yes_bid="0.50", yes_ask="0.52")
        result = step(
            state,
            snap(bk=bk, fair=FairValue(yes=0.55, no=0.45), pyth_age_s=10.0),
            cfg,
            fc,
            vol_spike_limit=0.01,
        )
        assert result.signal is not None
        assert result.signal.reason.startswith("stale_data")

    def test_hysteresis_gap_decay_alone_does_not_exit(self):
        cfg, fc = engine_cfg(converge_band=0.01, stop=0.08), fees_cfg()
        state = _long_yes_state(entry_price="0.50")
        # fair has drifted down toward the entry price (edge decayed) but
        # exit_value is nowhere near fair, not stopped, plenty of time left.
        bk = book(yes_bid="0.50", yes_ask="0.52")
        result = step(
            state, snap(bk=bk, fair=FairValue(yes=0.52, no=0.48)), cfg, fc, vol_spike_limit=0.01
        )
        assert result.signal is None


class TestFlip:
    def test_flip_hint_set_when_opposite_edge_qualifies(self):
        cfg, fc = engine_cfg(), fees_cfg()
        state = _long_yes_state(entry_price="0.50")
        # Overshoot on YES, and NO is now deeply mispriced too.
        bk = book(yes_bid="0.70", yes_ask="0.72", no_bid="0.10", no_ask="0.12")
        result = step(
            state, snap(bk=bk, fair=FairValue(yes=0.50, no=0.50)), cfg, fc, vol_spike_limit=0.01
        )
        assert result.signal is not None
        assert "flip_NO" in result.signal.reason
        assert result.state.flip_hint_side is Side.NO

    def test_flip_skips_cooldown_and_preseeds_persistence(self):
        cfg, fc = engine_cfg(cooldown_s=20.0, persist_s=1.5), fees_cfg()
        state = _long_yes_state(entry_price="0.50")
        bk = book(yes_bid="0.70", yes_ask="0.72", no_bid="0.10", no_ask="0.12")
        result = step(
            state, snap(bk=bk, fair=FairValue(yes=0.50, no=0.50)), cfg, fc, vol_spike_limit=0.01
        )
        sell_signal = result.signal
        fill = Fill(
            signal_id=sell_signal.id,
            window_ticker="KXGOLD15M-TEST",
            side=Side.YES,
            action=Action.SELL,
            price=Decimal("0.70"),
            size=Decimal(1),
            logged_at=T0 + timedelta(seconds=2),
        )
        new_state = apply_fill(result.state, fill, cfg, fc)
        assert new_state.position_state is PositionState.FLAT
        assert new_state.cooldown_until is None
        # Persistence for NO should already be satisfied (pre-seeded).
        assert new_state.gap_exceeded_since[Side.NO] is not None
        assert new_state.gap_exceeded_since[Side.NO] <= fill.logged_at - timedelta(seconds=1.5)

    def test_no_flip_hint_on_stop_alone_without_opposite_edge(self):
        cfg, fc = engine_cfg(), fees_cfg()
        state = _long_yes_state(entry_price="0.50")
        bk = book(yes_bid="0.41", yes_ask="0.43", no_bid="0.55", no_ask="0.57")
        result = step(
            state, snap(bk=bk, fair=FairValue(yes=0.55, no=0.45)), cfg, fc, vol_spike_limit=0.01
        )
        assert result.signal is not None
        assert "flip" not in result.signal.reason

    def test_stop_never_flips_even_when_opposite_side_qualifies(self):
        # Same deeply-mispriced NO side as test_flip_hint_set_when_opposite_edge_qualifies,
        # but this time the exit reason is "stop", not "overshoot" — a stop
        # must never flip, regardless of how attractive the other side looks.
        cfg, fc = engine_cfg(stop=0.08), fees_cfg()
        state = _long_yes_state(entry_price="0.50")
        bk = book(yes_bid="0.41", yes_ask="0.43", no_bid="0.10", no_ask="0.12")
        result = step(
            state, snap(bk=bk, fair=FairValue(yes=0.45, no=0.55)), cfg, fc, vol_spike_limit=0.01
        )
        assert result.signal is not None
        assert result.signal.reason.startswith("stop")
        assert "flip" not in result.signal.reason
        assert result.state.flip_hint_side is None


class TestSignalLifecycle:
    def test_pending_signal_blocks_new_decisions_until_resolved(self):
        cfg, fc = engine_cfg(), fees_cfg()
        state = EngineState()
        state = _persist_to_entry(state, cfg, fc, window())
        result = step(
            state,
            snap(now=T0 + timedelta(seconds=1.5), bk=EDGE_BOOK, fair=EDGE_FAIR_YES),
            cfg,
            fc,
            vol_spike_limit=0.01,
        )
        assert result.signal is not None
        pending = result.state
        # Even though the edge is still there, no second signal is issued.
        result2 = step(
            pending,
            snap(now=T0 + timedelta(seconds=2.0), bk=EDGE_BOOK, fair=EDGE_FAIR_YES),
            cfg,
            fc,
            vol_spike_limit=0.01,
        )
        assert result2.signal is None
        assert result2.state.pending_signal is pending.pending_signal

    def test_signal_expires_after_ttl_and_resets_persistence(self):
        cfg, fc = engine_cfg(signal_ttl_s=6.0), fees_cfg()
        state = EngineState()
        state = _persist_to_entry(state, cfg, fc, window())
        result = step(
            state,
            snap(now=T0 + timedelta(seconds=1.5), bk=EDGE_BOOK, fair=EDGE_FAIR_YES),
            cfg,
            fc,
            vol_spike_limit=0.01,
        )
        pending_state = result.state
        expiry_time = T0 + timedelta(seconds=1.5 + 6.0)
        result2 = step(
            pending_state,
            snap(now=expiry_time, bk=EDGE_BOOK, fair=EDGE_FAIR_YES),
            cfg,
            fc,
            vol_spike_limit=0.01,
        )
        assert result2.state.pending_signal is None
        assert len(result2.finalized_signals) == 1
        assert result2.finalized_signals[0].status is SignalStatus.EXPIRED
        # "Don't chase": persistence restarts fresh at the expiry step rather
        # than carrying over, so the same tick does not immediately re-fire.
        assert result2.signal is None
        assert result2.state.gap_exceeded_since[Side.YES] == expiry_time

        # Only after persist_s elapses again does a new signal fire.
        result3 = step(
            result2.state,
            snap(now=expiry_time + timedelta(seconds=1.5), bk=EDGE_BOOK, fair=EDGE_FAIR_YES),
            cfg,
            fc,
            vol_spike_limit=0.01,
        )
        assert result3.signal is not None

    def test_signal_expires_when_price_moves_past_limit(self):
        cfg, fc = engine_cfg(signal_ttl_s=60.0), fees_cfg()
        state = EngineState()
        state = _persist_to_entry(state, cfg, fc, window())
        result = step(
            state,
            snap(now=T0 + timedelta(seconds=1.5), bk=EDGE_BOOK, fair=EDGE_FAIR_YES),
            cfg,
            fc,
            vol_spike_limit=0.01,
        )
        worse_book = book(yes_bid="0.48", yes_ask="0.55", no_bid="0.30", no_ask="0.32")
        result2 = step(
            result.state,
            snap(now=T0 + timedelta(seconds=1.6), bk=worse_book, fair=EDGE_FAIR_YES),
            cfg,
            fc,
            vol_spike_limit=0.01,
        )
        assert result2.state.pending_signal is None
        assert result2.finalized_signals[0].status is SignalStatus.EXPIRED


class TestFillsAndSkips:
    def test_buy_fill_opens_position(self):
        cfg, fc = engine_cfg(), fees_cfg()
        state = EngineState()
        state = _persist_to_entry(state, cfg, fc, window())
        result = step(
            state,
            snap(now=T0 + timedelta(seconds=1.5), bk=EDGE_BOOK, fair=EDGE_FAIR_YES),
            cfg,
            fc,
            vol_spike_limit=0.01,
        )
        fill = Fill(
            signal_id=result.signal.id,
            window_ticker="KXGOLD15M-TEST",
            side=Side.YES,
            action=Action.BUY,
            price=Decimal("0.50"),
            size=Decimal(1),
            logged_at=T0 + timedelta(seconds=2),
        )
        new_state = apply_fill(result.state, fill, cfg, fc)
        assert new_state.position_state is PositionState.LONG_YES
        assert new_state.position.entry_price == Decimal("0.50")
        assert new_state.pending_signal is None

    def test_sell_fill_realizes_pnl_and_sets_cooldown(self):
        cfg, fc = engine_cfg(cooldown_s=20.0), fees_cfg()
        state = _long_yes_state(entry_price="0.50")
        fill = Fill(
            signal_id="s1",
            window_ticker="KXGOLD15M-TEST",
            side=Side.YES,
            action=Action.SELL,
            price=Decimal("0.60"),
            size=Decimal(1),
            logged_at=T0 + timedelta(seconds=10),
        )
        new_state = apply_fill(state, fill, cfg, fc)
        assert new_state.position_state is PositionState.FLAT
        assert new_state.round_trips == 1
        assert new_state.realized_pnl_today > Decimal("0")  # bought .50, sold .60, minus fees
        assert new_state.cooldown_until == fill.logged_at + timedelta(seconds=20.0)

    def test_skip_clears_pending_without_changing_position(self):
        cfg, fc = engine_cfg(), fees_cfg()
        state = EngineState()
        state = _persist_to_entry(state, cfg, fc, window())
        result = step(
            state,
            snap(now=T0 + timedelta(seconds=1.5), bk=EDGE_BOOK, fair=EDGE_FAIR_YES),
            cfg,
            fc,
            vol_spike_limit=0.01,
        )
        skipped = apply_skip(result.state)
        assert skipped.position_state is PositionState.FLAT
        assert skipped.pending_signal is None
        assert skipped.gap_exceeded_since[Side.YES] is None

    def test_sell_fill_with_no_position_raises(self):
        cfg, fc = engine_cfg(), fees_cfg()
        state = EngineState()
        fill = Fill(
            signal_id=None,
            window_ticker="KXGOLD15M-TEST",
            side=Side.YES,
            action=Action.SELL,
            price=Decimal("0.60"),
            size=Decimal(1),
            logged_at=T0,
        )
        with pytest.raises(ValueError):
            apply_fill(state, fill, cfg, fc)
