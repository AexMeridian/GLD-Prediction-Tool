from datetime import UTC, datetime, timedelta
from decimal import Decimal

from gold_edge.backtest.replay import RecordedWindow
from gold_edge.learning.proposer import propose_threshold_changes
from gold_edge.models import Tick, Window
from tests.test_backtest_replay import bt_cfg, model_cfg, vol_cfg
from tests.test_state_machine import book, engine_cfg, fees_cfg

T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _strong_edge_window(entry_cutoff_scenario_len_s: float = 80.0) -> RecordedWindow:
    """A window where the underlying has moved 1% above S0 (fair_yes pins
    near 0.99 given the tiny vol floor) and the book is priced well below
    that -- a real, comfortably-above-cost edge that only ENTRY_CUTOFF_S
    controls: the window is short enough (80s) that the DEFAULT cutoff
    (90s) blocks entry for its entire length, while a lower cutoff (e.g.
    30s) opens a real entry window and lets the position ride to a winning
    settlement (hold_to_settlement_when_itm, since fair >= 0.90)."""
    win = Window(
        ticker="KXGOLD15M-PROPOSER",
        event_ticker="KXGOLD15M-PROPOSEREVT",
        series_ticker="KXGOLD15M",
        open_time=T0,
        close_time=T0 + timedelta(seconds=entry_cutoff_scenario_len_s),
        s0=Decimal("2000.00"),
        status="open",
    )
    n = int(entry_cutoff_scenario_len_s) + 1
    books = [
        book(
            yes_bid="0.83",
            yes_ask="0.85",
            no_bid="0.10",
            no_ask="0.12",
            now=T0 + timedelta(seconds=i),
        ).model_copy(update={"window_ticker": win.ticker})
        for i in range(n)
    ]
    return RecordedWindow(window=win, books=books, settlement_result="yes")


def _strong_edge_ticks(n_seconds: int = 80) -> list[Tick]:
    return [
        Tick(
            symbol="Metal.XAU/USD",
            price=2020.0,
            conf=0.1,
            expo=-2,
            publish_time=T0 + timedelta(seconds=i),
            receive_time=T0 + timedelta(seconds=i),
        )
        for i in range(n_seconds + 1)
    ]


class TestProposeThresholdChanges:
    def test_proposes_a_lower_entry_cutoff_that_unlocks_a_winning_trade(self):
        rw = _strong_edge_window()
        ticks = _strong_edge_ticks()
        proposals = propose_threshold_changes(
            train_ticks=ticks,
            train_windows=[rw],
            test_ticks=ticks,
            test_windows=[rw],
            base_engine_cfg=engine_cfg(),  # default entry_cutoff_s=90 -> never enters here
            fees_cfg=fees_cfg(),
            vol_cfg=vol_cfg(),
            model_cfg=model_cfg(),
            backtest_cfg=bt_cfg(human_delay_min_s=1.0, human_delay_max_s=1.0),
            param_grid={"entry_cutoff_s": [90.0, 30.0]},
            seed=0,
            n_bootstrap=200,
        )
        assert len(proposals) == 1
        proposal = proposals[0]
        assert proposal.param_changes == {"entry_cutoff_s": 30.0}
        assert proposal.n_holdout_trades == 1
        assert proposal.holdout_pnl_delta > Decimal("0")
        assert "entry_cutoff_s" in proposal.rationale
        assert proposal.holdout_pnl_delta_ci_low <= proposal.holdout_pnl_delta_ci_high

    def test_no_proposals_when_nothing_beats_baseline(self):
        rw = _strong_edge_window()
        ticks = _strong_edge_ticks()
        proposals = propose_threshold_changes(
            train_ticks=ticks,
            train_windows=[rw],
            test_ticks=ticks,
            test_windows=[rw],
            base_engine_cfg=engine_cfg(),
            fees_cfg=fees_cfg(),
            vol_cfg=vol_cfg(),
            model_cfg=model_cfg(),
            backtest_cfg=bt_cfg(human_delay_min_s=1.0, human_delay_max_s=1.0),
            param_grid={"entry_cutoff_s": [90.0, 95.0]},
            seed=0,
            n_bootstrap=200,
        )
        assert proposals == []

    def test_caps_proposals_at_max_proposals(self):
        rw = _strong_edge_window()
        ticks = _strong_edge_ticks()
        proposals = propose_threshold_changes(
            train_ticks=ticks,
            train_windows=[rw],
            test_ticks=ticks,
            test_windows=[rw],
            base_engine_cfg=engine_cfg(),
            fees_cfg=fees_cfg(),
            vol_cfg=vol_cfg(),
            model_cfg=model_cfg(),
            backtest_cfg=bt_cfg(human_delay_min_s=1.0, human_delay_max_s=1.0),
            param_grid={"entry_cutoff_s": [10.0, 20.0, 30.0, 40.0]},
            seed=0,
            n_bootstrap=200,
            max_proposals=2,
        )
        assert len(proposals) <= 2
