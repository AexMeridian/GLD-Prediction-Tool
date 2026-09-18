from datetime import UTC, datetime, timedelta
from decimal import Decimal

from gold_edge.backtest.replay import RecordedWindow
from gold_edge.learning.shadow import (
    DEFAULT_SHADOW_SESSIONS,
    ShadowRun,
    ShadowSessionResult,
    extend_shadow_run,
    run_shadow_session,
)
from gold_edge.models import Tick, Window
from tests.test_backtest_replay import bt_cfg, model_cfg, vol_cfg
from tests.test_state_machine import book, engine_cfg, fees_cfg

T0 = datetime(2026, 1, 1, tzinfo=UTC)


def strong_edge_window(ticker: str, length_s: float = 80.0) -> RecordedWindow:
    win = Window(
        ticker=ticker,
        event_ticker=f"{ticker}EVT",
        series_ticker="KXGOLD15M",
        open_time=T0,
        close_time=T0 + timedelta(seconds=length_s),
        s0=Decimal("2000.00"),
        status="open",
    )
    n = int(length_s) + 1
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


def strong_edge_ticks(n_seconds: int = 80) -> list[Tick]:
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


class TestRunShadowSession:
    def test_candidate_wins_when_it_unlocks_a_trade_live_missed(self):
        rw = strong_edge_window("KXGOLD15M-SHADOW1")
        ticks = strong_edge_ticks()
        result = run_shadow_session(
            window=rw,
            ticks=ticks,
            live_engine_cfg=engine_cfg(),  # entry_cutoff_s=90 -> never enters
            candidate_param_changes={"entry_cutoff_s": 30.0},
            fees_cfg=fees_cfg(),
            vol_cfg=vol_cfg(),
            model_cfg=model_cfg(),
            backtest_cfg=bt_cfg(human_delay_min_s=1.0, human_delay_max_s=1.0),
            seed=0,
        )
        assert result.live_net_pnl == Decimal("0")
        assert result.candidate_net_pnl > Decimal("0")
        assert result.candidate_wins is True

    def test_identical_configs_tie_and_candidate_does_not_win(self):
        rw = strong_edge_window("KXGOLD15M-SHADOW2")
        ticks = strong_edge_ticks()
        result = run_shadow_session(
            window=rw,
            ticks=ticks,
            live_engine_cfg=engine_cfg(),
            candidate_param_changes={},
            fees_cfg=fees_cfg(),
            vol_cfg=vol_cfg(),
            model_cfg=model_cfg(),
            backtest_cfg=bt_cfg(human_delay_min_s=1.0, human_delay_max_s=1.0),
            seed=0,
        )
        assert result.live_net_pnl == result.candidate_net_pnl
        assert result.candidate_wins is False


class TestShadowRun:
    def test_beats_live_false_before_enough_sessions(self):
        sessions = [ShadowSessionResult("W1", Decimal("0"), Decimal("1.0")) for _ in range(3)]
        run = ShadowRun(candidate_param_changes={"entry_cutoff_s": 30.0}, sessions=sessions)
        assert run.beats_live(min_sessions=DEFAULT_SHADOW_SESSIONS) is False

    def test_beats_live_true_after_enough_winning_sessions(self):
        sessions = [
            ShadowSessionResult(f"W{i}", Decimal("0"), Decimal("1.0"))
            for i in range(DEFAULT_SHADOW_SESSIONS)
        ]
        run = ShadowRun(candidate_param_changes={"entry_cutoff_s": 30.0}, sessions=sessions)
        assert run.beats_live(min_sessions=DEFAULT_SHADOW_SESSIONS) is True
        assert run.total_candidate_net_pnl == Decimal(str(DEFAULT_SHADOW_SESSIONS))

    def test_beats_live_false_when_candidate_underperforms_overall(self):
        sessions = [
            ShadowSessionResult("W1", Decimal("5.0"), Decimal("1.0"))
        ] * DEFAULT_SHADOW_SESSIONS
        run = ShadowRun(candidate_param_changes={}, sessions=sessions)
        assert run.beats_live(min_sessions=DEFAULT_SHADOW_SESSIONS) is False


class TestExtendShadowRun:
    def test_appends_new_sessions_without_altering_history(self):
        initial = ShadowRun(
            candidate_param_changes={"entry_cutoff_s": 30.0},
            sessions=[ShadowSessionResult("OLD", Decimal("0"), Decimal("0"))],
        )
        rw = strong_edge_window("KXGOLD15M-SHADOW3")
        ticks = strong_edge_ticks()
        updated = extend_shadow_run(
            initial,
            new_windows=[rw],
            ticks=ticks,
            live_engine_cfg=engine_cfg(),
            fees_cfg=fees_cfg(),
            vol_cfg=vol_cfg(),
            model_cfg=model_cfg(),
            backtest_cfg=bt_cfg(human_delay_min_s=1.0, human_delay_max_s=1.0),
            seed=0,
        )
        assert updated.n_sessions == 2
        assert updated.sessions[0].window_ticker == "OLD"
        assert updated.sessions[1].window_ticker == "KXGOLD15M-SHADOW3"
        assert initial.n_sessions == 1  # original untouched
