"""Shadow mode, per CLAUDE.md: "runs silently alongside the live config for
SHADOW_SESSIONS (default 10) sessions, generating its own hypothetical
signals, graded with the same grader; must still beat live config."

A "shadow session" here is one already-recorded window: shadow mode doesn't
place any orders or affect the live engine, it just replays the SAME
recorded data the live config saw through the candidate config (reusing
`backtest.replay.replay_all`, never a separate strategy path) and compares
net P&L. This lets a candidate accumulate a track record on real recent
market conditions before a human ever approves it.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from decimal import Decimal

from gold_edge.backtest.replay import RecordedWindow, replay_all
from gold_edge.config import BacktestConfig, EngineConfig, FeesConfig, ModelConfig, VolatilityConfig
from gold_edge.models import Tick

DEFAULT_SHADOW_SESSIONS = 10


@dataclass(frozen=True)
class ShadowSessionResult:
    window_ticker: str
    live_net_pnl: Decimal
    candidate_net_pnl: Decimal

    @property
    def candidate_wins(self) -> bool:
        return self.candidate_net_pnl > self.live_net_pnl


@dataclass(frozen=True)
class ShadowRun:
    candidate_param_changes: dict[str, float]
    sessions: list[ShadowSessionResult] = field(default_factory=list)

    @property
    def n_sessions(self) -> int:
        return len(self.sessions)

    @property
    def total_live_net_pnl(self) -> Decimal:
        return sum((s.live_net_pnl for s in self.sessions), Decimal("0"))

    @property
    def total_candidate_net_pnl(self) -> Decimal:
        return sum((s.candidate_net_pnl for s in self.sessions), Decimal("0"))

    def beats_live(self, min_sessions: int = DEFAULT_SHADOW_SESSIONS) -> bool:
        """CLAUDE.md: shadow mode "must still beat live config" -- and,
        implicitly, must actually have run for the configured number of
        sessions first; a candidate that's only shadowed twice hasn't earned
        an opinion yet regardless of how those two sessions went."""
        if self.n_sessions < min_sessions:
            return False
        return self.total_candidate_net_pnl > self.total_live_net_pnl


def run_shadow_session(
    window: RecordedWindow,
    ticks: list[Tick],
    live_engine_cfg: EngineConfig,
    candidate_param_changes: dict[str, float],
    fees_cfg: FeesConfig,
    vol_cfg: VolatilityConfig,
    model_cfg: ModelConfig,
    backtest_cfg: BacktestConfig,
    seed: int | None = None,
) -> ShadowSessionResult:
    """Replays ONE recorded window through both the live config and the
    candidate, using the same tick/book data and the same fill-delay RNG
    seed for both so the only thing that differs between them is the
    parameter change itself -- not incidental fill-timing noise."""
    candidate_cfg = live_engine_cfg.model_copy(update=candidate_param_changes)

    live_result = replay_all(
        ticks,
        [window],
        live_engine_cfg,
        fees_cfg,
        vol_cfg,
        model_cfg,
        backtest_cfg,
        random.Random(seed),
    )
    candidate_result = replay_all(
        ticks,
        [window],
        candidate_cfg,
        fees_cfg,
        vol_cfg,
        model_cfg,
        backtest_cfg,
        random.Random(seed),
    )
    live_pnl = sum((rt.pnl for wr in live_result.windows for rt in wr.round_trips), Decimal("0"))
    candidate_pnl = sum(
        (rt.pnl for wr in candidate_result.windows for rt in wr.round_trips), Decimal("0")
    )
    return ShadowSessionResult(
        window_ticker=window.window.ticker, live_net_pnl=live_pnl, candidate_net_pnl=candidate_pnl
    )


def extend_shadow_run(
    shadow_run: ShadowRun,
    new_windows: list[RecordedWindow],
    ticks: list[Tick],
    live_engine_cfg: EngineConfig,
    fees_cfg: FeesConfig,
    vol_cfg: VolatilityConfig,
    model_cfg: ModelConfig,
    backtest_cfg: BacktestConfig,
    seed: int | None = None,
) -> ShadowRun:
    """Appends newly-closed sessions to an existing shadow run -- called
    once per window close while a candidate is pending approval, never
    re-running past sessions (a shadow run's history doesn't change once
    recorded, only grows)."""
    new_sessions = [
        run_shadow_session(
            w,
            [t for t in ticks if w.window.open_time <= t.receive_time <= w.window.close_time],
            live_engine_cfg,
            shadow_run.candidate_param_changes,
            fees_cfg,
            vol_cfg,
            model_cfg,
            backtest_cfg,
            seed,
        )
        for w in new_windows
    ]
    return ShadowRun(
        candidate_param_changes=shadow_run.candidate_param_changes,
        sessions=[*shadow_run.sessions, *new_sessions],
    )
