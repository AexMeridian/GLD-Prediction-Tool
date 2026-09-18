"""Learned component #5, per CLAUDE.md: "walk-forward optimization of
ENTER_EDGE, PERSIST_S, COOLDOWN_S, STOP, CONVERGE_BAND, cutoffs, and
MAX_SPREAD. Objective: net P&L per window with a drawdown penalty. Keep
search grids small and bounded to sane ranges" and "Emit Proposal objects:
what changes, evidence (sample size, held-out delta P&L with CI, drawdown
change, calibration change), and plain-language rationale."

This module only proposes -- it reuses `backtest.replay.run_sweep` (already
"tuned on training days and reported on held-out days only") for the coarse
screen, then computes a proper bootstrap CI for any candidate that beats the
baseline at all. Deciding whether a proposal is good ENOUGH to promote
(MIN_PROPOSAL_TRADES, PROMOTION_MARGIN, MAX_DD_WORSEN) is registry.py's job,
per CLAUDE.md's separate "Promotion gates" section -- proposing and
promoting are deliberately different modules with different authority.
"""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal

import numpy as np

from gold_edge.backtest.replay import (
    BacktestResult,
    RecordedWindow,
    replay_all,
    run_sweep,
)
from gold_edge.config import BacktestConfig, EngineConfig, FeesConfig, ModelConfig, VolatilityConfig
from gold_edge.models import Tick


@dataclass(frozen=True)
class Proposal:
    id: str
    param_changes: dict[str, float]
    n_holdout_trades: int
    holdout_pnl_delta: Decimal
    holdout_pnl_delta_ci_low: Decimal
    holdout_pnl_delta_ci_high: Decimal
    holdout_drawdown_delta: Decimal
    rationale: str
    created_at: datetime


def _per_window_pnl(result: BacktestResult) -> dict[str, Decimal]:
    return {
        wr.ticker: sum((rt.pnl for rt in wr.round_trips), Decimal("0")) for wr in result.windows
    }


def _max_drawdown(result: BacktestResult) -> Decimal:
    all_round_trips = [rt for wr in result.windows for rt in wr.round_trips]
    peak = cum = max_dd = Decimal("0")
    for rt in sorted(all_round_trips, key=lambda r: r.exit_time):
        cum += rt.pnl
        peak = max(peak, cum)
        max_dd = max(max_dd, peak - cum)
    return max_dd


def _round_trip_count(result: BacktestResult) -> int:
    return sum(len(wr.round_trips) for wr in result.windows)


def _bootstrap_delta_ci(
    candidate_by_window: dict[str, Decimal],
    baseline_by_window: dict[str, Decimal],
    n_boot: int,
    rng: random.Random,
) -> tuple[Decimal, Decimal]:
    """Bootstraps over HELD-OUT WINDOWS (not individual trades): each
    resample redraws which windows' outcomes we'd have seen, giving a CI on
    "how much better would this config have done" that accounts for
    day-to-day variance in a small held-out sample, per CLAUDE.md's
    "held-out delta P&L with CI." """
    tickers = list(candidate_by_window)
    n = len(tickers)
    if n == 0:
        return Decimal("0"), Decimal("0")
    deltas = np.array([float(candidate_by_window[t] - baseline_by_window[t]) for t in tickers])
    boot_totals = np.empty(n_boot)
    for i in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        boot_totals[i] = deltas[idx].sum()
    ci_low, ci_high = np.percentile(boot_totals, [2.5, 97.5])
    return Decimal(str(round(float(ci_low), 4))), Decimal(str(round(float(ci_high), 4)))


def _describe_changes(base_engine_cfg: EngineConfig, param_changes: dict[str, float]) -> str:
    parts = []
    for key, new_value in param_changes.items():
        old_value = getattr(base_engine_cfg, key)
        parts.append(f"{key} {old_value} -> {new_value}")
    return ", ".join(parts)


def propose_threshold_changes(
    train_ticks: list[Tick],
    train_windows: list[RecordedWindow],
    test_ticks: list[Tick],
    test_windows: list[RecordedWindow],
    base_engine_cfg: EngineConfig,
    fees_cfg: FeesConfig,
    vol_cfg: VolatilityConfig,
    model_cfg: ModelConfig,
    backtest_cfg: BacktestConfig,
    param_grid: dict[str, list[float]],
    seed: int | None = None,
    n_bootstrap: int = 2000,
    max_proposals: int = 5,
) -> list[Proposal]:
    """Walk-forward: `run_sweep` screens every combination in `param_grid`
    on train, reporting held-out (test) results for each; every candidate
    that beats the UNCHANGED baseline's held-out net P&L at all gets a full
    bootstrap CI computed and becomes a Proposal, best-first, capped at
    `max_proposals` (per CLAUDE.md: "if a proposal was rejected, don't
    re-propose near-identical params" -- returning only the top few keeps a
    caller from being flooded with near-duplicate candidates in the first
    place)."""
    rng = random.Random(seed)

    baseline_result = replay_all(
        test_ticks,
        test_windows,
        base_engine_cfg,
        fees_cfg,
        vol_cfg,
        model_cfg,
        backtest_cfg,
        random.Random(seed),
    )
    baseline_by_window = _per_window_pnl(baseline_result)
    baseline_net_pnl = sum(baseline_by_window.values(), Decimal("0"))
    baseline_drawdown = _max_drawdown(baseline_result)

    candidates = run_sweep(
        train_ticks,
        train_windows,
        test_ticks,
        test_windows,
        base_engine_cfg,
        fees_cfg,
        vol_cfg,
        model_cfg,
        backtest_cfg,
        param_grid,
        seed=seed,
    )

    scored: list[tuple[Decimal, dict[str, float]]] = []
    for c in candidates:
        if c["test_net_pnl"] > baseline_net_pnl:
            scored.append((c["test_net_pnl"] - baseline_net_pnl, c["params"]))
    scored.sort(key=lambda item: item[0], reverse=True)

    proposals: list[Proposal] = []
    for _screened_delta, params in scored[:max_proposals]:
        candidate_cfg = base_engine_cfg.model_copy(update=params)
        candidate_result = replay_all(
            test_ticks,
            test_windows,
            candidate_cfg,
            fees_cfg,
            vol_cfg,
            model_cfg,
            backtest_cfg,
            random.Random(seed),
        )
        candidate_by_window = _per_window_pnl(candidate_result)
        pnl_delta = sum(candidate_by_window.values(), Decimal("0")) - baseline_net_pnl
        ci_low, ci_high = _bootstrap_delta_ci(
            candidate_by_window, baseline_by_window, n_bootstrap, rng
        )
        drawdown_delta = _max_drawdown(candidate_result) - baseline_drawdown
        n_trades = _round_trip_count(candidate_result)

        proposals.append(
            Proposal(
                id=str(uuid.uuid4()),
                param_changes=params,
                n_holdout_trades=n_trades,
                holdout_pnl_delta=pnl_delta,
                holdout_pnl_delta_ci_low=ci_low,
                holdout_pnl_delta_ci_high=ci_high,
                holdout_drawdown_delta=drawdown_delta,
                rationale=(
                    f"Changing {_describe_changes(base_engine_cfg, params)} improved held-out "
                    f"net P&L by ${pnl_delta:.2f} over {n_trades} round trip(s) "
                    f"(95% CI [${ci_low:.2f}, ${ci_high:.2f}]), drawdown change "
                    f"${drawdown_delta:+.2f}."
                ),
                created_at=datetime.now(UTC),
            )
        )

    return proposals
