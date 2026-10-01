"""Builds a `ProbabilityModelProposal` -- the evidence a market+model blend
(or blend+isotonic) needs to clear `registry.py`'s
`evaluate_model_promotion_gates` before it can ever be promoted to the live
seam in server.py. Two independent sources of evidence, both required:

- The offline holdout result (`learning/feature_blend.py`'s `Scored`,
  scored on the days deliberately held back from fitting).
- The LIVE shadow track record (`data/shadow.sqlite`, written by
  `learning/blend_shadow.py`'s `ShadowEngine` while `gold-edge shadow` runs
  against real, free public data -- never real orders).

CLAUDE.md's shadow-mode requirement exists precisely because these two can
diverge (see docs/... the 2026-09-30 finding: a coverage bug made an early
live result look far worse than the same period's offline hindsight) --
this module keeps them as two separate, both-required numbers rather than
quietly trusting the backtest alone.
"""

from __future__ import annotations

import random
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from gold_edge.learning.feature_blend import Scored
from gold_edge.learning.market_study import _cluster_bootstrap
from gold_edge.learning.registry import ModelProposalEvidence


@dataclass(frozen=True)
class ShadowStats:
    n_trades: int
    n_days: int
    pnl_mean: float
    pnl_ci_low: float
    pnl_ci_high: float


def load_shadow_stats(shadow_db: Path, n_boot: int = 2000, seed: int = 0) -> ShadowStats:
    """Reads every settled trade from the shadow database (whatever
    versions it spans -- a caller wanting only trades from the CURRENT
    artifact version should filter by `model_hash` first; this reads all
    of them, since a small number of trades from an earlier, very similar
    version is still real live evidence, not noise to discard)."""
    if not shadow_db.exists():
        return ShadowStats(0, 0, 0.0, 0.0, 0.0)
    conn = sqlite3.connect(f"file:{shadow_db}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT ticker, ts, pnl FROM shadow_trades WHERE pnl IS NOT NULL"
        ).fetchall()
    finally:
        conn.close()
    if not rows:
        return ShadowStats(0, 0, 0.0, 0.0, 0.0)
    by_window = {ticker: [pnl] for ticker, _ts, pnl in rows}
    n_days = len({ts[:10] for _ticker, ts, _pnl in rows})
    mean, lo, hi = _cluster_bootstrap(by_window, n_boot, random.Random(seed))
    return ShadowStats(len(rows), n_days, mean, lo, hi)


@dataclass(frozen=True)
class ProbabilityModelProposal:
    kind: str  # "blend" | "blend+isotonic"
    artifact_path: str
    version_hash: str
    feature_names: tuple[str, ...]
    n_dev_windows: int
    n_holdout_windows: int
    holdout_brier_market: float
    holdout_brier_model: float
    holdout_brier_delta: float  # market - model; positive = model beat the market
    holdout_brier_ci_low: float
    holdout_brier_ci_high: float
    holdout_logloss_delta: float
    shadow: ShadowStats
    rationale: str
    created_at: datetime

    def to_gate_evidence(self) -> ModelProposalEvidence:
        return ModelProposalEvidence(
            n_holdout_windows=self.n_holdout_windows,
            holdout_brier_ci_low=self.holdout_brier_ci_low,
            n_shadow_trades=self.shadow.n_trades,
            n_shadow_days=self.shadow.n_days,
            shadow_pnl_ci_low=self.shadow.pnl_ci_low,
        )

    def to_evidence_dict(self) -> dict[str, str]:
        return {
            "artifact_path": self.artifact_path,
            "feature_names": ",".join(self.feature_names),
            "n_dev_windows": str(self.n_dev_windows),
            "n_holdout_windows": str(self.n_holdout_windows),
            "holdout_brier_market": f"{self.holdout_brier_market:.6f}",
            "holdout_brier_model": f"{self.holdout_brier_model:.6f}",
            "holdout_brier_delta": f"{self.holdout_brier_delta:+.6f}",
            "holdout_brier_ci_low": f"{self.holdout_brier_ci_low:+.6f}",
            "holdout_brier_ci_high": f"{self.holdout_brier_ci_high:+.6f}",
            "holdout_logloss_delta": f"{self.holdout_logloss_delta:+.6f}",
            "n_shadow_trades": str(self.shadow.n_trades),
            "n_shadow_days": str(self.shadow.n_days),
            "shadow_pnl_mean": f"{self.shadow.pnl_mean:+.4f}",
            "shadow_pnl_ci_low": f"{self.shadow.pnl_ci_low:+.4f}",
            "shadow_pnl_ci_high": f"{self.shadow.pnl_ci_high:+.4f}",
            "rationale": self.rationale,
        }


def build_model_proposal(
    kind: str,
    artifact_path: str,
    version_hash: str,
    feature_names: tuple[str, ...],
    n_dev_windows: int,
    holdout: Scored,
    shadow: ShadowStats,
    now: datetime,
) -> ProbabilityModelProposal:
    rationale = (
        f"{kind} on {holdout.n_windows} held-out windows: Brier vs market "
        f"{holdout.diff_vs_market:+.5f} CI [{holdout.ci_low:+.5f}, {holdout.ci_high:+.5f}]; "
        f"live shadow {shadow.n_trades} trades over {shadow.n_days} days, mean P&L "
        f"${shadow.pnl_mean:+.3f} CI [${shadow.pnl_ci_low:+.3f}, ${shadow.pnl_ci_high:+.3f}]"
    )
    return ProbabilityModelProposal(
        kind=kind,
        artifact_path=artifact_path,
        version_hash=version_hash,
        feature_names=feature_names,
        n_dev_windows=n_dev_windows,
        n_holdout_windows=holdout.n_windows,
        holdout_brier_market=holdout.market_brier,
        holdout_brier_model=holdout.brier,
        holdout_brier_delta=holdout.diff_vs_market,
        holdout_brier_ci_low=holdout.ci_low,
        holdout_brier_ci_high=holdout.ci_high,
        holdout_logloss_delta=holdout.market_log_loss - holdout.log_loss,
        shadow=shadow,
        rationale=rationale,
        created_at=now,
    )
