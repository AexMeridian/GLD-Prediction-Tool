"""Registry + promotion gates, per CLAUDE.md's "Promotion gates" section:

"A proposal becomes live only if ALL pass:
- >= MIN_PROPOSAL_TRADES held-out round trips
- held-out net P&L improvement >= PROMOTION_MARGIN with the CI lower bound
  above zero
- max drawdown not worse by more than MAX_DD_WORSEN
- Shadow mode ... must still beat live config
- User approval"

and "Registry rules": every config/model versioned (hash + timestamp +
evidence snapshot), every signal stores the version that produced it,
rollback in one step, max one promotion per day, no change mid-session,
and anti-overfitting re-proposal cooldown.

This module owns the gate LOGIC and the version bookkeeping; it doesn't
decide shadow-mode results or user approval itself -- those are inputs
(`shadow_beats_live`, `user_approved`) supplied by shadow.py and the
dashboard/CLI respectively, kept as plain booleans here so this module has
no import-time dependency on either.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from decimal import Decimal

from gold_edge.config import LearningConfig
from gold_edge.learning.proposer import Proposal

REPROPOSAL_COOLDOWN_DAYS = 14
# How close two param dicts must be, per shared key, to count as
# "near-identical" for the anti-overfitting re-proposal cooldown.
_NEAR_IDENTICAL_TOLERANCE = 1e-9


@dataclass(frozen=True)
class GateResult:
    passed: bool
    reasons: list[str] = field(default_factory=list)


def config_hash(param_changes: dict[str, float]) -> str:
    canonical = json.dumps(param_changes, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class ConfigVersion:
    version_hash: str
    param_changes: dict[str, float]
    created_at: datetime
    evidence: dict[str, str]  # JSON-serializable snapshot of the Proposal that produced it
    promoted: bool = False


@dataclass(frozen=True)
class RejectedProposal:
    param_changes: dict[str, float]
    rejected_at: datetime
    reasons: list[str]


def evaluate_promotion_gates(
    proposal: Proposal,
    learning_cfg: LearningConfig,
    shadow_beats_live: bool,
    user_approved: bool,
) -> GateResult:
    """Every gate CLAUDE.md lists, checked independently so a rejection
    always names every reason it failed, not just the first one -- useful
    for both the dashboard's Approve/Reject panel and the "don't re-propose
    near-identical params" cooldown, which needs to know if a proposal is
    even in the right ballpark to be worth retrying later."""
    reasons: list[str] = []

    if proposal.n_holdout_trades < learning_cfg.min_proposal_trades:
        reasons.append(
            f"only {proposal.n_holdout_trades} held-out round trips "
            f"(need >= {learning_cfg.min_proposal_trades})"
        )

    margin = Decimal(str(learning_cfg.promotion_margin))
    if proposal.holdout_pnl_delta < margin:
        reasons.append(
            f"held-out P&L improvement ${proposal.holdout_pnl_delta:.2f} below "
            f"PROMOTION_MARGIN ${margin:.2f}"
        )
    if proposal.holdout_pnl_delta_ci_low <= Decimal("0"):
        reasons.append(f"CI lower bound ${proposal.holdout_pnl_delta_ci_low:.2f} is not above zero")

    max_dd_worsen = Decimal(str(learning_cfg.max_dd_worsen))
    if proposal.holdout_drawdown_delta > max_dd_worsen:
        reasons.append(
            f"drawdown worsened by ${proposal.holdout_drawdown_delta:.2f} "
            f"(max allowed ${max_dd_worsen:.2f})"
        )

    if not shadow_beats_live:
        reasons.append("did not beat live config in shadow mode")
    if not user_approved:
        reasons.append("not yet approved by the user")

    return GateResult(passed=not reasons, reasons=reasons)


def _params_near_identical(a: dict[str, float], b: dict[str, float]) -> bool:
    if set(a) != set(b):
        return False
    return all(abs(a[k] - b[k]) < _NEAR_IDENTICAL_TOLERANCE for k in a)


def is_in_reproposal_cooldown(
    param_changes: dict[str, float], rejected: list[RejectedProposal], now: datetime
) -> bool:
    """CLAUDE.md: "if a proposal was rejected, don't re-propose near-
    identical params within N days." """
    cutoff = now - timedelta(days=REPROPOSAL_COOLDOWN_DAYS)
    return any(
        r.rejected_at >= cutoff and _params_near_identical(r.param_changes, param_changes)
        for r in rejected
    )


@dataclass
class Registry:
    """In-memory registry; callers persist via recorder.py's
    model_versions/config_versions/proposals tables (plain primitives, same
    circular-import-avoidance pattern as everywhere else recorder.py talks
    to learning/*). Kept separate from storage so the gate/version logic is
    testable without a database."""

    versions: list[ConfigVersion] = field(default_factory=list)
    rejected: list[RejectedProposal] = field(default_factory=list)
    promotions_by_date: dict[date, int] = field(default_factory=dict)

    def register_proposal_evidence(self, proposal: Proposal) -> dict[str, str]:
        return {
            "param_changes": json.dumps(proposal.param_changes),
            "n_holdout_trades": str(proposal.n_holdout_trades),
            "holdout_pnl_delta": str(proposal.holdout_pnl_delta),
            "holdout_pnl_delta_ci_low": str(proposal.holdout_pnl_delta_ci_low),
            "holdout_pnl_delta_ci_high": str(proposal.holdout_pnl_delta_ci_high),
            "holdout_drawdown_delta": str(proposal.holdout_drawdown_delta),
            "rationale": proposal.rationale,
        }

    def can_promote_today(self, today: date) -> bool:
        """CLAUDE.md: "Maximum one promotion per day.\""""
        return self.promotions_by_date.get(today, 0) == 0

    def promote(
        self,
        proposal: Proposal,
        learning_cfg: LearningConfig,
        shadow_beats_live: bool,
        user_approved: bool,
        now: datetime,
    ) -> GateResult:
        gate_result = evaluate_promotion_gates(
            proposal, learning_cfg, shadow_beats_live, user_approved
        )
        today = now.date()
        if gate_result.passed and not self.can_promote_today(today):
            gate_result = GateResult(
                passed=False, reasons=[*gate_result.reasons, "already promoted a change today"]
            )

        if not gate_result.passed:
            self.rejected.append(
                RejectedProposal(
                    param_changes=proposal.param_changes,
                    rejected_at=now,
                    reasons=gate_result.reasons,
                )
            )
            return gate_result

        version = ConfigVersion(
            version_hash=config_hash(proposal.param_changes),
            param_changes=proposal.param_changes,
            created_at=now,
            evidence=self.register_proposal_evidence(proposal),
            promoted=True,
        )
        self.versions.append(version)
        self.promotions_by_date[today] = self.promotions_by_date.get(today, 0) + 1
        return gate_result

    def current_version(self) -> ConfigVersion | None:
        promoted = [v for v in self.versions if v.promoted]
        return promoted[-1] if promoted else None

    def rollback(self) -> ConfigVersion | None:
        """Restores the previous promoted version in one step, per
        CLAUDE.md's "rollback restores any prior version in one step." The
        rolled-back version stays in history (marked not promoted) rather
        than being deleted, so version history is never rewritten."""
        promoted = [v for v in self.versions if v.promoted]
        if not promoted:
            return None
        rolled_back = promoted[-1]
        idx = self.versions.index(rolled_back)
        self.versions[idx] = ConfigVersion(
            version_hash=rolled_back.version_hash,
            param_changes=rolled_back.param_changes,
            created_at=rolled_back.created_at,
            evidence=rolled_back.evidence,
            promoted=False,
        )
        remaining_promoted = [v for v in self.versions if v.promoted]
        return remaining_promoted[-1] if remaining_promoted else None


# --- Probability-model promotion (parallel to the threshold-change path
# above): today's `Proposal`/`ConfigVersion` are strictly
# `param_changes: dict[str, float]` with a replay-derived P&L gate, which
# doesn't fit swapping in a learned probability model (a blend of market
# mid + model fair value, or a calibrated version of one). Rather than
# overload `Proposal` with fields that don't apply to threshold changes,
# this is its own explicitly-named path, sharing the same "check every
# gate independently, one promotion per day, never auto-promotes" spirit. ---


@dataclass(frozen=True)
class ModelProposalEvidence:
    """Evidence a probability-model candidate must carry to be gated, per
    model_proposer.py's `ProbabilityModelProposal` (kept here as a narrow
    protocol so registry.py doesn't import model_proposer.py -- the two
    would otherwise form a cycle, since model_proposer.py needs
    `evaluate_model_promotion_gates` from here)."""

    n_holdout_windows: int
    holdout_brier_ci_low: float
    n_shadow_trades: int
    n_shadow_days: int
    shadow_pnl_ci_low: float


@dataclass(frozen=True)
class ModelVersion:
    version_hash: str
    kind: str
    artifact_path: str
    created_at: datetime
    evidence: dict[str, str]
    promoted: bool = False


@dataclass(frozen=True)
class RejectedModelProposal:
    version_hash: str
    rejected_at: datetime
    reasons: list[str]


def evaluate_model_promotion_gates(
    evidence: ModelProposalEvidence, learning_cfg: LearningConfig, user_approved: bool
) -> GateResult:
    """Mirrors `evaluate_promotion_gates`'s style: every gate checked
    independently, every failure reason reported. Reuses the same
    `min_proposal_trades`/`shadow_sessions` numbers CLAUDE.md already
    specifies for threshold changes -- a probability-model swap is at
    least as consequential, not less."""
    reasons: list[str] = []

    if evidence.n_holdout_windows < learning_cfg.min_proposal_trades:
        reasons.append(
            f"only {evidence.n_holdout_windows} held-out windows "
            f"(need >= {learning_cfg.min_proposal_trades})"
        )
    if evidence.holdout_brier_ci_low <= 0:
        reasons.append(
            f"held-out Brier-vs-market CI lower bound {evidence.holdout_brier_ci_low:+.5f} "
            "is not above zero"
        )
    if evidence.n_shadow_trades < learning_cfg.min_proposal_trades:
        reasons.append(
            f"only {evidence.n_shadow_trades} settled live-shadow trades "
            f"(need >= {learning_cfg.min_proposal_trades})"
        )
    if evidence.n_shadow_days < learning_cfg.shadow_sessions:
        reasons.append(
            f"only {evidence.n_shadow_days} shadow days (need >= {learning_cfg.shadow_sessions})"
        )
    if evidence.shadow_pnl_ci_low <= 0:
        reasons.append(
            f"live-shadow P&L CI lower bound ${evidence.shadow_pnl_ci_low:+.3f} is not above "
            "zero -- backtest looking good is not enough on its own"
        )
    if not user_approved:
        reasons.append("not yet approved by the user")

    return GateResult(passed=not reasons, reasons=reasons)


@dataclass
class ModelRegistry:
    """Mirrors `Registry`'s shape for the model-promotion path. The
    one-promotion-per-day counter is passed in from outside (see
    actions.py's `_combined_promotions_today`) rather than kept separately,
    so a config-threshold promotion and a model promotion on the same day
    correctly count against the same "max one promotion per day" limit."""

    versions: list[ModelVersion] = field(default_factory=list)
    rejected: list[RejectedModelProposal] = field(default_factory=list)

    def promote(
        self,
        version_hash: str,
        kind: str,
        artifact_path: str,
        evidence: dict[str, str],
        gate_result: GateResult,
        can_promote_today: bool,
        now: datetime,
    ) -> GateResult:
        if gate_result.passed and not can_promote_today:
            gate_result = GateResult(
                passed=False, reasons=[*gate_result.reasons, "already promoted a change today"]
            )
        if not gate_result.passed:
            self.rejected.append(
                RejectedModelProposal(
                    version_hash=version_hash, rejected_at=now, reasons=gate_result.reasons
                )
            )
            return gate_result

        self.versions.append(
            ModelVersion(
                version_hash=version_hash,
                kind=kind,
                artifact_path=artifact_path,
                created_at=now,
                evidence=evidence,
                promoted=True,
            )
        )
        return gate_result

    def current_version(self) -> ModelVersion | None:
        promoted = [v for v in self.versions if v.promoted]
        return promoted[-1] if promoted else None

    def rollback(self) -> ModelVersion | None:
        promoted = [v for v in self.versions if v.promoted]
        if not promoted:
            return None
        rolled_back = promoted[-1]
        idx = self.versions.index(rolled_back)
        self.versions[idx] = ModelVersion(
            version_hash=rolled_back.version_hash,
            kind=rolled_back.kind,
            artifact_path=rolled_back.artifact_path,
            created_at=rolled_back.created_at,
            evidence=rolled_back.evidence,
            promoted=False,
        )
        remaining_promoted = [v for v in self.versions if v.promoted]
        return remaining_promoted[-1] if remaining_promoted else None
