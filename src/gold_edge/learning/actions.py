"""The actual promote/rollback ACTIONS (load evidence, evaluate gates,
persist the result) factored out of `cli.py` so the dashboard's Learning
tab (server.py) can trigger the exact same registry logic instead of a
parallel reimplementation -- CLAUDE.md's "promote CLI" and the Learning
tab's "Approve/Reject" are two entry points to one action, not two.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from gold_edge.config import LearningConfig
from gold_edge.learning.drift_metrics import baseline_to_evidence, snapshot_drift_baseline
from gold_edge.learning.model_proposer import ProbabilityModelProposal
from gold_edge.learning.proposer import Proposal
from gold_edge.learning.queries import load_config_versions, load_model_versions, load_proposals
from gold_edge.learning.registry import (
    ConfigVersion,
    GateResult,
    ModelRegistry,
    ModelVersion,
    Registry,
    evaluate_model_promotion_gates,
)
from gold_edge.learning.shadow import ShadowRun, ShadowSessionResult
from gold_edge.recorder import Recorder


@dataclass(frozen=True)
class PromoteOutcome:
    found: bool
    gate_result: GateResult | None
    promoted_version: ConfigVersion | None
    shadow_sessions_seen: int
    shadow_beats_live: bool


def _load_shadow_run(sqlite_path: Path, param_changes: dict) -> ShadowRun:
    import sqlite3

    conn = sqlite3.connect(sqlite_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT * FROM shadow_signals WHERE candidate_param_changes_json = ?",
            (json.dumps(param_changes),),
        ).fetchall()
    finally:
        conn.close()
    return ShadowRun(
        candidate_param_changes=param_changes,
        sessions=[
            ShadowSessionResult(
                r["window_ticker"], Decimal(r["live_net_pnl"]), Decimal(r["candidate_net_pnl"])
            )
            for r in rows
        ],
    )


def _rehydrated_registry(sqlite_path: Path) -> Registry:
    registry = Registry(versions=load_config_versions(sqlite_path))
    for v in registry.versions:
        if v.promoted:
            day = v.created_at.date()
            registry.promotions_by_date[day] = registry.promotions_by_date.get(day, 0) + 1
    return registry


def promote_proposal(
    sqlite_path: Path, proposal_id: str, learning_cfg: LearningConfig
) -> PromoteOutcome:
    """CLAUDE.md's "User approval in the Learning tab (or `promote` CLI)":
    calling this function IS the approval -- both entry points treat the
    act of invoking it as `user_approved=True`, since a human explicitly
    chose to call it either way."""
    matches = [p for p, _status in load_proposals(sqlite_path) if p.id == proposal_id]
    if not matches:
        return PromoteOutcome(
            found=False,
            gate_result=None,
            promoted_version=None,
            shadow_sessions_seen=0,
            shadow_beats_live=False,
        )
    proposal: Proposal = matches[0]

    shadow_run = _load_shadow_run(sqlite_path, proposal.param_changes)
    shadow_beats_live = shadow_run.beats_live(min_sessions=learning_cfg.shadow_sessions)

    registry = _rehydrated_registry(sqlite_path)
    now = datetime.now(UTC)
    gate_result = registry.promote(
        proposal, learning_cfg, shadow_beats_live, user_approved=True, now=now
    )

    recorder = Recorder(sqlite_path)
    promoted_version: ConfigVersion | None = None
    if gate_result.passed:
        promoted_version = registry.current_version()
        assert promoted_version is not None
        # Drift monitoring (CLAUDE.md: "vs the promoted version's validation
        # results") needs a baseline snapshot taken AT promotion time -- fold
        # it into the stored evidence alongside the proposal's own P&L
        # evidence. None (not enough graded history yet) just means the live
        # drift loop has nothing to check this version against yet.
        evidence = dict(promoted_version.evidence)
        baseline = snapshot_drift_baseline(sqlite_path, as_of=promoted_version.created_at)
        if baseline is not None:
            evidence.update(baseline_to_evidence(baseline))
        asyncio.run(
            recorder.record_config_version(
                version_hash=promoted_version.version_hash,
                param_changes=promoted_version.param_changes,
                evidence=evidence,
                promoted=True,
                created_at=promoted_version.created_at,
            )
        )
        asyncio.run(recorder.update_proposal_status(proposal_id, "approved"))
    else:
        asyncio.run(recorder.update_proposal_status(proposal_id, "rejected"))
    recorder.close()

    return PromoteOutcome(
        found=True,
        gate_result=gate_result,
        promoted_version=promoted_version,
        shadow_sessions_seen=shadow_run.n_sessions,
        shadow_beats_live=shadow_beats_live,
    )


def _combined_promotions_today(sqlite_path: Path, today) -> bool:  # noqa: ANN001 - date
    """CLAUDE.md's "maximum one promotion per day" applies across BOTH the
    threshold-change path and this probability-model path together, not as
    two separate budgets -- a day that already promoted a config change
    can't also promote a model swap."""
    config_days = {v.created_at.date() for v in load_config_versions(sqlite_path) if v.promoted}
    model_days = {v.created_at.date() for v in load_model_versions(sqlite_path) if v.promoted}
    return today not in config_days and today not in model_days


@dataclass(frozen=True)
class PromoteModelOutcome:
    gate_result: GateResult
    promoted_version: ModelVersion | None


def promote_model(
    sqlite_path: Path, proposal: ProbabilityModelProposal, learning_cfg: LearningConfig
) -> PromoteModelOutcome:
    """Calling this function IS the user approval, same convention as
    `promote_proposal` -- both `propose-model --promote` (CLI) and a future
    Learning-tab button are entry points to this one action."""
    now = proposal.created_at
    evidence = proposal.to_gate_evidence()
    gate_result = evaluate_model_promotion_gates(evidence, learning_cfg, user_approved=True)
    can_promote_today = _combined_promotions_today(sqlite_path, now.date())

    registry = ModelRegistry(versions=load_model_versions(sqlite_path))
    gate_result = registry.promote(
        version_hash=proposal.version_hash,
        kind=proposal.kind,
        artifact_path=proposal.artifact_path,
        evidence=proposal.to_evidence_dict(),
        gate_result=gate_result,
        can_promote_today=can_promote_today,
        now=now,
    )

    recorder = Recorder(sqlite_path)
    promoted_version: ModelVersion | None = None
    if gate_result.passed:
        promoted_version = registry.current_version()
        assert promoted_version is not None
        evidence = dict(promoted_version.evidence)
        baseline = snapshot_drift_baseline(sqlite_path, as_of=promoted_version.created_at)
        if baseline is not None:
            evidence.update(baseline_to_evidence(baseline))
        asyncio.run(
            recorder.record_model_version(
                version_hash=promoted_version.version_hash,
                kind=promoted_version.kind,
                evidence=evidence,
                promoted=True,
                created_at=promoted_version.created_at,
            )
        )
    recorder.close()
    return PromoteModelOutcome(gate_result=gate_result, promoted_version=promoted_version)


def rollback_model(sqlite_path: Path) -> ModelVersion | None:
    registry = ModelRegistry(versions=load_model_versions(sqlite_path))
    restored = registry.rollback()
    if restored is None:
        return None

    recorder = Recorder(sqlite_path)
    for version in registry.versions:
        asyncio.run(
            recorder.record_model_version(
                version_hash=version.version_hash,
                kind=version.kind,
                evidence=version.evidence,
                promoted=version.promoted,
                created_at=version.created_at,
            )
        )
    recorder.close()
    return restored


def rollback_config(sqlite_path: Path) -> ConfigVersion | None:
    registry = _rehydrated_registry(sqlite_path)
    restored = registry.rollback()
    if restored is None:
        return None

    recorder = Recorder(sqlite_path)
    for version in registry.versions:
        asyncio.run(
            recorder.record_config_version(
                version_hash=version.version_hash,
                param_changes=version.param_changes,
                evidence=version.evidence,
                promoted=version.promoted,
                created_at=version.created_at,
            )
        )
    recorder.close()
    return restored
