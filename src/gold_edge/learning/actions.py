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
from gold_edge.learning.proposer import Proposal
from gold_edge.learning.queries import load_config_versions, load_proposals
from gold_edge.learning.registry import ConfigVersion, GateResult, Registry
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
        asyncio.run(
            recorder.record_config_version(
                version_hash=promoted_version.version_hash,
                param_changes=promoted_version.param_changes,
                evidence=promoted_version.evidence,
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
