"""Shared read-only SQLite queries over recorder.py's learning tables, used
by both `cli.py` (learn/review/proposals/promote/rollback) and the
dashboard's Review/Learning tab endpoints (server.py) -- one place that
knows how to turn a `grades`/`attribution`/... row back into the dataclass
it came from, so the CLI and the dashboard can never disagree about what a
row means.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from gold_edge.learning.attribution import AttributionResult, Cause
from gold_edge.learning.drift import DriftEvent
from gold_edge.learning.grader import Grade
from gold_edge.learning.insights import GradedTrade
from gold_edge.learning.opportunities import FilterEvent, Opportunity
from gold_edge.learning.patterns import BucketStat
from gold_edge.learning.proposer import Proposal
from gold_edge.learning.registry import ConfigVersion
from gold_edge.models import Side


def _ranged(
    query: str, column: str, start: datetime | None, end: datetime | None
) -> tuple[str, list]:
    params: list = []
    conditions = []
    if start:
        conditions.append(f"{column} >= ?")
        params.append(start.isoformat())
    if end:
        conditions.append(f"{column} <= ?")
        params.append(end.isoformat())
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    return query, params


def load_session_data(
    sqlite_path: Path, start: datetime | None = None, end: datetime | None = None
) -> tuple[list[GradedTrade], list[AttributionResult], list[Opportunity], list[FilterEvent]]:
    conn = sqlite3.connect(sqlite_path)
    conn.row_factory = sqlite3.Row
    try:
        q, p = _ranged("SELECT * FROM grades", "graded_at", start, end)
        trades = [
            GradedTrade(
                row["window_ticker"],
                Grade(row["primary_grade"]),
                Decimal(row["net_pnl"]),
                Decimal("0"),
            )
            for row in conn.execute(q, p).fetchall()
        ]

        q, p = _ranged("SELECT * FROM attribution", "attributed_at", start, end)
        attribution = [
            AttributionResult(
                row["window_ticker"],
                Cause(row["cause"]),
                Decimal(row["dollar_impact"]),
                row["detail"],
            )
            for row in conn.execute(q, p).fetchall()
        ]

        q, p = _ranged("SELECT * FROM opportunities", "at", start, end)
        opportunities = [
            Opportunity(
                row["window_ticker"],
                Side(row["side"]),
                datetime.fromisoformat(row["at"]),
                row["reason"],
                Decimal(row["realistic_net_pnl"]),
                Decimal(row["oracle_net_pnl"]),
            )
            for row in conn.execute(q, p).fetchall()
        ]

        q, p = _ranged("SELECT * FROM filter_events", "at", start, end)
        filter_events = [
            FilterEvent(
                row["window_ticker"],
                Side(row["side"]),
                datetime.fromisoformat(row["at"]),
                row["reason"],
                Decimal(row["realistic_net_pnl"]),
            )
            for row in conn.execute(q, p).fetchall()
        ]
    finally:
        conn.close()
    return trades, attribution, opportunities, filter_events


def load_pattern_stats(sqlite_path: Path) -> list[BucketStat]:
    conn = sqlite3.connect(sqlite_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("SELECT * FROM pattern_stats ORDER BY computed_at DESC").fetchall()
    finally:
        conn.close()
    return [
        BucketStat(
            dimension=row["dimension"],
            bucket=row["bucket"],
            n=row["n"],
            mean_pnl=Decimal(row["mean_pnl"]),
            ci_low=Decimal(row["ci_low"]),
            ci_high=Decimal(row["ci_high"]),
            p_value=row["p_value"],
            significant=bool(row["significant"]),
        )
        for row in rows
    ]


def load_proposals(sqlite_path: Path) -> list[tuple[Proposal, str]]:
    """Returns (Proposal, status) pairs, newest first."""
    conn = sqlite3.connect(sqlite_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("SELECT * FROM proposals ORDER BY created_at DESC").fetchall()
    finally:
        conn.close()
    return [
        (
            Proposal(
                id=row["id"],
                param_changes=json.loads(row["param_changes_json"]),
                n_holdout_trades=row["n_holdout_trades"],
                holdout_pnl_delta=Decimal(row["holdout_pnl_delta"]),
                holdout_pnl_delta_ci_low=Decimal(row["holdout_pnl_delta_ci_low"]),
                holdout_pnl_delta_ci_high=Decimal(row["holdout_pnl_delta_ci_high"]),
                holdout_drawdown_delta=Decimal(row["holdout_drawdown_delta"]),
                rationale=row["rationale"],
                created_at=datetime.fromisoformat(row["created_at"]),
            ),
            row["status"],
        )
        for row in rows
    ]


def load_config_versions(sqlite_path: Path) -> list[ConfigVersion]:
    conn = sqlite3.connect(sqlite_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("SELECT * FROM config_versions ORDER BY created_at").fetchall()
    finally:
        conn.close()
    return [
        ConfigVersion(
            version_hash=row["version_hash"],
            param_changes=json.loads(row["param_changes_json"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            evidence=json.loads(row["evidence_json"]),
            promoted=bool(row["promoted"]),
        )
        for row in rows
    ]


def load_latest_drift_event(sqlite_path: Path) -> DriftEvent | None:
    conn = sqlite3.connect(sqlite_path)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT * FROM drift_events ORDER BY detected_at DESC LIMIT 1"
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    return DriftEvent(
        detected_at=datetime.fromisoformat(row["detected_at"]),
        degraded_metrics=json.loads(row["degraded_metrics_json"]),
        suggestion=row["suggestion"],
    )
