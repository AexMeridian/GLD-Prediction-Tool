"""Turns the grades/markouts/user_fills_latency tables `gold-edge learn`
already populates into drift.py's `DriftBaseline`/`DriftMetrics` shapes.

The SAME aggregation (`_aggregate`) is used twice: once to snapshot a
`DriftBaseline` at promotion time (over everything graded up to that
moment -- the validation history that justified promoting this version),
and repeatedly afterward to compute a live `DriftMetrics` over a trailing
rolling window. Using one function for both means there is no separate
"baseline model" to keep in sync with the live computation; drift is always
"has recent grading moved away from what this version shipped against."

No new modeling math: every number here is a count, mean, or median over
rows the grader/markout/fill-latency recorder already wrote.
"""

from __future__ import annotations

import sqlite3
import statistics
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from gold_edge.learning.calibrator import brier_score
from gold_edge.learning.drift import DriftBaseline, DriftMetrics

# Below this many graded round trips in range, a drift check reports nothing
# rather than acting on a noisy tiny sample -- same MIN_BUCKET_N-style
# convention CLAUDE.md already applies to pattern mining.
MIN_ROUND_TRIPS = 30


def _aggregate(
    conn: sqlite3.Connection, start: datetime | None, end: datetime | None
) -> tuple[int, float, float, Decimal, float] | None:
    """Returns (n_round_trips, calibration_error, markout_sign_rate,
    pnl_per_round_trip, median_delay_s) over graded round trips in
    [start, end], or None if there isn't enough graded history in range to
    report anything."""
    conn.row_factory = sqlite3.Row
    where: list[str] = []
    params: list[str] = []
    if start is not None:
        where.append("graded_at >= ?")
        params.append(start.isoformat())
    if end is not None:
        where.append("graded_at <= ?")
        params.append(end.isoformat())
    clause = (" WHERE " + " AND ".join(where)) if where else ""

    grade_rows = conn.execute(
        f"SELECT entry_signal_id, net_pnl FROM grades{clause}", params
    ).fetchall()
    if len(grade_rows) < MIN_ROUND_TRIPS:
        return None

    n_round_trips = len(grade_rows)
    pnl_per_round_trip = (
        sum((Decimal(r["net_pnl"]) for r in grade_rows), Decimal("0")) / n_round_trips
    )
    entry_ids = [r["entry_signal_id"] for r in grade_rows]
    placeholders = ",".join("?" * len(entry_ids))

    calib_rows = conn.execute(
        "SELECT s.fair AS fair, s.side AS side, se.result AS result "
        "FROM signals s JOIN settlements se ON se.ticker = s.window_ticker "
        f"WHERE s.id IN ({placeholders})",
        entry_ids,
    ).fetchall()
    fairs = [r["fair"] for r in calib_rows if r["result"] in ("yes", "no")]
    outcomes = [
        1.0 if r["result"] == r["side"].lower() else 0.0
        for r in calib_rows
        if r["result"] in ("yes", "no")
    ]
    if not fairs:
        return None
    calibration_error = brier_score(fairs, outcomes)

    markout_rows = conn.execute(
        f"SELECT edge_markout FROM markouts WHERE signal_id IN ({placeholders}) "
        "AND edge_markout IS NOT NULL",
        entry_ids,
    ).fetchall()
    if not markout_rows:
        return None
    markout_sign_rate = sum(1 for r in markout_rows if r["edge_markout"] > 0) / len(markout_rows)

    delay_rows = conn.execute(
        f"SELECT delay_s FROM user_fills_latency WHERE signal_id IN ({placeholders}) "
        "AND delay_s IS NOT NULL",
        entry_ids,
    ).fetchall()
    median_delay_s = (
        statistics.median(r["delay_s"] for r in delay_rows) if delay_rows else 0.0
    )

    return n_round_trips, calibration_error, markout_sign_rate, pnl_per_round_trip, median_delay_s


def snapshot_drift_baseline(sqlite_path: Path, as_of: datetime) -> DriftBaseline | None:
    """Everything graded up to `as_of` (promotion time) -- the validation
    history that justified promoting this version. None if there isn't
    enough graded history yet to form a baseline."""
    conn = sqlite3.connect(sqlite_path)
    try:
        result = _aggregate(conn, start=None, end=as_of)
    finally:
        conn.close()
    if result is None:
        return None
    _n, calibration_error, markout_sign_rate, pnl_per_round_trip, median_delay_s = result
    return DriftBaseline(
        calibration_error=calibration_error,
        markout_sign_rate=markout_sign_rate,
        pnl_per_round_trip=pnl_per_round_trip,
        median_delay_s=median_delay_s,
    )


def compute_live_drift_metrics(
    sqlite_path: Path, since: datetime, now: datetime
) -> DriftMetrics | None:
    """A trailing rolling window of live grading, for comparison against a
    promoted version's `DriftBaseline`. None if there isn't enough graded
    history in the window yet."""
    conn = sqlite3.connect(sqlite_path)
    try:
        result = _aggregate(conn, start=since, end=now)
    finally:
        conn.close()
    if result is None:
        return None
    n_round_trips, calibration_error, markout_sign_rate, pnl_per_round_trip, median_delay_s = result
    return DriftMetrics(
        calibration_error=calibration_error,
        markout_sign_rate=markout_sign_rate,
        pnl_per_round_trip=pnl_per_round_trip,
        median_delay_s=median_delay_s,
        n_round_trips=n_round_trips,
    )


_EVIDENCE_KEYS = (
    "drift_baseline_calibration_error",
    "drift_baseline_markout_sign_rate",
    "drift_baseline_pnl_per_round_trip",
    "drift_baseline_median_delay_s",
)


def baseline_to_evidence(baseline: DriftBaseline) -> dict[str, str]:
    """Folded into a promoted ConfigVersion's/ModelVersion's evidence dict
    so the live drift loop can rebuild the exact baseline it was promoted
    against without a separate storage location."""
    return {
        _EVIDENCE_KEYS[0]: str(baseline.calibration_error),
        _EVIDENCE_KEYS[1]: str(baseline.markout_sign_rate),
        _EVIDENCE_KEYS[2]: str(baseline.pnl_per_round_trip),
        _EVIDENCE_KEYS[3]: str(baseline.median_delay_s),
    }


def baseline_from_evidence(evidence: dict[str, str]) -> DriftBaseline | None:
    """None for any version promoted before this evidence snapshot existed,
    or with too little graded history at promotion time to form one --
    either way, the drift loop simply has nothing to check that version
    against yet."""
    if not all(k in evidence for k in _EVIDENCE_KEYS):
        return None
    return DriftBaseline(
        calibration_error=float(evidence[_EVIDENCE_KEYS[0]]),
        markout_sign_rate=float(evidence[_EVIDENCE_KEYS[1]]),
        pnl_per_round_trip=Decimal(evidence[_EVIDENCE_KEYS[2]]),
        median_delay_s=float(evidence[_EVIDENCE_KEYS[3]]),
    )
