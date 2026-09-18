"""Drift monitoring, per CLAUDE.md: "Rolling live checks: calibration
error, markout sign rate, net P&L per round trip, delay profile. If metrics
degrade beyond thresholds vs the promoted version's validation results, show
a warning banner, suggest raising ENTER_EDGE or pausing, and offer
rollback. Drift alerts never change config automatically."

This module only compares live rolling metrics against the baseline a
promoted `registry.ConfigVersion` shipped with (stored in its `evidence`
snapshot) -- it never touches config itself; the CLI/dashboard decide what
to do with a `DriftEvent`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal

# How much worse a live metric can get before it counts as drift, relative
# to the promoted version's own validation numbers. Kept as constants (not
# config) since these are about detecting degradation from whatever
# thresholds THAT version was promoted under, not a tunable strategy
# parameter itself.
CALIBRATION_ERROR_TOLERANCE = 0.05  # absolute Brier/log-loss increase
MARKOUT_SIGN_RATE_TOLERANCE = 0.10  # fractional drop in "markout agreed with call" rate
PNL_PER_TRIP_TOLERANCE_FRACTION = 0.5  # live can be this much worse (as a fraction) than baseline
DELAY_MULTIPLE_TOLERANCE = 1.5  # live median delay this many times the baseline is a warning


@dataclass(frozen=True)
class DriftBaseline:
    """Snapshot taken at promotion time -- what "normal" looked like for
    this config version during its own validation."""

    calibration_error: float
    markout_sign_rate: float
    pnl_per_round_trip: Decimal
    median_delay_s: float


@dataclass(frozen=True)
class DriftMetrics:
    """The same four metrics, computed over a recent rolling live window."""

    calibration_error: float
    markout_sign_rate: float
    pnl_per_round_trip: Decimal
    median_delay_s: float
    n_round_trips: int


@dataclass(frozen=True)
class DriftEvent:
    detected_at: datetime
    degraded_metrics: list[str]
    suggestion: str

    @property
    def has_drift(self) -> bool:
        return bool(self.degraded_metrics)


def check_drift(baseline: DriftBaseline, live: DriftMetrics, now: datetime) -> DriftEvent:
    """Checks each metric independently and names every one that's
    degraded, so a rollback decision is informed by the full picture, not
    just whichever check happened to run first."""
    degraded: list[str] = []

    if live.calibration_error - baseline.calibration_error > CALIBRATION_ERROR_TOLERANCE:
        degraded.append(
            f"calibration error {live.calibration_error:.3f} vs baseline "
            f"{baseline.calibration_error:.3f}"
        )

    if baseline.markout_sign_rate - live.markout_sign_rate > MARKOUT_SIGN_RATE_TOLERANCE:
        degraded.append(
            f"markout sign rate {live.markout_sign_rate:.1%} vs baseline "
            f"{baseline.markout_sign_rate:.1%}"
        )

    pnl_floor = baseline.pnl_per_round_trip * Decimal(str(1 - PNL_PER_TRIP_TOLERANCE_FRACTION))
    if baseline.pnl_per_round_trip > 0 and live.pnl_per_round_trip < pnl_floor:
        degraded.append(
            f"P&L per round trip ${live.pnl_per_round_trip:.3f} vs baseline "
            f"${baseline.pnl_per_round_trip:.3f}"
        )

    if (
        baseline.median_delay_s > 0
        and live.median_delay_s > baseline.median_delay_s * DELAY_MULTIPLE_TOLERANCE
    ):
        degraded.append(
            f"median fill delay {live.median_delay_s:.1f}s vs baseline "
            f"{baseline.median_delay_s:.1f}s"
        )

    suggestion = (
        "Consider raising ENTER_EDGE, pausing trading, or rolling back to the "
        "previous config version."
        if degraded
        else "No action needed."
    )
    return DriftEvent(detected_at=now, degraded_metrics=degraded, suggestion=suggestion)


@dataclass
class DriftMonitor:
    """Keeps the most recent `DriftEvent`s so a dashboard banner can show
    current status without re-querying storage on every render."""

    events: list[DriftEvent] = field(default_factory=list)
    max_history: int = 100

    def record(self, event: DriftEvent) -> None:
        self.events.append(event)
        if len(self.events) > self.max_history:
            self.events = self.events[-self.max_history :]

    @property
    def current_status(self) -> DriftEvent | None:
        return self.events[-1] if self.events else None
