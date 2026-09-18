"""Learned component #4, per CLAUDE.md: "from logged fills, learn the
user's actual latency distribution (signal created_at -> fill logged_at,
and fill price vs limit). Replace the default 1-2s backtest delay with this
profile once n >= 50 fills. Also learn which signal types the user tends to
miss."

The profile is deliberately non-parametric: it stores the observed delay
samples and samples from them empirically (bootstrap), rather than fitting
a distribution family that might not match a real human's click-latency
shape (multi-modal, heavy-tailed from distractions, etc).
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass, field

MIN_FILLS_TO_LEARN = 50


@dataclass(frozen=True)
class FillLatencySample:
    signal_id: str
    action: str  # "BUY" | "SELL"
    reason: str  # the originating signal's `reason` (e.g. "enter_edge", "stop")
    delay_s: float
    was_missed: bool  # True if this candidate expired/was skipped rather than filled


@dataclass(frozen=True)
class DelayProfile:
    samples: list[float] = field(default_factory=list)
    miss_rate_by_reason: dict[str, float] = field(default_factory=dict)
    n: int = 0

    @property
    def is_learned(self) -> bool:
        return self.n >= MIN_FILLS_TO_LEARN

    def sample_delay_s(self, rng: random.Random) -> float:
        """Empirical bootstrap draw from observed fill delays. Falls back to
        nothing meaningful if called before `is_learned` -- callers should
        check that first and use the configured default range instead."""
        if not self.samples:
            raise ValueError("no delay samples to draw from; check is_learned first")
        return rng.choice(self.samples)

    def mean_delay_s(self) -> float:
        if not self.samples:
            raise ValueError("no delay samples")
        return sum(self.samples) / len(self.samples)

    def percentile_delay_s(self, pct: float) -> float:
        if not self.samples:
            raise ValueError("no delay samples")
        ordered = sorted(self.samples)
        idx = min(len(ordered) - 1, max(0, round(pct / 100.0 * (len(ordered) - 1))))
        return ordered[idx]


def build_delay_profile(fill_samples: Sequence[FillLatencySample]) -> DelayProfile:
    filled = [s for s in fill_samples if not s.was_missed]
    delays = [s.delay_s for s in filled]

    by_reason_total: dict[str, int] = {}
    by_reason_missed: dict[str, int] = {}
    for s in fill_samples:
        by_reason_total[s.reason] = by_reason_total.get(s.reason, 0) + 1
        if s.was_missed:
            by_reason_missed[s.reason] = by_reason_missed.get(s.reason, 0) + 1
    miss_rate = {
        reason: by_reason_missed.get(reason, 0) / total for reason, total in by_reason_total.items()
    }

    return DelayProfile(samples=delays, miss_rate_by_reason=miss_rate, n=len(filled))


def sample_delay_s(
    profile: DelayProfile, rng: random.Random, fallback_min_s: float, fallback_max_s: float
) -> float:
    """The actual drop-in replacement for `backtest.fills.sample_delay_s`'s
    fixed uniform range: once the profile has learned enough (CLAUDE.md's
    n >= 50), draw from real observed delays; otherwise keep using the
    configured uniform fallback so early sessions (with no fill history
    yet) aren't left without any delay model at all."""
    if profile.is_learned:
        return profile.sample_delay_s(rng)
    return rng.uniform(fallback_min_s, fallback_max_s)
