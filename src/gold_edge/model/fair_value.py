"""Fair-value probability of YES from the live Pyth price.

    tau        = minutes remaining in the window
    fair_yes   = norm.cdf( ln(S / S0) / (sigma_per_minute * sqrt(tau)) )
    fair_no    = 1 - fair_yes

At tau <= 0 (window closed or closing) or sigma <= 0 (a perfectly flat
feed, which shouldn't happen once volatility.py's floor is applied, but is
handled here too) the ratio is undefined; the outcome is then deterministic
from S vs S0, with ties resolving YES per the contract rules confirmed in
docs/contract_notes.md. Values are clamped to [min_fair_value,
max_fair_value] so a signal is never taken against a "certain" 0 or 1 that
the market itself won't quote to.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class FairValue:
    yes: float
    no: float


def compute_fair_value(
    s: float,
    s0: float,
    sigma_per_minute: float,
    tau_minutes: float,
    min_fair_value: float,
    max_fair_value: float,
) -> FairValue:
    if s <= 0 or s0 <= 0:
        raise ValueError(f"prices must be positive, got s={s}, s0={s0}")

    denom = sigma_per_minute * math.sqrt(tau_minutes) if tau_minutes > 0 else 0.0
    if denom <= 0:
        raw_yes = 1.0 if s >= s0 else 0.0
    else:
        z = math.log(s / s0) / denom
        # Scalar normal CDF via erf: identical to scipy's norm.cdf but ~100x
        # faster, which matters because replay/sweeps call this per book update.
        raw_yes = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

    yes = min(max(raw_yes, min_fair_value), max_fair_value)
    return FairValue(yes=yes, no=1.0 - yes)
