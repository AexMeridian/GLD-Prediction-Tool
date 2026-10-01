"""Loads a promoted market+model blend artifact for the live seam in
server.py. Never raises: every failure mode (missing file, corrupt JSON, a
tampered/half-written file, a stale model) just means "run on the raw
baseline" -- see server.py's `_recompute_and_step`, which always computes
the raw fair value regardless of whether this returns something, and only
substitutes a `LoadedModel`'s prediction when one loads cleanly.

Loaded once at server startup only (same as `Settings` itself) -- no
mid-session hot-swap, per CLAUDE.md's "no change mid-session."
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from gold_edge.learning.blend import blend_probability
from gold_edge.learning.blend_shadow import BlendArtifact, make_artifact
from gold_edge.model.fair_value import FairValue

logger = logging.getLogger(__name__)

DEFAULT_MAX_AGE_DAYS = 30.0


@dataclass(frozen=True)
class LoadedModel:
    artifact: BlendArtifact

    def predict_fair(self, market_mid: float, raw_fair_yes: float) -> FairValue:
        p = blend_probability(self.artifact.weights, market_mid, raw_fair_yes)
        if self.artifact.calibrator is not None:
            p = self.artifact.calibrator.predict(p)
        return FairValue(yes=p, no=1.0 - p)


def _hash_is_valid(artifact: BlendArtifact) -> bool:
    """Recomputes the artifact's own version hash from its recorded
    `trained_at` timestamp and compares -- catches a hand-edited or
    half-written file where the weights/metadata no longer match the hash
    that was supposedly computed from them."""
    try:
        trained_at = datetime.fromisoformat(artifact.trained_at)
    except ValueError:
        return False
    recomputed = make_artifact(
        artifact.weights, artifact.half_life_s, artifact.n_windows, trained_at,
        calibrator=artifact.calibrator,
    )
    return recomputed.version_hash == artifact.version_hash


def load_active_model(
    path: str | Path | None, max_age_days: float = DEFAULT_MAX_AGE_DAYS
) -> LoadedModel | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        logger.warning("active_model_path %s does not exist; running on the raw baseline", p)
        return None
    try:
        artifact = BlendArtifact.load(p)
    except Exception as exc:  # noqa: BLE001 - a bad model file must never crash the server
        logger.warning("failed to load model artifact %s (%r); running on the raw baseline", p, exc)
        return None
    if not _hash_is_valid(artifact):
        logger.warning(
            "model artifact %s failed its version-hash check; running on the raw baseline", p
        )
        return None
    trained_at = datetime.fromisoformat(artifact.trained_at)
    age = datetime.now(UTC) - trained_at
    if age > timedelta(days=max_age_days):
        logger.warning(
            "model artifact %s is %.0f days old (max %.0f); running on the raw baseline",
            p,
            age.total_seconds() / 86400.0,
            max_age_days,
        )
        return None
    logger.info("loaded active model %s (version %s)", p, artifact.version_hash)
    return LoadedModel(artifact=artifact)
