from datetime import UTC, datetime

from gold_edge.config import LearningConfig
from gold_edge.learning.actions import promote_model, rollback_model
from gold_edge.learning.model_proposer import ProbabilityModelProposal, ShadowStats
from gold_edge.recorder import Recorder

T0 = datetime(2026, 1, 1, tzinfo=UTC)


def learning_cfg(**overrides) -> LearningConfig:
    base = dict(
        min_opportunity=0.03,
        min_bucket_n=30,
        exit_regret=0.02,
        min_proposal_trades=200,
        promotion_margin=0.01,
        max_dd_worsen=0.10,
        shadow_sessions=10,
        filter_min_prob=0.5,
        markout_horizons_s=[5.0, 15.0, 30.0, 60.0],
        news_shock_threshold=0.08,
    )
    base.update(overrides)
    return LearningConfig(**base)


def proposal(**overrides) -> ProbabilityModelProposal:
    base = dict(
        kind="blend",
        artifact_path="data/blend_model.json",
        version_hash="v1",
        feature_names=("logit_mid", "logit_fair"),
        n_dev_windows=3000,
        n_holdout_windows=250,
        holdout_brier_market=0.158,
        holdout_brier_model=0.157,
        holdout_brier_delta=0.001,
        holdout_brier_ci_low=0.0002,
        holdout_brier_ci_high=0.002,
        holdout_logloss_delta=0.005,
        shadow=ShadowStats(220, 12, 0.08, 0.02, 0.14),
        rationale="test",
        created_at=T0,
    )
    base.update(overrides)
    return ProbabilityModelProposal(**base)


def test_promote_model_persists_and_current_version_readable(tmp_path):
    sqlite_path = tmp_path / "test.sqlite"
    Recorder(sqlite_path).close()

    outcome = promote_model(sqlite_path, proposal(), learning_cfg())
    assert outcome.gate_result.passed
    assert outcome.promoted_version is not None
    assert outcome.promoted_version.version_hash == "v1"

    from gold_edge.learning.queries import load_model_versions

    versions = load_model_versions(sqlite_path)
    assert len(versions) == 1
    assert versions[0].promoted
    assert versions[0].artifact_path == "data/blend_model.json"


def test_promote_model_rejects_and_persists_nothing_when_gates_fail(tmp_path):
    sqlite_path = tmp_path / "test.sqlite"
    Recorder(sqlite_path).close()

    bad = proposal(shadow=ShadowStats(5, 1, 0.0, -0.1, 0.1))
    outcome = promote_model(sqlite_path, bad, learning_cfg())
    assert not outcome.gate_result.passed
    assert outcome.promoted_version is None

    from gold_edge.learning.queries import load_model_versions

    assert load_model_versions(sqlite_path) == []


def test_second_promotion_same_day_is_blocked(tmp_path):
    sqlite_path = tmp_path / "test.sqlite"
    Recorder(sqlite_path).close()

    first = promote_model(sqlite_path, proposal(version_hash="v1"), learning_cfg())
    assert first.gate_result.passed

    second = promote_model(sqlite_path, proposal(version_hash="v2"), learning_cfg())
    assert not second.gate_result.passed
    assert any("already promoted" in r for r in second.gate_result.reasons)


def test_rollback_restores_earlier_version(tmp_path):
    sqlite_path = tmp_path / "test.sqlite"
    Recorder(sqlite_path).close()

    promote_model(sqlite_path, proposal(version_hash="v1", created_at=T0), learning_cfg())
    from datetime import timedelta

    promote_model(
        sqlite_path,
        proposal(version_hash="v2", created_at=T0 + timedelta(days=1)),
        learning_cfg(),
    )

    restored = rollback_model(sqlite_path)
    assert restored is not None and restored.version_hash == "v1"


def test_rollback_with_nothing_promoted_returns_none(tmp_path):
    sqlite_path = tmp_path / "test.sqlite"
    Recorder(sqlite_path).close()
    assert rollback_model(sqlite_path) is None
