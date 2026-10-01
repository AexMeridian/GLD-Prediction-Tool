from datetime import UTC, datetime, timedelta

from gold_edge.config import LearningConfig
from gold_edge.learning.blend_shadow import make_artifact
from gold_edge.learning.calibrator import IsotonicCalibrator
from gold_edge.learning.feature_blend import Scored
from gold_edge.learning.model_loader import load_active_model
from gold_edge.learning.model_proposer import (
    ShadowStats,
    build_model_proposal,
)
from gold_edge.learning.registry import (
    ModelProposalEvidence,
    ModelRegistry,
    ModelVersion,
    evaluate_model_promotion_gates,
)

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


def good_evidence(**overrides) -> ModelProposalEvidence:
    base = dict(
        n_holdout_windows=250,
        holdout_brier_ci_low=0.001,
        n_shadow_trades=220,
        n_shadow_days=12,
        shadow_pnl_ci_low=0.02,
    )
    base.update(overrides)
    return ModelProposalEvidence(**base)


class TestEvaluateModelPromotionGates:
    def test_all_gates_pass(self):
        result = evaluate_model_promotion_gates(good_evidence(), learning_cfg(), user_approved=True)
        assert result.passed
        assert result.reasons == []

    def test_each_gate_fails_independently_and_is_named(self):
        cfg = learning_cfg()
        cases = {
            "held-out windows": good_evidence(n_holdout_windows=10),
            "CI lower bound": good_evidence(holdout_brier_ci_low=-0.001),
            "shadow trades": good_evidence(n_shadow_trades=5),
            "shadow days": good_evidence(n_shadow_days=1),
            "shadow P&L": good_evidence(shadow_pnl_ci_low=-0.01),
        }
        for label, evidence in cases.items():
            result = evaluate_model_promotion_gates(evidence, cfg, user_approved=True)
            assert not result.passed, label
            assert len(result.reasons) == 1, (label, result.reasons)

    def test_not_approved_is_its_own_reason(self):
        result = evaluate_model_promotion_gates(
            good_evidence(), learning_cfg(), user_approved=False
        )
        assert not result.passed
        assert any("approved" in r for r in result.reasons)

    def test_multiple_failures_all_reported(self):
        result = evaluate_model_promotion_gates(
            good_evidence(n_holdout_windows=1, n_shadow_trades=1),
            learning_cfg(),
            user_approved=False,
        )
        assert len(result.reasons) == 3


class TestModelRegistry:
    def test_promote_then_current_version(self):
        reg = ModelRegistry()
        gate = evaluate_model_promotion_gates(good_evidence(), learning_cfg(), user_approved=True)
        result = reg.promote(
            version_hash="v1",
            kind="blend",
            artifact_path="data/blend_model.json",
            evidence={"x": "1"},
            gate_result=gate,
            can_promote_today=True,
            now=T0,
        )
        assert result.passed
        current = reg.current_version()
        assert current is not None
        assert current.version_hash == "v1" and current.promoted

    def test_second_promotion_same_day_is_blocked(self):
        reg = ModelRegistry()
        gate = evaluate_model_promotion_gates(good_evidence(), learning_cfg(), user_approved=True)
        reg.promote("v1", "blend", "p", {}, gate, can_promote_today=True, now=T0)
        result = reg.promote("v2", "blend", "p", {}, gate, can_promote_today=False, now=T0)
        assert not result.passed
        assert any("already promoted" in r for r in result.reasons)
        assert reg.current_version().version_hash == "v1"

    def test_failed_gate_is_rejected_not_promoted(self):
        reg = ModelRegistry()
        bad_gate = evaluate_model_promotion_gates(
            good_evidence(n_shadow_trades=1), learning_cfg(), user_approved=True
        )
        result = reg.promote("v1", "blend", "p", {}, bad_gate, can_promote_today=True, now=T0)
        assert not result.passed
        assert reg.current_version() is None
        assert len(reg.rejected) == 1

    def test_rollback_falls_back_to_an_earlier_still_promoted_version(self):
        reg = ModelRegistry(
            versions=[
                ModelVersion("v1", "blend", "p1", T0, {}, promoted=True),
                ModelVersion("v2", "blend", "p2", T0 + timedelta(days=1), {}, promoted=True),
            ]
        )
        assert reg.current_version().version_hash == "v2"
        restored = reg.rollback()
        assert restored is not None and restored.version_hash == "v1"
        assert reg.current_version().version_hash == "v1"

    def test_rollback_with_only_one_promoted_version_returns_none(self):
        reg = ModelRegistry(versions=[ModelVersion("v1", "blend", "p1", T0, {}, promoted=True)])
        assert reg.rollback() is None

    def test_rollback_with_no_promoted_version_returns_none(self):
        assert ModelRegistry().rollback() is None


def _scored(n_windows: int, diff: float, ci_low: float, ci_high: float) -> Scored:
    return Scored(
        names=("logit_mid", "logit_fair"),
        n_rows=n_windows * 10,
        n_windows=n_windows,
        market_log_loss=0.47,
        log_loss=0.46,
        market_brier=0.158,
        brier=0.157,
        diff_vs_market=diff,
        ci_low=ci_low,
        ci_high=ci_high,
        rows=[],
    )


class TestBuildModelProposal:
    def test_round_trips_into_gate_evidence(self):
        holdout = _scored(300, diff=0.001, ci_low=0.0002, ci_high=0.002)
        shadow = ShadowStats(
            n_trades=210, n_days=11, pnl_mean=0.08, pnl_ci_low=0.02, pnl_ci_high=0.14
        )
        proposal = build_model_proposal(
            "blend", "data/blend_model.json", "abc123", ("logit_mid", "logit_fair"),
            n_dev_windows=3000, holdout=holdout, shadow=shadow, now=T0,
        )
        evidence = proposal.to_gate_evidence()
        assert evidence.n_holdout_windows == 300
        assert evidence.holdout_brier_ci_low == 0.0002
        assert evidence.n_shadow_trades == 210
        assert evidence.n_shadow_days == 11
        assert evidence.shadow_pnl_ci_low == 0.02

    def test_evidence_dict_has_no_missing_fields(self):
        holdout = _scored(300, diff=0.001, ci_low=0.0002, ci_high=0.002)
        shadow = ShadowStats(0, 0, 0.0, 0.0, 0.0)
        proposal = build_model_proposal(
            "blend", "p.json", "h", ("logit_mid", "logit_fair"), 100, holdout, shadow, T0
        )
        d = proposal.to_evidence_dict()
        assert d["artifact_path"] == "p.json"
        assert d["n_shadow_trades"] == "0"


class TestLoadActiveModel:
    def test_none_path_returns_none(self):
        assert load_active_model(None) is None

    def test_missing_file_returns_none(self, tmp_path):
        assert load_active_model(tmp_path / "nope.json") is None

    def test_valid_fresh_artifact_loads(self, tmp_path):
        artifact = make_artifact((0.0, 1.0, 0.0), 900.0, 100, datetime.now(UTC))
        path = tmp_path / "m.json"
        artifact.save(path)
        loaded = load_active_model(path)
        assert loaded is not None
        assert loaded.artifact.version_hash == artifact.version_hash

    def test_tampered_weights_fail_hash_check(self, tmp_path):
        artifact = make_artifact((0.0, 1.0, 0.0), 900.0, 100, datetime.now(UTC))
        path = tmp_path / "m.json"
        artifact.save(path)
        text = path.read_text(encoding="utf-8")
        path.write_text(text.replace("1.0", "5.0"), encoding="utf-8")
        assert load_active_model(path) is None

    def test_stale_artifact_is_rejected(self, tmp_path):
        old = datetime.now(UTC) - timedelta(days=90)
        artifact = make_artifact((0.0, 1.0, 0.0), 900.0, 100, old)
        path = tmp_path / "m.json"
        artifact.save(path)
        assert load_active_model(path, max_age_days=30.0) is None

    def test_predict_fair_uses_blend_probability(self, tmp_path):
        artifact = make_artifact((0.0, 1.0, 0.0), 900.0, 100, datetime.now(UTC))
        path = tmp_path / "m.json"
        artifact.save(path)
        loaded = load_active_model(path)
        fv = loaded.predict_fair(market_mid=0.7, raw_fair_yes=0.1)
        assert abs(fv.yes - 0.7) < 1e-9  # weight 1 on market, 0 on model -> trusts the market
        assert abs(fv.yes + fv.no - 1.0) < 1e-9

    def test_predict_fair_applies_the_calibrator_when_present(self, tmp_path):
        cal = IsotonicCalibrator(x_thresholds=[0.0, 1.0], y_values=[0.9, 0.9])
        artifact = make_artifact(
            (0.0, 1.0, 0.0), 900.0, 100, datetime.now(UTC), calibrator=cal
        )
        path = tmp_path / "m.json"
        artifact.save(path)
        loaded = load_active_model(path)
        fv = loaded.predict_fair(market_mid=0.7, raw_fair_yes=0.1)
        assert abs(fv.yes - 0.9) < 1e-6
        assert abs(fv.yes + fv.no - 1.0) < 1e-9

    def test_tampered_calibrator_fails_hash_check(self, tmp_path):
        import json

        cal = IsotonicCalibrator(x_thresholds=[0.0, 1.0], y_values=[0.5, 0.5])
        artifact = make_artifact(
            (0.0, 1.0, 0.0), 900.0, 100, datetime.now(UTC), calibrator=cal
        )
        path = tmp_path / "m.json"
        artifact.save(path)
        d = json.loads(path.read_text(encoding="utf-8"))
        d["calibrator"]["y_values"] = [0.9, 0.9]
        path.write_text(json.dumps(d), encoding="utf-8")
        assert load_active_model(path) is None
