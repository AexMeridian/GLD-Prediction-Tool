from datetime import UTC, datetime, timedelta
from decimal import Decimal

from gold_edge.learning.proposer import Proposal
from gold_edge.learning.registry import (
    Registry,
    RejectedProposal,
    config_hash,
    evaluate_promotion_gates,
    is_in_reproposal_cooldown,
)
from tests.test_learning_grader import learning_cfg

T0 = datetime(2026, 1, 1, tzinfo=UTC)


def proposal(
    n_trades=250,
    pnl_delta="5.00",
    ci_low="1.00",
    ci_high="9.00",
    dd_delta="0.05",
    params=None,
) -> Proposal:
    return Proposal(
        id="p1",
        param_changes=params or {"enter_edge": 0.04},
        n_holdout_trades=n_trades,
        holdout_pnl_delta=Decimal(pnl_delta),
        holdout_pnl_delta_ci_low=Decimal(ci_low),
        holdout_pnl_delta_ci_high=Decimal(ci_high),
        holdout_drawdown_delta=Decimal(dd_delta),
        rationale="test",
        created_at=T0,
    )


class TestEvaluatePromotionGates:
    def test_passes_when_everything_clears(self):
        cfg = learning_cfg(min_proposal_trades=200, promotion_margin=0.01, max_dd_worsen=0.10)
        result = evaluate_promotion_gates(
            proposal(), cfg, shadow_beats_live=True, user_approved=True
        )
        assert result.passed is True
        assert result.reasons == []

    def test_fails_below_min_trades(self):
        result = evaluate_promotion_gates(
            proposal(n_trades=50),
            learning_cfg(min_proposal_trades=200),
            shadow_beats_live=True,
            user_approved=True,
        )
        assert result.passed is False
        assert any("held-out round trips" in r for r in result.reasons)

    def test_fails_below_promotion_margin(self):
        result = evaluate_promotion_gates(
            proposal(pnl_delta="0.001"),
            learning_cfg(promotion_margin=0.01),
            shadow_beats_live=True,
            user_approved=True,
        )
        assert result.passed is False
        assert any("PROMOTION_MARGIN" in r for r in result.reasons)

    def test_fails_when_ci_lower_bound_not_above_zero(self):
        result = evaluate_promotion_gates(
            proposal(ci_low="-0.50"),
            learning_cfg(),
            shadow_beats_live=True,
            user_approved=True,
        )
        assert result.passed is False
        assert any("CI lower bound" in r for r in result.reasons)

    def test_fails_when_drawdown_worsens_too_much(self):
        result = evaluate_promotion_gates(
            proposal(dd_delta="5.00"),
            learning_cfg(max_dd_worsen=0.10),
            shadow_beats_live=True,
            user_approved=True,
        )
        assert result.passed is False
        assert any("drawdown worsened" in r for r in result.reasons)

    def test_fails_without_shadow_win(self):
        result = evaluate_promotion_gates(
            proposal(), learning_cfg(), shadow_beats_live=False, user_approved=True
        )
        assert result.passed is False
        assert any("shadow mode" in r for r in result.reasons)

    def test_fails_without_user_approval(self):
        result = evaluate_promotion_gates(
            proposal(), learning_cfg(), shadow_beats_live=True, user_approved=False
        )
        assert result.passed is False
        assert any("approved" in r for r in result.reasons)

    def test_reports_every_failing_gate_not_just_the_first(self):
        result = evaluate_promotion_gates(
            proposal(n_trades=10, pnl_delta="0.001"),
            learning_cfg(min_proposal_trades=200, promotion_margin=0.01),
            shadow_beats_live=False,
            user_approved=False,
        )
        assert len(result.reasons) >= 4


class TestConfigHash:
    def test_same_params_same_hash(self):
        assert config_hash({"a": 1.0, "b": 2.0}) == config_hash({"b": 2.0, "a": 1.0})

    def test_different_params_different_hash(self):
        assert config_hash({"a": 1.0}) != config_hash({"a": 2.0})


class TestReproposalCooldown:
    def test_in_cooldown_for_near_identical_recent_rejection(self):
        rejected = [
            RejectedProposal(param_changes={"enter_edge": 0.04}, rejected_at=T0, reasons=["r"])
        ]
        assert is_in_reproposal_cooldown({"enter_edge": 0.04}, rejected, now=T0 + timedelta(days=3))

    def test_not_in_cooldown_after_window_passes(self):
        rejected = [
            RejectedProposal(param_changes={"enter_edge": 0.04}, rejected_at=T0, reasons=["r"])
        ]
        assert not is_in_reproposal_cooldown(
            {"enter_edge": 0.04}, rejected, now=T0 + timedelta(days=15)
        )

    def test_not_in_cooldown_for_different_params(self):
        rejected = [
            RejectedProposal(param_changes={"enter_edge": 0.04}, rejected_at=T0, reasons=["r"])
        ]
        assert not is_in_reproposal_cooldown(
            {"enter_edge": 0.10}, rejected, now=T0 + timedelta(days=1)
        )


class TestRegistry:
    def test_promote_succeeds_and_becomes_current_version(self):
        reg = Registry()
        result = reg.promote(
            proposal(), learning_cfg(), shadow_beats_live=True, user_approved=True, now=T0
        )
        assert result.passed is True
        assert reg.current_version() is not None
        assert reg.current_version().param_changes == {"enter_edge": 0.04}

    def test_promote_rejected_records_reasons_and_no_current_version(self):
        reg = Registry()
        result = reg.promote(
            proposal(n_trades=1), learning_cfg(), shadow_beats_live=True, user_approved=True, now=T0
        )
        assert result.passed is False
        assert reg.current_version() is None
        assert len(reg.rejected) == 1

    def test_only_one_promotion_per_day(self):
        reg = Registry()
        first = reg.promote(
            proposal(params={"enter_edge": 0.04}),
            learning_cfg(),
            shadow_beats_live=True,
            user_approved=True,
            now=T0,
        )
        second = reg.promote(
            proposal(params={"enter_edge": 0.05}),
            learning_cfg(),
            shadow_beats_live=True,
            user_approved=True,
            now=T0 + timedelta(hours=2),
        )
        assert first.passed is True
        assert second.passed is False
        assert any("already promoted" in r for r in second.reasons)

    def test_promotion_allowed_again_next_day(self):
        reg = Registry()
        reg.promote(
            proposal(params={"enter_edge": 0.04}),
            learning_cfg(),
            shadow_beats_live=True,
            user_approved=True,
            now=T0,
        )
        second = reg.promote(
            proposal(params={"enter_edge": 0.05}),
            learning_cfg(),
            shadow_beats_live=True,
            user_approved=True,
            now=T0 + timedelta(days=1),
        )
        assert second.passed is True

    def test_rollback_restores_previous_version(self):
        reg = Registry()
        reg.promote(
            proposal(params={"enter_edge": 0.04}),
            learning_cfg(),
            shadow_beats_live=True,
            user_approved=True,
            now=T0,
        )
        reg.promote(
            proposal(params={"enter_edge": 0.05}),
            learning_cfg(),
            shadow_beats_live=True,
            user_approved=True,
            now=T0 + timedelta(days=1),
        )
        assert reg.current_version().param_changes == {"enter_edge": 0.05}
        restored = reg.rollback()
        assert restored.param_changes == {"enter_edge": 0.04}
        assert reg.current_version().param_changes == {"enter_edge": 0.04}

    def test_rollback_with_no_promotions_returns_none(self):
        reg = Registry()
        assert reg.rollback() is None
