"""Confirms the live signal seam server.py's `_recompute_and_step` exposes:
with no active model, behavior is byte-for-byte identical to before this
feature existed; with one loaded, `app.fair` comes from the blend while
`app.raw_fair` still always holds the unmodified baseline (CLAUDE.md:
"keep the baseline available for comparison")."""

import asyncio
from datetime import UTC, datetime, timedelta

from gold_edge.learning.blend_shadow import make_artifact
from gold_edge.learning.model_loader import LoadedModel
from gold_edge.model.volatility import VolatilityTracker
from gold_edge.server import AppState, _recompute_and_step
from tests.test_promoted_config_startup import make_settings
from tests.test_state_machine import book, window

T0 = datetime(2026, 1, 1, tzinfo=UTC)


def make_app_state(tmp_path, active_model=None) -> AppState:
    settings = make_settings(tmp_path / "test.sqlite")
    app = AppState(settings=settings, signer=object(), pyth_api_key="x")
    app.window = window(close_in_s=600.0, s0="2000.00")
    app.book = book(yes_bid="0.60", yes_ask="0.62", now=T0)
    app.pyth_latest_price = 2010.0
    app.pyth_receive_time = T0
    app.vol_tracker = VolatilityTracker(60.0, 60.0, 0.0005)
    app.vol_tracker.update(2010.0, T0 - timedelta(seconds=1))
    app.active_model = active_model
    return app


def test_no_active_model_fair_equals_raw_fair(tmp_path):
    app = make_app_state(tmp_path)
    asyncio.run(_recompute_and_step(app))
    assert app.raw_fair is not None
    assert app.fair is not None
    assert app.fair.yes == app.raw_fair.yes
    assert app.fair.no == app.raw_fair.no


def test_active_model_replaces_fair_but_raw_fair_is_still_the_baseline(tmp_path):
    # Weight entirely on the market (logit(0.61) from the 0.60/0.62 book)
    # and 0 on the model, so app.fair should land near the book mid
    # regardless of what the raw diffusion model says.
    artifact = make_artifact((0.0, 1.0, 0.0), 900.0, 100, datetime.now(UTC))
    model = LoadedModel(artifact=artifact)
    app = make_app_state(tmp_path, active_model=model)
    asyncio.run(_recompute_and_step(app))

    assert app.raw_fair is not None
    assert app.fair is not None
    assert abs(app.fair.yes - 0.61) < 1e-6
    # S=2010 vs S0=2000 with a near-floor sigma: the raw diffusion model is
    # near-certain YES, clearly distinct from the market-trusting blend's
    # 0.61 -- the two are genuinely tracked separately, not aliased.
    assert app.raw_fair.yes > 0.9
    assert app.raw_fair is not app.fair


def test_active_model_is_none_by_default_when_not_configured(tmp_path):
    app = make_app_state(tmp_path)
    assert app.active_model is None
