"""FastAPI app: runs the feeds + engine and pushes state to the dashboard
over a WebSocket. Position state changes ONLY when the user logs a fill via
POST /api/fill (or /api/skip) — the background loop only ever calls
`step()`, never `apply_fill`.

The JSON-payload builders (`_signal_to_json`, `_position_to_json`,
`_build_state_payload`) are pure functions of plain arguments, kept
separate from the live orchestration (`AppState`, background tasks, WS
plumbing) specifically so they can be unit tested without a live feed —
see tests/test_server_payload.py. The orchestration itself needs a real
PYTH_API_KEY and Kalshi credentials to exercise end to end.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from gold_edge.config import FeesConfig, Settings
from gold_edge.engine.state_machine import EngineState, MarketSnapshot, apply_fill, apply_skip, step
from gold_edge.feeds.kalshi_auth import KalshiSigner
from gold_edge.feeds.kalshi_rest import KalshiRestClient
from gold_edge.feeds.kalshi_ws import KalshiWsClient
from gold_edge.feeds.pyth import PythFeedClient, get_market_hours
from gold_edge.learning.actions import PromoteOutcome, promote_proposal, rollback_config
from gold_edge.learning.insights import session_report_card
from gold_edge.learning.opportunities import filter_scorecard
from gold_edge.learning.queries import (
    load_config_versions,
    load_latest_drift_event,
    load_pattern_stats,
    load_proposals,
    load_session_data,
)
from gold_edge.model import fees as fees_mod
from gold_edge.model.fair_value import FairValue, compute_fair_value
from gold_edge.model.fees import settle_position_pnl
from gold_edge.model.volatility import VolatilityTracker
from gold_edge.models import Action, BookSnapshot, Fill, Position, PositionState, Window
from gold_edge.recorder import Recorder
from gold_edge.windows import check_settlement_source, get_current_window, get_next_window

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DASHBOARD_PATH = REPO_ROOT / "web" / "index.html"

SESSION_LOG_MAX = 500
SESSION_LOG_SENT_TO_CLIENT = 100


# --------------------------------------------------------------------------
# Pure JSON payload builders — unit tested without any live feed.
# --------------------------------------------------------------------------


def _signal_to_json(signal) -> dict[str, Any]:
    return {
        "id": signal.id,
        "window_ticker": signal.window_ticker,
        "action": signal.action.value,
        "side": signal.side.value,
        "limit_price": float(signal.limit_price),
        "size": float(signal.size),
        "fair": signal.fair,
        "market_price": float(signal.market_price),
        "edge_after_costs": signal.edge_after_costs,
        "reason": signal.reason,
        "created_at": signal.created_at.isoformat(),
        "expires_at": signal.expires_at.isoformat(),
        "status": signal.status.value,
    }


def _position_to_json(
    position: Position | None, book: BookSnapshot | None, fees_cfg: FeesConfig
) -> dict[str, Any] | None:
    if position is None:
        return None
    unrealized_pnl = None
    if book is not None:
        # exit_fee is a total dollar amount for closing the whole position
        # (the fee formula's C is total contracts) — subtract it once from
        # the total gross P&L, not per-contract before multiplying by size.
        exit_fee_total = fees_mod.taker_fee(
            position.size,
            book.bid(position.side),
            Decimal(str(fees_cfg.fee_multiplier)),
            Decimal(str(fees_cfg.base_rate)),
        )
        gross_pnl = (book.bid(position.side) - position.entry_price) * position.size
        unrealized_pnl = float(gross_pnl - exit_fee_total)
    return {
        "window_ticker": position.window_ticker,
        "side": position.side.value,
        "size": float(position.size),
        "entry_price": float(position.entry_price),
        "entered_at": position.entered_at.isoformat(),
        "state": position.state.value,
        "unrealized_pnl": unrealized_pnl,
    }


def _window_to_json(window: Window | None) -> dict[str, Any] | None:
    if window is None:
        return None
    return {
        "ticker": window.ticker,
        "event_ticker": window.event_ticker,
        "s0": float(window.s0) if window.s0 is not None else None,
        "open_time": window.open_time.isoformat(),
        "close_time": window.close_time.isoformat(),
        "status": window.status,
    }


def _book_to_json(book: BookSnapshot | None) -> dict[str, Any] | None:
    if book is None:
        return None
    return {
        "yes_bid": float(book.yes_bid),
        "yes_ask": float(book.yes_ask),
        "no_bid": float(book.no_bid),
        "no_ask": float(book.no_ask),
    }


def _build_state_payload(
    *,
    window: Window | None,
    latest_price: float | None,
    book: BookSnapshot | None,
    fair: FairValue | None,
    engine_state: EngineState,
    fees_cfg: FeesConfig,
    pyth_age_s: float | None,
    kalshi_age_s: float | None,
    stale_s: float,
    kill_switch: bool,
    session_log_tail: list[dict[str, Any]],
    settlement_source_ok: bool | None = None,
    pyth_market_open: bool | None = None,
) -> dict[str, Any]:
    pyth_stale = pyth_age_s is None or pyth_age_s > stale_s
    kalshi_stale = kalshi_age_s is None or kalshi_age_s > stale_s
    return {
        "type": "state",
        "generated_at": datetime.now(UTC).isoformat(),
        "window": _window_to_json(window),
        "market": (
            {"s": latest_price, **(_book_to_json(book) or {})} if book else {"s": latest_price}
        ),
        "fair": {"yes": fair.yes, "no": fair.no} if fair is not None else None,
        "position": _position_to_json(engine_state.position, book, fees_cfg),
        "signal": _signal_to_json(engine_state.pending_signal)
        if engine_state.pending_signal is not None
        else None,
        "feed_status": {
            "pyth_age_s": pyth_age_s,
            "kalshi_age_s": kalshi_age_s,
            "pyth_stale": pyth_stale,
            "kalshi_stale": kalshi_stale,
            "settlement_source_ok": settlement_source_ok,
            "pyth_market_open": pyth_market_open,
        },
        "session": {
            "round_trips": engine_state.round_trips,
            "realized_pnl_today": float(engine_state.realized_pnl_today),
            "kill_switch": kill_switch,
        },
        "session_log": session_log_tail,
    }


def _review_payload(
    sqlite_path: Path, start: datetime | None, end: datetime | None
) -> dict[str, Any]:
    if not sqlite_path.exists():
        return {"report": "No recorded data yet."}
    trades, attribution, opportunities, filter_events = load_session_data(sqlite_path, start, end)
    if not trades and not attribution and not opportunities and not filter_events:
        return {"report": "No graded data in that range yet -- run `learn` first."}
    report = session_report_card(
        trades, attribution, opportunities, filter_scorecard(filter_events)
    )
    return {"report": report}


def _learning_payload(sqlite_path: Path) -> dict[str, Any]:
    if not sqlite_path.exists():
        return {"patterns": [], "proposals": [], "versions": [], "drift": None}
    patterns = [
        {
            "dimension": s.dimension,
            "bucket": s.bucket,
            "n": s.n,
            "mean_pnl": float(s.mean_pnl),
            "ci_low": float(s.ci_low),
            "ci_high": float(s.ci_high),
            "p_value": s.p_value,
            "significant": s.significant,
        }
        for s in load_pattern_stats(sqlite_path)
        if s.significant
    ]
    proposals = [
        {
            "id": p.id,
            "param_changes": p.param_changes,
            "n_holdout_trades": p.n_holdout_trades,
            "holdout_pnl_delta": float(p.holdout_pnl_delta),
            "holdout_pnl_delta_ci_low": float(p.holdout_pnl_delta_ci_low),
            "holdout_pnl_delta_ci_high": float(p.holdout_pnl_delta_ci_high),
            "holdout_drawdown_delta": float(p.holdout_drawdown_delta),
            "rationale": p.rationale,
            "created_at": p.created_at.isoformat(),
            "status": status,
        }
        for p, status in load_proposals(sqlite_path)
    ]
    versions = [
        {
            "version_hash": v.version_hash,
            "param_changes": v.param_changes,
            "created_at": v.created_at.isoformat(),
            "promoted": v.promoted,
        }
        for v in load_config_versions(sqlite_path)
    ]
    drift = load_latest_drift_event(sqlite_path)
    drift_json = (
        {
            "detected_at": drift.detected_at.isoformat(),
            "degraded_metrics": drift.degraded_metrics,
            "suggestion": drift.suggestion,
            "has_drift": drift.has_drift,
        }
        if drift is not None
        else None
    )
    return {
        "patterns": patterns,
        "proposals": proposals,
        "versions": versions,
        "drift": drift_json,
    }


def _version_status_payload(sqlite_path: Path) -> dict[str, Any]:
    if not sqlite_path.exists():
        return {"current_version": None, "drift": None}
    promoted_versions = [v for v in load_config_versions(sqlite_path) if v.promoted]
    current = promoted_versions[-1] if promoted_versions else None
    drift = load_latest_drift_event(sqlite_path)
    return {
        "current_version": (
            {"version_hash": current.version_hash, "param_changes": current.param_changes}
            if current is not None
            else None
        ),
        "drift": (
            {
                "has_drift": drift.has_drift,
                "degraded_metrics": drift.degraded_metrics,
                "suggestion": drift.suggestion,
            }
            if drift is not None
            else None
        ),
    }


def _promote_outcome_payload(outcome: PromoteOutcome) -> dict[str, Any]:
    assert outcome.gate_result is not None
    return {
        "passed": outcome.gate_result.passed,
        "reasons": outcome.gate_result.reasons,
        "promoted_params": (
            outcome.promoted_version.param_changes if outcome.promoted_version is not None else None
        ),
    }


# --------------------------------------------------------------------------
# Live orchestration.
# --------------------------------------------------------------------------


@dataclass
class AppState:
    settings: Settings
    signer: KalshiSigner
    pyth_api_key: str

    window: Window | None = None
    latest_price: float | None = None
    pyth_receive_time: datetime | None = None
    book: BookSnapshot | None = None
    kalshi_receive_time: datetime | None = None
    fair: FairValue | None = None

    engine_state: EngineState = field(default_factory=EngineState)
    # The UTC calendar date `engine_state.realized_pnl_today` has been
    # accumulating for. DAILY_LOSS_STOP is meant to gate on losses *today*,
    # but realized_pnl_today otherwise just carries forward across window
    # boundaries forever — this is what actually rolls it over at midnight
    # UTC (see _window_rollover_loop).
    realized_pnl_date: date | None = None
    kill_switch: bool = False
    session_log: list[dict[str, Any]] = field(default_factory=list)
    connections: set[WebSocket] = field(default_factory=set)

    # Trustworthiness flags for the live price feed, surfaced to the
    # dashboard so a mismatch or a closed underlying market isn't silent.
    settlement_source_ok: bool | None = None
    pyth_market_open: bool | None = None

    vol_tracker: VolatilityTracker | None = None
    recorder: Recorder | None = None
    rest_client: KalshiRestClient | None = None

    _kalshi_task: asyncio.Task | None = None
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def log(self, kind: str, payload: dict[str, Any]) -> None:
        entry = {
            "id": str(uuid.uuid4()),
            "at": datetime.now(UTC).isoformat(),
            "kind": kind,
            "payload": payload,
        }
        self.session_log.append(entry)
        if len(self.session_log) > SESSION_LOG_MAX:
            self.session_log = self.session_log[-SESSION_LOG_MAX:]

    def pyth_age_s(self, now: datetime) -> float | None:
        """Effective staleness of the Pyth price for both the dashboard and
        the engine — a confirmed-closed underlying market (e.g. XAU/USD's
        daily halt) counts as stale even if messages keep arriving, since a
        receive-time gap alone can't detect a frozen/heartbeat price."""
        age = (now - self.pyth_receive_time).total_seconds() if self.pyth_receive_time else None
        if self.pyth_market_open is False:
            floor = self.settings.engine.stale_s + 1.0
            age = floor if age is None else max(age, floor)
        return age

    def payload(self) -> dict[str, Any]:
        now = datetime.now(UTC)
        pyth_age_s = self.pyth_age_s(now)
        kalshi_age_s = (
            (now - self.kalshi_receive_time).total_seconds() if self.kalshi_receive_time else None
        )
        return _build_state_payload(
            window=self.window,
            latest_price=self.latest_price,
            book=self.book,
            fair=self.fair,
            engine_state=self.engine_state,
            fees_cfg=self.settings.fees,
            pyth_age_s=pyth_age_s,
            kalshi_age_s=kalshi_age_s,
            stale_s=self.settings.engine.stale_s,
            kill_switch=self.kill_switch,
            session_log_tail=self.session_log[-SESSION_LOG_SENT_TO_CLIENT:],
            settlement_source_ok=self.settlement_source_ok,
            pyth_market_open=self.pyth_market_open,
        )

    async def broadcast(self) -> None:
        payload = self.payload()
        dead: list[WebSocket] = []
        for ws in self.connections:
            try:
                await ws.send_json(payload)
            except Exception:  # noqa: BLE001 - a dead socket shouldn't kill the loop
                dead.append(ws)
        for ws in dead:
            self.connections.discard(ws)


async def _recompute_and_step(app: AppState) -> None:
    if app.window is None or app.window.s0 is None or app.latest_price is None or app.book is None:
        await app.broadcast()
        return

    now = datetime.now(UTC)
    _pyth_age = app.pyth_age_s(now)
    pyth_age_s = _pyth_age if _pyth_age is not None else 999.0
    kalshi_age_s = (
        (now - app.kalshi_receive_time).total_seconds() if app.kalshi_receive_time else 999.0
    )
    tau_minutes = app.window.seconds_left(now) / 60.0
    app.fair = compute_fair_value(
        app.latest_price,
        float(app.window.s0),
        app.vol_tracker.sigma_per_minute,
        tau_minutes,
        app.settings.model.min_fair_value,
        app.settings.model.max_fair_value,
    )

    async with app._lock:
        if app.kill_switch and app.engine_state.position_state is PositionState.FLAT:
            await app.broadcast()
            return

        snapshot = MarketSnapshot(
            now=now,
            window=app.window,
            book=app.book,
            fair=app.fair,
            pyth_age_s=pyth_age_s,
            kalshi_age_s=kalshi_age_s,
            short_horizon_sigma_per_minute=app.vol_tracker.short_horizon_sigma_per_minute,
            underlying_price=app.latest_price,
        )
        result = step(
            app.engine_state,
            snapshot,
            app.settings.engine,
            app.settings.fees,
            app.settings.volatility.vol_spike_limit,
        )
        app.engine_state = result.state

        for finalized in result.finalized_signals:
            if app.recorder is not None:
                await app.recorder.update_signal_status(finalized.id, finalized.status.value)
            app.log(kind="expired", payload=_signal_to_json(finalized))

        if result.signal is not None:
            if app.recorder is not None:
                await app.recorder.record_signal(result.signal)
            app.log(kind="signal_issued", payload=_signal_to_json(result.signal))

        if result.note:
            app.log(kind="note", payload={"note": result.note})

    await app.broadcast()


async def _pyth_loop(app: AppState) -> None:
    client = PythFeedClient(
        app.settings.pyth.hermes_base, app.pyth_api_key, app.settings.pyth.price_feed_id
    )
    async for tick in client.ticks():
        app.latest_price = tick.price
        app.pyth_receive_time = tick.receive_time
        if app.vol_tracker is not None:
            app.vol_tracker.update(tick.price, tick.publish_time)
        if app.recorder is not None:
            await app.recorder.record_tick(tick)
        await _recompute_and_step(app)


async def _kalshi_book_loop(app: AppState, window_ticker: str) -> None:
    client = KalshiWsClient(app.settings.kalshi.ws_url, app.signer, window_ticker)
    async for book in client.snapshots():
        app.book = book
        app.kalshi_receive_time = book.receive_time
        if app.recorder is not None:
            await app.recorder.record_book_snapshot(book)
        await _recompute_and_step(app)


async def _ticker_loop(app: AppState) -> None:
    """Re-evaluates step() on a fixed cadence so time-based transitions
    (entry/exit cutoffs, signal TTL expiry) fire even when no new tick or
    book update has arrived recently."""
    while True:
        await asyncio.sleep(0.5)
        await _recompute_and_step(app)


async def _pyth_market_hours_loop(app: AppState) -> None:
    """Polls whether the configured Pyth feed's underlying market is open.
    Market hours don't flip every second, so a slow poll is fine — this
    exists purely to catch the daily halt independent of tick-arrival gaps
    (see the pyth_age_s override in _recompute_and_step)."""
    while True:
        try:
            is_open = await get_market_hours(
                app.settings.pyth.hermes_base,
                app.settings.pyth.price_feed_id,
                app.settings.pyth.price_feed_query,
            )
            if is_open is None:
                logger.warning(
                    "Could not find feed %s while checking market hours; leaving prior state",
                    app.settings.pyth.price_feed_id,
                )
            else:
                if app.pyth_market_open is True and is_open is False:
                    app.log(kind="warning", payload={"warning": "pyth feed market now closed"})
                elif app.pyth_market_open is False and is_open is True:
                    app.log(kind="note", payload={"note": "pyth feed market reopened"})
                app.pyth_market_open = is_open
        except Exception as exc:  # noqa: BLE001 - must not let this task die
            logger.warning("Pyth market-hours check failed (%s)", exc)
        await asyncio.sleep(60)


async def _handle_window_closed(app: AppState) -> None:
    """Fetch the just-closed window's settlement, reconcile any position
    still open into realized P&L (see settle_position_pnl — this is the
    fix for held-to-settlement trades never being counted), record it, and
    reset per-window engine state for the new window."""
    assert app.rest_client is not None
    closed_window = app.window
    assert closed_window is not None

    closed_market = await app.rest_client.get_market(closed_window.ticker)
    result = closed_market.get("result") or None
    # settlement_timer_seconds is ~1s per Kalshi, but give it a little more
    # room in case our poll caught the rollover right at the boundary.
    for _ in range(3):
        if result:
            break
        await asyncio.sleep(1.0)
        closed_market = await app.rest_client.get_market(closed_window.ticker)
        result = closed_market.get("result") or None

    position = app.engine_state.position
    if app.engine_state.position_state is not PositionState.FLAT and position is not None:
        if result:
            pnl = settle_position_pnl(position, result, app.settings.fees)
            app.engine_state = replace(
                app.engine_state, realized_pnl_today=app.engine_state.realized_pnl_today + pnl
            )
            app.log(
                kind="settled",
                payload={
                    "window_ticker": closed_window.ticker,
                    "side": position.side.value,
                    "result": result,
                    "pnl": float(pnl),
                },
            )
        else:
            app.log(
                kind="warning",
                payload={
                    "warning": "position open at window rollover with no settlement result "
                    "available yet — this trade's P&L was NOT recorded automatically",
                    "window_ticker": closed_window.ticker,
                },
            )

    if app.recorder is not None:
        settlement_ts = closed_market.get("settlement_ts")
        await app.recorder.record_settlement(
            closed_window.ticker,
            closed_market,
            result,
            datetime.fromisoformat(settlement_ts.replace("Z", "+00:00")) if settlement_ts else None,
        )
    if app._kalshi_task is not None:
        app._kalshi_task.cancel()
    # Round-trip cap is per window; realized P&L carries across windows
    # within the same UTC day (the daily reset itself happens in
    # _window_rollover_loop once the next window's date is known).
    app.engine_state = EngineState(realized_pnl_today=app.engine_state.realized_pnl_today)
    app.book = None


async def _window_rollover_loop(app: AppState) -> None:
    """Discovers/advances the current window. Wrapped in try/except with
    backoff so a transient Kalshi API error (rate limit, timeout, brief
    5xx) logs and retries instead of silently killing window discovery for
    the rest of the session."""
    assert app.rest_client is not None
    series_ticker = app.settings.kalshi.series_ticker

    backoff = 1.0
    while True:
        try:
            matched = await check_settlement_source(
                app.rest_client, series_ticker, app.settings.pyth.price_feed_symbol
            )
            app.settlement_source_ok = matched
            app.log(
                kind="note",
                payload={"note": f"settlement_source_check: {'ok' if matched else 'mismatch'}"},
            )
            break
        except Exception as exc:  # noqa: BLE001 - must not let this task die
            logger.warning("Settlement-source check failed (%s); retrying in %.1fs", exc, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30.0)

    backoff = 1.0
    while True:
        try:
            window = await get_current_window(app.rest_client, series_ticker)
            if window is None:
                next_window = await get_next_window(app.rest_client, series_ticker)
                if next_window is None:
                    await asyncio.sleep(10)
                else:
                    wait_s = max(0.0, (next_window.open_time - datetime.now(UTC)).total_seconds())
                    await asyncio.sleep(min(wait_s + 1, 30))
                backoff = 1.0
                continue

            if app.window is None or window.ticker != app.window.ticker:
                if app.window is not None:
                    await _handle_window_closed(app)
                app.window = window
                new_date = window.open_time.date()
                if app.realized_pnl_date is not None and new_date != app.realized_pnl_date:
                    app.engine_state = replace(app.engine_state, realized_pnl_today=Decimal("0"))
                    app.log(
                        kind="note",
                        payload={
                            "note": f"new trading day ({new_date.isoformat()} UTC); "
                            "realized_pnl_today reset to 0"
                        },
                    )
                app.realized_pnl_date = new_date
                if app.recorder is not None:
                    await app.recorder.record_window(window)
                app.log(
                    kind="window",
                    payload={
                        "ticker": window.ticker,
                        "s0": float(window.s0) if window.s0 else None,
                    },
                )
                app._kalshi_task = asyncio.create_task(_kalshi_book_loop(app, window.ticker))

            backoff = 1.0
            await asyncio.sleep(5)
        except Exception as exc:  # noqa: BLE001 - must not let this task die
            logger.warning("Window rollover loop error (%s); retrying in %.1fs", exc, backoff)
            app.log(kind="warning", payload={"warning": f"window rollover error: {exc}"})
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30.0)


class FillRequest(BaseModel):
    signal_id: str
    action: str  # "BUY" | "SELL"
    price: float
    size: float | None = None


class SkipRequest(BaseModel):
    signal_id: str


class KillSwitchRequest(BaseModel):
    enabled: bool


class PromoteRequest(BaseModel):
    proposal_id: str


def create_app(settings: Settings, signer: KalshiSigner, pyth_api_key: str) -> FastAPI:
    app_state = AppState(settings=settings, signer=signer, pyth_api_key=pyth_api_key)

    async def lifespan(_: FastAPI):
        app_state.recorder = Recorder(settings.sqlite_path)
        app_state.rest_client = KalshiRestClient(settings.kalshi.rest_base, signer)
        app_state.vol_tracker = VolatilityTracker(
            half_life_s=settings.volatility.ewma_half_life_s,
            short_horizon_s=settings.volatility.short_horizon_s,
            min_sigma_per_minute=settings.volatility.min_sigma_per_minute,
        )
        tasks = [
            asyncio.create_task(_window_rollover_loop(app_state)),
            asyncio.create_task(_pyth_loop(app_state)),
            asyncio.create_task(_ticker_loop(app_state)),
            asyncio.create_task(_pyth_market_hours_loop(app_state)),
        ]
        try:
            yield
        finally:
            for t in tasks:
                t.cancel()
            if app_state._kalshi_task is not None:
                app_state._kalshi_task.cancel()
            if app_state.rest_client is not None:
                await app_state.rest_client.aclose()
            if app_state.recorder is not None:
                app_state.recorder.close()

    fastapi_app = FastAPI(lifespan=lifespan)

    @fastapi_app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return DASHBOARD_PATH.read_text(encoding="utf-8")

    @fastapi_app.get("/api/state")
    async def get_state() -> JSONResponse:
        return JSONResponse(app_state.payload())

    @fastapi_app.websocket("/ws")
    async def ws_endpoint(websocket: WebSocket) -> None:
        await websocket.accept()
        app_state.connections.add(websocket)
        try:
            await websocket.send_json(app_state.payload())
            while True:
                # The client sends nothing; this just keeps the socket open.
                await websocket.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            app_state.connections.discard(websocket)

    @fastapi_app.post("/api/fill")
    async def post_fill(req: FillRequest) -> JSONResponse:
        pending = app_state.engine_state.pending_signal
        if pending is None or pending.id != req.signal_id:
            return JSONResponse({"error": "no matching pending signal"}, status_code=409)
        size = Decimal(str(req.size)) if req.size is not None else pending.size
        fill = Fill(
            signal_id=pending.id,
            window_ticker=pending.window_ticker,
            side=pending.side,
            action=Action(req.action),
            price=Decimal(str(req.price)),
            size=size,
            logged_at=datetime.now(UTC),
        )
        async with app_state._lock:
            app_state.engine_state = apply_fill(
                app_state.engine_state, fill, app_state.settings.engine, app_state.settings.fees
            )
        if app_state.recorder is not None:
            await app_state.recorder.record_fill(fill)
            await app_state.recorder.update_signal_status(pending.id, "FILLED")
        app_state.log(
            kind="filled",
            payload={"signal_id": pending.id, "price": req.price, "size": float(size)},
        )
        await app_state.broadcast()
        return JSONResponse({"ok": True})

    @fastapi_app.post("/api/skip")
    async def post_skip(req: SkipRequest) -> JSONResponse:
        pending = app_state.engine_state.pending_signal
        if pending is None or pending.id != req.signal_id:
            return JSONResponse({"error": "no matching pending signal"}, status_code=409)
        fill = Fill(
            signal_id=pending.id,
            window_ticker=pending.window_ticker,
            side=pending.side,
            action=pending.action,
            price=pending.limit_price,
            size=pending.size,
            logged_at=datetime.now(UTC),
            is_skip=True,
        )
        async with app_state._lock:
            app_state.engine_state = apply_skip(app_state.engine_state)
        if app_state.recorder is not None:
            await app_state.recorder.record_fill(fill)
            await app_state.recorder.update_signal_status(pending.id, "SKIPPED")
        app_state.log(kind="skipped", payload={"signal_id": pending.id})
        await app_state.broadcast()
        return JSONResponse({"ok": True})

    @fastapi_app.post("/api/kill_switch")
    async def post_kill_switch(req: KillSwitchRequest) -> JSONResponse:
        app_state.kill_switch = req.enabled
        app_state.log(kind="kill_switch", payload={"enabled": req.enabled})
        await app_state.broadcast()
        return JSONResponse({"ok": True})

    # ----------------------------------------------------------------
    # Review / Learning tabs. Read-only queries over already-graded
    # recorder data (see learning/queries.py) plus the promote/rollback
    # actions (learning/actions.py) -- no learning computation happens on
    # these routes, only on `learn` (CLI). `/api/version_status` is the
    # one exception polled from the Live tab, and it's a cheap read, not
    # a computation, per CLAUDE.md's "no learning computation in the live
    # path."
    # ----------------------------------------------------------------

    @fastapi_app.get("/api/review")
    async def get_review(start: str | None = None, end: str | None = None) -> JSONResponse:
        start_dt = datetime.fromisoformat(start) if start else None
        end_dt = datetime.fromisoformat(end) if end else None
        return JSONResponse(_review_payload(settings.sqlite_path, start_dt, end_dt))

    @fastapi_app.get("/api/learning")
    async def get_learning() -> JSONResponse:
        return JSONResponse(_learning_payload(settings.sqlite_path))

    @fastapi_app.get("/api/version_status")
    async def get_version_status() -> JSONResponse:
        """Polled by the Live tab for its small version badge + drift
        banner -- cheap reads only, no refitting."""
        return JSONResponse(_version_status_payload(settings.sqlite_path))

    @fastapi_app.post("/api/learning/promote")
    async def post_promote(req: PromoteRequest) -> JSONResponse:
        outcome = promote_proposal(settings.sqlite_path, req.proposal_id, settings.learning)
        if not outcome.found:
            return JSONResponse({"error": "no such proposal"}, status_code=404)
        return JSONResponse(_promote_outcome_payload(outcome))

    @fastapi_app.post("/api/learning/rollback")
    async def post_rollback() -> JSONResponse:
        restored = rollback_config(settings.sqlite_path)
        if restored is None:
            return JSONResponse({"error": "no promoted config version to roll back from"})
        return JSONResponse({"restored_params": restored.param_changes})

    return fastapi_app
