"""CLI entry point. Phase 1: `record` and `validate-feed`. Phase 2 adds
`fair-value`. Phase 4 adds `live` (the FastAPI dashboard server). Phase 5
adds `backtest`. Phase 6 adds `learn`, `review`, `proposals`, `promote`,
and `rollback`.
"""

from __future__ import annotations

import argparse
import asyncio
import random
import sqlite3
import sys
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import httpx
import numpy as np

from gold_edge.backtest.replay import load_recorded_data, replay_all, run_sweep
from gold_edge.backtest.report import format_report, format_sweep_report
from gold_edge.config import Settings, get_settings
from gold_edge.feeds.gold_proxy import ProxyFeedClient
from gold_edge.feeds.kalshi_auth import KalshiSigner
from gold_edge.feeds.kalshi_rest import KalshiRestClient
from gold_edge.feeds.kalshi_ws import KalshiWsClient
from gold_edge.feeds.pyth import PythFeedClient
from gold_edge.learning.actions import (
    promote_model,
    promote_proposal,
    rollback_config,
    rollback_model,
)
from gold_edge.learning.blend import fit_blend, format_blend, walk_forward_blend
from gold_edge.learning.blend_shadow import (
    BlendArtifact,
    PriceState,
    ShadowEngine,
    ShadowStore,
    ShadowWindow,
    make_artifact,
    parse_market_quote,
)
from gold_edge.learning.calibrator import fit_isotonic_calibrator, should_promote_calibrator
from gold_edge.learning.delay_profile import FillLatencySample
from gold_edge.learning.feature_blend import (
    BASE,
    fit_and_score,
    paired_diff,
    split_dev_holdout,
    walk_forward,
)
from gold_edge.learning.historical_gold import (
    DailyLaggedSeries,
    PriceBar,
    build_calibration_points,
    build_daily_lagged_return_series,
    evaluate_historical_calibration,
    fetch_yahoo_chart,
    fit_gld_session_vol_multipliers,
    format_historical_report,
    format_vol_multiplier_report,
    format_walkforward_report,
    run_walkforward_rounds,
    summarize_realized_vol,
)
from gold_edge.learning.history import (
    COINBASE_BASE,
    KALSHI_BASE,
    backfill_candles,
    backfill_kraken_paxg,
    backfill_paxg,
    backfill_windows,
    densify,
    load_kraken_paxg_bars,
    load_paxg_bars,
    merge_consensus,
    open_history,
    parse_coinbase_candles,
)
from gold_edge.learning.history import _get as _history_get
from gold_edge.learning.insights import session_report_card, weekly_report
from gold_edge.learning.market_study import (
    BookQuote,
    GldSeries,
    WindowRef,
    _cluster_bootstrap,
    build_study_rows,
    compare_brier,
    format_study,
    hold_to_settlement,
    load_gld_cache,
    proxy_agreement,
    save_gld_cache,
)
from gold_edge.learning.model_proposer import build_model_proposal, load_shadow_stats
from gold_edge.learning.opportunities import filter_scorecard
from gold_edge.learning.patterns import load_macro_events
from gold_edge.learning.pipeline import run_learning_pipeline
from gold_edge.learning.queries import load_proposals, load_session_data
from gold_edge.learning.registry import RejectedProposal, evaluate_model_promotion_gates
from gold_edge.model.fair_value import compute_fair_value
from gold_edge.model.volatility import VolatilityTracker
from gold_edge.models import BookSnapshot, Window
from gold_edge.recorder import SCHEMA, Recorder
from gold_edge.windows import check_settlement_source, get_current_window, get_next_window


def _build_signer(settings: Settings) -> KalshiSigner:
    if not settings.kalshi_api_key_id or not settings.kalshi_private_key_path:
        print(
            "Missing KALSHI_API_KEY_ID / KALSHI_PRIVATE_KEY_PATH. Copy .env.example to "
            ".env and fill them in.",
            file=sys.stderr,
        )
        sys.exit(1)
    key_path = settings.kalshi_private_key_full_path
    if not key_path.exists():
        print(f"Kalshi private key not found at {key_path}", file=sys.stderr)
        sys.exit(1)
    return KalshiSigner(settings.kalshi_api_key_id, key_path)


def _require_pyth_key(settings: Settings) -> str:
    if not settings.pyth_api_key:
        print(
            "Missing PYTH_API_KEY. Hermes requires it since the Aug 2026 upgrade — "
            "see docs/contract_notes.md for how to get one.",
            file=sys.stderr,
        )
        sys.exit(1)
    return settings.pyth_api_key


async def _supervised(name: str, make_loop, max_backoff_s: float = 30.0) -> None:
    """Keeps a recording loop alive. A bare `create_task` that raises just
    ends silently, and the rest of `record` keeps running as if nothing
    happened -- which left days of Kalshi books recorded with no price
    ticks at all. Any non-cancellation error is logged and the loop is
    restarted with backoff."""
    backoff = 1.0
    while True:
        try:
            await make_loop()
            return
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - a feed task must never die silently
            print(f"[{name}] recording loop crashed ({exc!r}); restarting in {backoff:.1f}s")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, max_backoff_s)


async def _pyth_recording_loop(client: PythFeedClient, recorder: Recorder) -> None:
    count = 0
    async for tick in client.ticks():
        await recorder.record_tick(tick)
        count += 1
        if count % 10 == 0:
            print(f"[pyth]   {tick.publish_time.isoformat()}  price={tick.price:.2f}  (n={count})")


async def _proxy_recording_loop(client: ProxyFeedClient, recorder: Recorder) -> None:
    """Always-on 24/7 fallback recording (see feeds/gold_proxy.py) -- run
    concurrently with Pyth, not just when Pyth is stale, so `learn` has a
    continuous proxy series to reconstruct the basis from later, and so the
    daily halt / weekends aren't a total recording gap."""
    count = 0
    async for tick in client.ticks():
        await recorder.record_tick(tick)
        count += 1
        if count % 10 == 0:
            print(f"[proxy]  {tick.publish_time.isoformat()}  price={tick.price:.2f}  (n={count})")


async def _kalshi_recording_loop(
    ws_url: str, signer: KalshiSigner, window_ticker: str, recorder: Recorder
) -> None:
    client = KalshiWsClient(ws_url, signer, window_ticker)
    count = 0
    async for book in client.snapshots():
        await recorder.record_book_snapshot(book)
        count += 1
        if count % 10 == 0:
            print(
                f"[kalshi] {window_ticker}  yes={book.yes_bid}/{book.yes_ask}  "
                f"no={book.no_bid}/{book.no_ask}  (n={count})"
            )


async def _run_record(settings: Settings) -> None:
    signer = _build_signer(settings)
    pyth_api_key = _require_pyth_key(settings)
    recorder = Recorder(settings.sqlite_path)
    print(f"Recording to {settings.sqlite_path}")

    async with KalshiRestClient(settings.kalshi.rest_base, signer) as rest:
        matched = await check_settlement_source(
            rest, settings.kalshi.series_ticker, settings.pyth.price_feed_symbol
        )
        print(f"Settlement source check: {'OK' if matched else 'MISMATCH (see warning above)'}")

        pyth_client = PythFeedClient(
            settings.pyth.hermes_base, pyth_api_key, settings.pyth.price_feed_id
        )
        pyth_task = asyncio.create_task(
            _supervised("pyth", lambda: _pyth_recording_loop(pyth_client, recorder))
        )

        proxy_client = ProxyFeedClient(settings.gold_proxy.ws_url, settings.gold_proxy.product_id)
        proxy_task = asyncio.create_task(
            _supervised("proxy", lambda: _proxy_recording_loop(proxy_client, recorder))
        )

        current_window: Window | None = None
        kalshi_task: asyncio.Task | None = None
        backoff = 1.0
        try:
            while True:
                try:
                    window = await get_current_window(rest, settings.kalshi.series_ticker)
                    if window is None:
                        next_window = await get_next_window(rest, settings.kalshi.series_ticker)
                        if next_window is None:
                            print("No open or upcoming KXGOLD15M market found; retrying in 10s")
                            await asyncio.sleep(10)
                            continue
                        wait_s = max(
                            0.0, (next_window.open_time - datetime.now(UTC)).total_seconds()
                        )
                        print(f"Next window {next_window.ticker} opens in {wait_s:.0f}s; waiting")
                        await asyncio.sleep(min(wait_s + 1, 30))
                        continue

                    if current_window is None or window.ticker != current_window.ticker:
                        if current_window is not None:
                            closed_market = await rest.get_market(current_window.ticker)
                            await recorder.record_settlement(
                                current_window.ticker,
                                closed_market,
                                closed_market.get("result") or None,
                                datetime.fromisoformat(
                                    closed_market["settlement_ts"].replace("Z", "+00:00")
                                )
                                if closed_market.get("settlement_ts")
                                else None,
                            )
                            if kalshi_task is not None:
                                kalshi_task.cancel()
                        await recorder.record_window(window)
                        print(f"Window {window.ticker}: S0={window.s0}  close={window.close_time}")
                        kalshi_task = asyncio.create_task(
                            _kalshi_recording_loop(
                                settings.kalshi.ws_url, signer, window.ticker, recorder
                            )
                        )
                        current_window = window

                    backoff = 1.0
                    await asyncio.sleep(5)
                except Exception as exc:  # noqa: BLE001 - a network hiccup must not kill recording
                    print(f"Window polling error ({exc}); retrying in {backoff:.1f}s")
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 30.0)
        finally:
            pyth_task.cancel()
            proxy_task.cancel()
            if kalshi_task is not None:
                kalshi_task.cancel()
            recorder.close()


def _candle_close_at(conn: sqlite3.Connection, target: datetime, tolerance_s: float = 90.0):
    row = conn.execute(
        "SELECT price, publish_time FROM ticks WHERE publish_time <= ? "
        "ORDER BY publish_time DESC LIMIT 1",
        (target.isoformat(),),
    ).fetchone()
    if row is None:
        return None
    price, publish_time_str = row
    publish_time = datetime.fromisoformat(publish_time_str)
    if (target - publish_time).total_seconds() > tolerance_s:
        return None
    return price


async def _run_validate_feed(settings: Settings, limit: int) -> None:
    async with KalshiRestClient(settings.kalshi.rest_base) as rest:
        data = await rest.get_markets(settings.kalshi.series_ticker, status="settled", limit=limit)
    markets = data.get("markets", [])
    if not markets:
        print("No settled KXGOLD15M markets returned by Kalshi.")
        return

    settings.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.sqlite_path)
    conn.executescript(SCHEMA)
    total = 0
    open_agree = 0
    close_agree = 0
    result_agree = 0
    checked = 0
    for market in markets:
        open_time = datetime.fromisoformat(market["open_time"].replace("Z", "+00:00"))
        close_time = datetime.fromisoformat(market["close_time"].replace("Z", "+00:00"))
        floor_strike = market.get("floor_strike")
        expiration_value = market.get("expiration_value")
        kalshi_result = market.get("result")
        total += 1

        our_open = _candle_close_at(conn, open_time)
        our_close = _candle_close_at(conn, close_time)
        if our_open is None or our_close is None or floor_strike is None or not expiration_value:
            continue
        checked += 1

        if abs(our_open - float(floor_strike)) < 0.01:
            open_agree += 1
        if abs(our_close - float(expiration_value)) < 0.01:
            close_agree += 1
        our_result = "yes" if our_close >= our_open else "no"
        if our_result == kalshi_result:
            result_agree += 1
        else:
            print(
                f"DISAGREEMENT {market['ticker']}: ours open={our_open} close={our_close} "
                f"-> {our_result}; kalshi open={floor_strike} close={expiration_value} "
                f"-> {kalshi_result}"
            )

    print(f"\n{total} settled markets fetched, {checked} had recorded ticks to check.")
    if checked:
        print(f"Open-price agreement:  {open_agree}/{checked} ({open_agree / checked:.1%})")
        print(f"Close-price agreement: {close_agree}/{checked} ({close_agree / checked:.1%})")
        print(f"Result agreement:      {result_agree}/{checked} ({result_agree / checked:.1%})")
    else:
        print(
            "No overlap between recorded ticks and these settled windows — run `record` "
            "for a while first, then re-run validate-feed."
        )


async def _run_fair_value(settings: Settings) -> None:
    """Print live fair value vs Kalshi bid/ask once a second, for sanity-
    checking the model against a real window (Phase 2)."""
    signer = _build_signer(settings)
    pyth_api_key = _require_pyth_key(settings)

    async with KalshiRestClient(settings.kalshi.rest_base, signer) as rest:
        window = await get_current_window(rest, settings.kalshi.series_ticker)
        if window is None or window.s0 is None:
            print("No active KXGOLD15M window with a known S0 right now.")
            return
        print(f"Window {window.ticker}  S0={window.s0}  closes {window.close_time.isoformat()}")

        tracker = VolatilityTracker(
            half_life_s=settings.volatility.ewma_half_life_s,
            short_horizon_s=settings.volatility.short_horizon_s,
            min_sigma_per_minute=settings.volatility.min_sigma_per_minute,
        )
        state: dict[str, float | BookSnapshot | None] = {"price": None, "book": None}

        pyth_client = PythFeedClient(
            settings.pyth.hermes_base, pyth_api_key, settings.pyth.price_feed_id
        )

        async def pyth_loop() -> None:
            async for tick in pyth_client.ticks():
                state["price"] = tick.price
                tracker.update(tick.price, tick.publish_time)

        async def kalshi_loop() -> None:
            ws_client = KalshiWsClient(settings.kalshi.ws_url, signer, window.ticker)
            async for book in ws_client.snapshots():
                state["book"] = book

        pyth_task = asyncio.create_task(pyth_loop())
        kalshi_task = asyncio.create_task(kalshi_loop())
        try:
            while True:
                await asyncio.sleep(1.0)
                now = datetime.now(UTC)
                tau_minutes = window.seconds_left(now) / 60.0
                price = state["price"]
                if price is None:
                    print("waiting for Pyth ticks...")
                    continue
                fv = compute_fair_value(
                    price,
                    float(window.s0),
                    tracker.sigma_per_minute,
                    tau_minutes,
                    settings.model.min_fair_value,
                    settings.model.max_fair_value,
                )
                book = state["book"]
                book_str = "no book yet"
                if book is not None:
                    book_str = f"yes {book.yes_bid}/{book.yes_ask}  no {book.no_bid}/{book.no_ask}"
                print(
                    f"S={price:.2f}  S0={window.s0}  tau={tau_minutes:.2f}m  "
                    f"sigma={tracker.sigma_per_minute:.5f}  "
                    f"fair_yes={fv.yes:.3f}  fair_no={fv.no:.3f}  {book_str}"
                )
        finally:
            pyth_task.cancel()
            kalshi_task.cancel()


def _parse_sweep_param(spec: str) -> tuple[str, list[float]]:
    key, _, values = spec.partition("=")
    if not key or not values:
        print(f"Invalid --sweep-param {spec!r}; expected key=v1,v2,...", file=sys.stderr)
        sys.exit(1)
    try:
        return key, [float(v) for v in values.split(",")]
    except ValueError:
        print(f"Invalid --sweep-param {spec!r}; values must be numbers", file=sys.stderr)
        sys.exit(1)


def _run_backtest(
    settings: Settings,
    start: str | None,
    end: str | None,
    seed: int | None,
    bucket_width: float,
    sweep_params: list[str] | None,
    train_end: str | None,
) -> None:
    start_dt = datetime.fromisoformat(start).replace(tzinfo=UTC) if start else None
    end_dt = datetime.fromisoformat(end).replace(tzinfo=UTC) if end else None

    if not settings.sqlite_path.exists():
        print(f"No recorded data found at {settings.sqlite_path}. Run `record` or `live` first.")
        sys.exit(1)

    ticks, windows = load_recorded_data(settings.sqlite_path, start_dt, end_dt)
    if not windows:
        print(
            "No windows with a known S0 found in the recorded data for that range "
            "(a window needs to have actually opened, and have book snapshots recorded, "
            "to be replayable)."
        )
        return

    print(f"Loaded {len(ticks)} ticks across {len(windows)} window(s).")

    if sweep_params:
        if not train_end:
            print("--sweep-param requires --train-end (the training/held-out split date).")
            sys.exit(1)
        train_end_dt = datetime.fromisoformat(train_end).replace(tzinfo=UTC)
        train_windows = [w for w in windows if w.window.close_time <= train_end_dt]
        test_windows = [w for w in windows if w.window.open_time > train_end_dt]
        if not train_windows or not test_windows:
            print(
                "--train-end leaves no windows on one side of the split; pick a date "
                "that falls strictly between the earliest and latest recorded window."
            )
            sys.exit(1)
        train_ticks = [t for t in ticks if t.receive_time <= train_end_dt]
        test_ticks = [t for t in ticks if t.receive_time > train_end_dt]
        param_grid = dict(_parse_sweep_param(spec) for spec in sweep_params)
        print(
            f"Sweeping {len(param_grid)} parameter(s) — {len(train_windows)} training "
            f"window(s), {len(test_windows)} held-out."
        )
        candidates = run_sweep(
            train_ticks=train_ticks,
            train_windows=train_windows,
            test_ticks=test_ticks,
            test_windows=test_windows,
            base_engine_cfg=settings.engine,
            fees_cfg=settings.fees,
            vol_cfg=settings.volatility,
            model_cfg=settings.model,
            backtest_cfg=settings.backtest,
            param_grid=param_grid,
            seed=seed,
        )
        print()
        print(format_sweep_report(candidates))
        return

    result = replay_all(
        ticks=ticks,
        windows=windows,
        engine_cfg=settings.engine,
        fees_cfg=settings.fees,
        vol_cfg=settings.volatility,
        model_cfg=settings.model,
        backtest_cfg=settings.backtest,
        rng=random.Random(seed),
    )
    print()
    print(format_report(result, bucket_width=bucket_width))


def _run_live(settings: Settings, host: str, port: int) -> None:
    import uvicorn

    from gold_edge.server import create_app

    signer = _build_signer(settings)
    pyth_api_key = _require_pyth_key(settings)
    app = create_app(settings, signer, pyth_api_key)
    uvicorn.run(app, host=host, port=port, log_level="info")


def _load_fill_latency_samples(sqlite_path: Path) -> list[FillLatencySample]:
    """CLAUDE.md's delay_profile.py input: real signal-to-fill latency from
    logged fills, joined against the signal that produced them.

    A signal the user never touched at all -- no fill, no explicit skip --
    only ever gets its `signals.status` updated to EXPIRED (see
    server.py's finalized-signal handling); it has no row in `fills`. Those
    are exactly the "missed by pure inaction" signals CLAUDE.md's "learn
    which signal types the user tends to miss" is asking about, so they're
    pulled in separately here (no real delay to measure, but they still
    count toward miss_rate_by_reason)."""
    conn = sqlite3.connect(sqlite_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT f.signal_id, f.action, s.reason, f.logged_at, s.created_at, s.status "
            "FROM fills f JOIN signals s ON f.signal_id = s.id WHERE f.signal_id IS NOT NULL"
        ).fetchall()
        expired_rows = conn.execute(
            "SELECT id, action, reason FROM signals WHERE status = 'EXPIRED' "
            "AND id NOT IN (SELECT signal_id FROM fills WHERE signal_id IS NOT NULL)"
        ).fetchall()
    finally:
        conn.close()

    samples = []
    for row in rows:
        created_at = datetime.fromisoformat(row["created_at"])
        logged_at = datetime.fromisoformat(row["logged_at"])
        delay_s = (logged_at - created_at).total_seconds()
        was_missed = row["status"] in ("EXPIRED", "SKIPPED", "MISSED")
        samples.append(
            FillLatencySample(row["signal_id"], row["action"], row["reason"], delay_s, was_missed)
        )
    for row in expired_rows:
        samples.append(FillLatencySample(row["id"], row["action"], row["reason"], None, True))
    return samples


def _load_rejected_proposals(sqlite_path: Path) -> list[RejectedProposal]:
    """Feeds registry.is_in_reproposal_cooldown via the pipeline -- CLAUDE.md's
    anti-overfitting rule needs to know what's already been tried and turned
    down, not just what's pending now."""
    return [
        RejectedProposal(param_changes=p.param_changes, rejected_at=p.created_at, reasons=[])
        for p, status in load_proposals(sqlite_path)
        if status == "rejected"
    ]


def _run_learn(settings: Settings, start: str | None, end: str | None, seed: int | None) -> None:
    if not settings.sqlite_path.exists():
        print(f"No recorded data found at {settings.sqlite_path}. Run `record` or `live` first.")
        sys.exit(1)

    start_dt = datetime.fromisoformat(start).replace(tzinfo=UTC) if start else None
    end_dt = datetime.fromisoformat(end).replace(tzinfo=UTC) if end else None
    ticks, windows = load_recorded_data(settings.sqlite_path, start_dt, end_dt)
    if not windows:
        print("No windows with a known S0 found in that range.")
        return

    fill_samples = _load_fill_latency_samples(settings.sqlite_path)
    macro_events = load_macro_events(settings.sqlite_path.parent / "macro_events.json")
    rejected_proposals = _load_rejected_proposals(settings.sqlite_path)

    print(f"Running learning pipeline over {len(windows)} window(s), {len(ticks)} tick(s)...")
    result = run_learning_pipeline(
        ticks=ticks,
        windows=windows,
        engine_cfg=settings.engine,
        fees_cfg=settings.fees,
        vol_cfg=settings.volatility,
        model_cfg=settings.model,
        backtest_cfg=settings.backtest,
        learning_cfg=settings.learning,
        fill_latency_samples=fill_samples,
        macro_events=macro_events,
        rejected_proposals=rejected_proposals,
        seed=seed,
    )

    now = datetime.now(UTC)
    recorder = Recorder(settings.sqlite_path)
    for mr in result.markouts:
        asyncio.run(recorder.record_markout(mr.signal_id, mr.markout))
    for grt in result.graded_round_trips:
        asyncio.run(
            recorder.record_grade(
                entry_signal_id=grt.entry_signal_id or "unknown",
                exit_signal_id=grt.exit_signal_id,
                window_ticker=grt.window_ticker,
                primary_grade=grt.grade.primary.value,
                tags=[t.value for t in grt.grade.tags],
                net_pnl=grt.net_pnl,
                graded_at=now,
            )
        )
    for attr in result.attribution_results:
        asyncio.run(
            recorder.record_attribution(
                attr.window_ticker, attr.cause.value, attr.dollar_impact, attr.detail, now
            )
        )
    for opp in result.opportunities:
        asyncio.run(
            recorder.record_opportunity(
                opp.window_ticker,
                opp.side.value,
                opp.at,
                opp.reason,
                opp.realistic_net_pnl,
                opp.oracle_net_pnl,
            )
        )
    for gfe in result.graded_filter_events:
        fe = gfe.filter_event
        asyncio.run(
            recorder.record_filter_event(
                fe.window_ticker, fe.side.value, fe.at, fe.reason, fe.realistic_net_pnl
            )
        )
    for stat in result.pattern_stats:
        asyncio.run(
            recorder.record_pattern_stat(
                stat.dimension,
                stat.bucket,
                stat.n,
                stat.mean_pnl,
                stat.ci_low,
                stat.ci_high,
                stat.p_value,
                stat.significant,
                now,
            )
        )
    for prop in result.proposals:
        asyncio.run(
            recorder.record_proposal(
                proposal_id=prop.id,
                param_changes=prop.param_changes,
                n_holdout_trades=prop.n_holdout_trades,
                holdout_pnl_delta=prop.holdout_pnl_delta,
                holdout_pnl_delta_ci_low=prop.holdout_pnl_delta_ci_low,
                holdout_pnl_delta_ci_high=prop.holdout_pnl_delta_ci_high,
                holdout_drawdown_delta=prop.holdout_drawdown_delta,
                rationale=prop.rationale,
                status="pending",
                created_at=prop.created_at,
            )
        )
    recorder.close()

    filter_stats = filter_scorecard([gfe.filter_event for gfe in result.graded_filter_events])
    print()
    print(
        session_report_card(
            result.graded_trades, result.attribution_results, result.opportunities, filter_stats
        )
    )
    print()
    print(
        weekly_report(
            result.pattern_stats,
            result.delay_profile,
            result.proposals,
            None,
            settings.learning.min_bucket_n,
        )
    )
    if result.calibrator is not None:
        verdict = "would improve" if result.calibrator_promoted else "would NOT improve"
        print(
            f"\nCalibrator: refit on this data {verdict} held-out log-loss/Brier vs the raw "
            "model. Not auto-applied -- see CLAUDE.md's promotion gates."
        )


def _run_review(settings: Settings, start: str | None, end: str | None) -> None:
    if not settings.sqlite_path.exists():
        print(f"No recorded data found at {settings.sqlite_path}.")
        sys.exit(1)

    start_dt = datetime.fromisoformat(start).replace(tzinfo=UTC) if start else None
    end_dt = datetime.fromisoformat(end).replace(tzinfo=UTC) if end else None
    trades, attribution, opportunities, filter_events = load_session_data(
        settings.sqlite_path, start_dt, end_dt
    )

    if not trades and not attribution and not opportunities and not filter_events:
        print("No graded data in that range yet -- run `learn` first.")
        return

    print(session_report_card(trades, attribution, opportunities, filter_scorecard(filter_events)))


def _run_proposals(settings: Settings) -> None:
    if not settings.sqlite_path.exists():
        print(f"No recorded data found at {settings.sqlite_path}.")
        sys.exit(1)
    proposals = load_proposals(settings.sqlite_path)
    if not proposals:
        print("No proposals recorded yet. Run `learn` first.")
        return
    for proposal, status in proposals:
        print(f"[{status.upper():9s}] {proposal.id}  {proposal.rationale}")


def _run_promote(settings: Settings, proposal_id: str) -> None:
    if not settings.sqlite_path.exists():
        print(f"No recorded data found at {settings.sqlite_path}.")
        sys.exit(1)

    outcome = promote_proposal(settings.sqlite_path, proposal_id, settings.learning)
    if not outcome.found:
        print(f"No proposal with id {proposal_id!r}. Run `proposals` to list them.")
        sys.exit(1)

    assert outcome.gate_result is not None
    if outcome.gate_result.passed:
        assert outcome.promoted_version is not None
        print(f"Promoted proposal {proposal_id}.")
        print("Apply these values to config.yaml's `engine:` section before the next live session:")
        for k, v in outcome.promoted_version.param_changes.items():
            print(f"  {k}: {v}")
    else:
        print(f"Rejected proposal {proposal_id} -- failed gate(s):")
        for reason in outcome.gate_result.reasons:
            print(f"  - {reason}")
        print(
            f"(shadow sessions seen: {outcome.shadow_sessions_seen}, "
            f"beats_live={outcome.shadow_beats_live})"
        )


def _run_rollback(settings: Settings) -> None:
    if not settings.sqlite_path.exists():
        print(f"No recorded data found at {settings.sqlite_path}.")
        sys.exit(1)

    restored = rollback_config(settings.sqlite_path)
    if restored is None:
        print("No promoted config version to roll back from.")
        return

    print(f"Rolled back. Now active: {restored.param_changes}")
    print("Apply these values to config.yaml's `engine:` section:")
    for k, v in restored.param_changes.items():
        print(f"  {k}: {v}")


async def _run_analyze_history(
    settings: Settings,
    daily_symbol: str,
    daily_range: str,
    intraday_symbol: str,
    intraday_interval: str,
    intraday_range: str,
) -> None:
    print(
        f"Fetching {daily_range} of daily history for {daily_symbol} from Yahoo Finance "
        "(free, keyless)..."
    )
    try:
        daily_bars = await fetch_yahoo_chart(daily_symbol, "1d", daily_range)
    except httpx.HTTPError as exc:
        print(f"  fetch failed ({exc}); skipping daily check")
        daily_bars = []
    print(f"  got {len(daily_bars)} daily bars")
    if daily_bars:
        vol_summary = summarize_realized_vol(daily_bars, settings.learning.min_bucket_n)
        # A 5-trading-day "window" (Mon open -> Fri close) so there are
        # interior daily closes (Tue/Wed/Thu) to use as the live "current"
        # price -- a 1-day window has no interior bar at all with only
        # daily granularity, which is why this isn't horizon_bars=1. See
        # build_calibration_points' docstring for why the window's own
        # close can never be used as "current".
        points = build_calibration_points(
            daily_bars,
            settings.volatility,
            settings.model,
            window_bars=5,
            max_window_minutes=10_080.0,
            round_by="year",
        )
        calibration = evaluate_historical_calibration(points)
        print()
        print(
            format_historical_report(
                f"{daily_symbol} daily bars, 5-trading-day-window calibration "
                f"({len(daily_bars)} real trading days)",
                vol_summary,
                calibration,
            )
        )
        # Multiple rounds of analysis, one per calendar year, walking
        # forward: does a calibrator fit on earlier years still help on a
        # year it has never seen? A single 80/20 split above can't answer
        # that -- it could just be one lucky split.
        print()
        print(
            format_walkforward_report(
                f"{daily_symbol} daily, year-by-year", run_walkforward_rounds(points)
            )
        )

    print()
    print(
        f"Fetching {intraday_range} of {intraday_interval} intraday history for "
        f"{intraday_symbol} from Yahoo Finance (free, keyless; NYSE hours only)..."
    )
    try:
        intraday_bars = await fetch_yahoo_chart(intraday_symbol, intraday_interval, intraday_range)
    except httpx.HTTPError as exc:
        print(f"  fetch failed ({exc}); skipping intraday check")
        intraday_bars = []
    print(f"  got {len(intraday_bars)} intraday bars")
    if intraday_bars:
        vol_summary = summarize_realized_vol(intraday_bars, settings.learning.min_bucket_n)
        # 1-minute bars, 15-bar windows == exactly Kalshi's real window
        # length, with 14 interior "current price" points per window.
        # max_window_minutes excludes any window whose open-to-close time
        # jumped across an overnight/weekend close in this NYSE-hours-only
        # series -- see historical_gold.py's module docstring.
        points = build_calibration_points(
            intraday_bars,
            settings.volatility,
            settings.model,
            window_bars=15,
            max_window_minutes=20.0,
            round_by="day",
        )
        calibration = evaluate_historical_calibration(points)
        print()
        print(
            format_historical_report(
                f"{intraday_symbol} intraday bars, 15-minute-window calibration "
                f"({len(intraday_bars)} bars, NYSE hours only)",
                vol_summary,
                calibration,
            )
        )
        # Multiple rounds of analysis, one per trading day -- these are the
        # real 15-minute sessions (the same window length and horizon
        # Kalshi's contracts use), walked forward day by day rather than
        # judged on one single split.
        print()
        print(
            format_walkforward_report(
                f"{intraday_symbol} 15-min sessions, day-by-day", run_walkforward_rounds(points)
            )
        )
        # "Learn behaviors of GLD": a real fitted sigma multiplier by vol
        # regime and time-of-day session, from real historical price
        # action -- this is CLAUDE.md's learned component #2 (volatility
        # scaling), just fit on GLD's spot-price history instead of live
        # recorded ticks because there's vastly more of it. Session
        # bucketing needs real intraday time variation, so this is only
        # meaningful on the intraday dataset, not the daily one.
        print()
        print(
            format_vol_multiplier_report(
                fit_gld_session_vol_multipliers(
                    points, settings.volatility.vol_spike_limit, settings.learning.min_bucket_n
                ),
                settings.learning.min_bucket_n,
            )
        )

    print()
    print(
        "NOTE: this is a spot-price/model-calibration check on real historical GLD\n"
        "prices, not a trading backtest -- no historical Kalshi orderbook, spread, or\n"
        "fee data exists for past dates, so it cannot produce fill/P&L numbers or a\n"
        "Proposal (that requires real recorded round trips, per CLAUDE.md). The\n"
        "walk-forward rounds above show whether a finding held up session after\n"
        "session rather than in one lucky split, and the fitted volatility-multiplier\n"
        "table is CLAUDE.md's learned component #2, fit on real GLD history -- but\n"
        "none of this is applied to live trading automatically. Use it to\n"
        "sanity-check the fair-value formula and config.yaml's volatility defaults by\n"
        "hand, not as a performance result. See docs/historical_calibration.md."
    )


def _load_settled_windows(
    conn: sqlite3.Connection, start: datetime | None, end: datetime | None
) -> list[WindowRef]:
    out = []
    for ticker, open_t, close_t, result in conn.execute(
        "SELECT w.ticker, w.open_time, w.close_time, s.result FROM windows w "
        "JOIN settlements s ON s.ticker = w.ticker WHERE s.result IN ('yes','no') "
        "ORDER BY w.open_time"
    ):
        o, c = datetime.fromisoformat(open_t), datetime.fromisoformat(close_t)
        if (start and o < start) or (end and c > end):
            continue
        out.append(WindowRef(ticker, o, c, result))
    return out


def _quote_fetcher(conn: sqlite3.Connection, max_staleness_s: float = 120.0):
    """The book at time t is the latest update at or before t -- snapshots are
    only written on change, so a quiet book simply has an older last update."""

    def quote_at(ticker: str, t: datetime) -> BookQuote | None:
        row = conn.execute(
            "SELECT yes_bid, yes_ask, no_bid, no_ask, receive_time FROM book_snapshots "
            "WHERE window_ticker = ? AND receive_time <= ? ORDER BY receive_time DESC LIMIT 1",
            (ticker, t.isoformat()),
        ).fetchone()
        if row is None:
            return None
        rt = datetime.fromisoformat(row[4])
        if (t - rt).total_seconds() > max_staleness_s:
            return None
        return BookQuote(*(Decimal(x) for x in row[:4]), rt)

    return quote_at


async def _run_backfill(settings: Settings, days: int, db: str) -> None:
    conn = open_history(Path(db))
    since = datetime.now(UTC) - timedelta(days=days)
    async with httpx.AsyncClient(timeout=30.0, headers={"User-Agent": "Mozilla/5.0"}) as client:
        n = await backfill_windows(conn, settings.kalshi.series_ticker, since, client)
        print(f"windows: +{n} settled windows since {since.date()} (public, keyless)")
        n = await backfill_candles(conn, settings.kalshi.series_ticker, client)
        print(f"candles: fetched minute quotes for {n} windows")
        lo = conn.execute("SELECT MIN(open_time) FROM kalshi_windows").fetchone()[0]
        hi = conn.execute("SELECT MAX(close_time) FROM kalshi_windows").fetchone()[0]
        if lo and hi:
            n = await backfill_paxg(
                conn,
                datetime.fromisoformat(lo) - timedelta(hours=1),
                datetime.fromisoformat(hi) + timedelta(minutes=5),
                client,
            )
            print(f"paxg: +{n} one-minute PAXG-USD closes from Coinbase (public, keyless)")
            n = await backfill_kraken_paxg(
                conn,
                datetime.fromisoformat(lo) - timedelta(hours=1),
                datetime.fromisoformat(hi) + timedelta(minutes=5),
                client,
            )
            print(f"kraken_paxg: +{n} one-minute PAXG-USD closes from Kraken (public, keyless)")
    tw = conn.execute("SELECT COUNT(*) FROM kalshi_windows").fetchone()[0]
    tc = conn.execute("SELECT COUNT(DISTINCT ticker) FROM kalshi_candles").fetchone()[0]
    print(f"history db {db}: {tw} windows, {tc} with candles")


def _history_quote_fetcher(conn: sqlite3.Connection, mode: str = "next_open"):
    """mode "next_open": the FIRST bid/ask of the minute after the signal minute
    (`open_dollars` of the next candle). mode "close": the closing bid/ask of
    the minute that just ended -- the quote actually known at the signal
    instant, so it can't contain any information from after the signal."""
    table: dict[str, dict[int, tuple[float, float]]] = {}
    col = "yes_bid_open, yes_ask_open" if mode == "next_open" else "yes_bid_close, yes_ask_close"
    for ticker, end_ts, bid_o, ask_o in conn.execute(
        f"SELECT ticker, end_ts, {col} FROM kalshi_candles"
    ):
        if bid_o is not None and ask_o is not None:
            table.setdefault(ticker, {})[end_ts] = (bid_o, ask_o)
    offset = 60 if mode == "next_open" else 0

    def quote_at(ticker: str, t: datetime) -> BookQuote | None:
        boundary = t.replace(second=0, microsecond=0)
        got = table.get(ticker, {}).get(int(boundary.timestamp()) + offset)
        if got is None:
            return None
        bid, ask = got
        if not (0.0 < bid < ask < 1.0) or ask - bid > 0.10:
            return None
        yb, ya = Decimal(str(bid)), Decimal(str(ask))
        return BookQuote(yb, ya, Decimal(1) - ya, Decimal(1) - yb, t)

    return quote_at


def _history_windows(
    conn: sqlite3.Connection, start: datetime | None = None, end: datetime | None = None
) -> tuple[list[WindowRef], dict[str, float]]:
    import math

    windows, truth = [], {}
    for ticker, o, c, s0, sv, res in conn.execute(
        "SELECT ticker, open_time, close_time, s0, settle_value, result FROM kalshi_windows "
        "ORDER BY open_time"
    ):
        ot = datetime.fromisoformat(o)
        if (start and ot < start) or (end and ot >= end):
            continue
        windows.append(WindowRef(ticker, ot, datetime.fromisoformat(c), res))
        if s0 and sv:
            truth[ticker] = math.log(sv / s0)
    return windows, truth


FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "base": BASE,
    "+late": (*BASE, "fair_x_late", "mid_x_late"),
    "+momentum": (*BASE, "mom1"),
    "+spread/activity": (*BASE, "spread", "logvol"),
    "+multi-vol": (*BASE, "logit_fair_h300", "logit_fair_h1800"),
    "+time-of-day": (*BASE, "hsin", "hcos"),
    "+paxg-momentum": (*BASE, "ret5z"),
    "+dollar": (*BASE, "dxy_lag1"),
    "+yield": (*BASE, "tnx_lag1"),
    "all": (
        *BASE, "fair_x_late", "mid_x_late", "mom1", "spread", "logvol",
        "logit_fair_h300", "logit_fair_h1800", "hsin", "hcos", "ret5z", "dxy_lag1", "tnx_lag1",
    ),
}


def _history_extras(
    conn: sqlite3.Connection,
    settings: Settings,
    series: dict[float, GldSeries],
    dxy: DailyLaggedSeries | None = None,
    tnx: DailyLaggedSeries | None = None,
):
    import math

    from gold_edge.learning.blend import _logit

    candles: dict[str, dict[int, tuple[float, float]]] = {}
    for ticker, end_ts, bid_c, ask_c, vol in conn.execute(
        "SELECT ticker, end_ts, yes_bid_close, yes_ask_close, volume FROM kalshi_candles"
    ):
        if bid_c is not None and ask_c is not None:
            candles.setdefault(ticker, {})[end_ts] = ((bid_c + ask_c) / 2.0, vol or 0.0)
    main = series[900.0]

    def extra_at(w: WindowRef, minute: int, t: datetime, fair: float) -> dict[str, float]:
        ts = int(t.timestamp())
        table = candles.get(w.ticker, {})
        cur, prev = table.get(ts), table.get(ts - 60)
        out = {"mom1": (cur[0] - prev[0]) if cur and prev else 0.0,
               "logvol": math.log1p(cur[1]) if cur else 0.0}
        hour = t.hour + t.minute / 60.0
        angle = 2 * math.pi * hour / 24
        out["hsin"], out["hcos"] = math.sin(angle), math.cos(angle)
        s, s0 = main.price_at(t), main.price_at(w.open_time)
        sigma5 = main.sigma_at(t)
        p5 = main.price_at(t - timedelta(minutes=5))
        out["ret5z"] = (
            math.log(s / p5) / (sigma5 * math.sqrt(5.0)) if s and p5 and sigma5 else 0.0
        )
        out["dxy_lag1"] = dxy.at(w.open_time.date()) if dxy is not None else 0.0
        out["tnx_lag1"] = tnx.at(w.open_time.date()) if tnx is not None else 0.0
        for half_life, key in ((300.0, "logit_fair_h300"), (1800.0, "logit_fair_h1800")):
            g = series[half_life]
            sg = g.sigma_at(t)
            out[key] = (
                _logit(
                    compute_fair_value(
                        s, s0, sg, float(15 - minute),
                        settings.model.min_fair_value, settings.model.max_fair_value,
                    ).yes
                )
                if s and s0 and sg
                else _logit(fair)
            )
        return out

    return extra_at


async def _run_study_features(settings: Settings, db: str, holdout_days: int) -> None:
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    windows, _ = _history_windows(conn)
    bars = densify(load_paxg_bars(conn))
    short = settings.volatility.short_horizon_s
    series = {h: GldSeries(bars, h, short) for h in (300.0, 900.0, 1800.0)}
    try:
        dxy_bars = await fetch_yahoo_chart("DX-Y.NYB", "1d", "2y")
        dxy = build_daily_lagged_return_series(dxy_bars)
        print(f"dollar index (DX-Y.NYB): {len(dxy_bars)} daily closes fetched (free, keyless)")
    except httpx.HTTPError as exc:
        print(f"dollar index fetch failed ({exc!r}); '+dollar'/'all' will score dxy_lag1=0.0")
        dxy = DailyLaggedSeries()
    try:
        tnx_bars = await fetch_yahoo_chart("^TNX", "1d", "2y")
        tnx = build_daily_lagged_return_series(tnx_bars)
        print(f"10y treasury yield (^TNX): {len(tnx_bars)} daily closes fetched (free, keyless)")
    except httpx.HTTPError as exc:
        print(f"treasury yield fetch failed ({exc!r}); '+yield'/'all' will score tnx_lag1=0.0")
        tnx = DailyLaggedSeries()
    rows = build_study_rows(
        windows, series[900.0], _history_quote_fetcher(conn, "close"), settings.model, 1.5,
        extra_at=_history_extras(conn, settings, series, dxy, tnx),
    )
    dev, hold = split_dev_holdout(rows, holdout_days)
    print(
        f"{len(rows)} rows: DEV {len({r.ticker for r in dev})} windows (feature choice), "
        f"HOLDOUT {len({r.ticker for r in hold})} windows (last {holdout_days} days, scored once)"
    )
    print("\n=== DEV: expanding walk-forward, each set scored on days it never trained on ===")
    dev_res = {}
    for name, feats in FEATURE_SETS.items():
        res = walk_forward(dev, feats)
        if res is None:
            continue
        dev_res[name] = res
        print(
            f"{name:18s} log-loss {res.log_loss:.4f} (market {res.market_log_loss:.4f})  "
            f"Brier vs market {res.diff_vs_market:+.5f} [{res.ci_low:+.5f}, {res.ci_high:+.5f}]"
        )
    base = dev_res["base"]
    best_name = min(dev_res, key=lambda k: dev_res[k].log_loss)
    chosen = best_name
    if best_name != "base":
        d, lo, hi = paired_diff(base, dev_res[best_name])
        print(f"\nbest on DEV: {best_name}; vs base Brier gain {d:+.5f} CI [{lo:+.5f}, {hi:+.5f}]")
        if not lo > 0:
            chosen = "base"
            print("gain not reliable -> keeping the base blend (pre-declared rule)")
    print(f"\n=== HOLDOUT (never used for any choice above): chosen = {chosen} ===")
    finals = {n: fit_and_score(dev, hold, FEATURE_SETS[n]) for n in {"base", chosen}}
    for n, res in finals.items():
        if res is None:
            continue
        print(
            f"{n:18s} log-loss {res.log_loss:.4f} (market {res.market_log_loss:.4f})  "
            f"Brier vs market {res.diff_vs_market:+.5f} [{res.ci_low:+.5f}, {res.ci_high:+.5f}]"
        )
        for th in (0.03, 0.05, 0.08):
            h = hold_to_settlement(res.rows, settings.fees, th, n_boot=500)
            if h.n_trades:
                print(
                    f"   gap>={th:.2f}: trades={h.n_trades} win={h.win_rate:.1%} "
                    f"mean net ${h.mean_pnl:+.3f} CI [${h.ci_low:+.3f}, ${h.ci_high:+.3f}]"
                )
    if chosen != "base" and finals[chosen] and finals["base"]:
        d, lo, hi = paired_diff(finals["base"], finals[chosen])
        print(f"holdout, {chosen} vs base: Brier gain {d:+.5f} CI [{lo:+.5f}, {hi:+.5f}]")


def _run_propose_model(
    settings: Settings, history_db: str, shadow_db: str, holdout_days: int, out: str, promote: bool
) -> None:
    """`gold-edge propose-model [--promote]`: the one CLI entry point for
    the probability-model promotion path (registry.py's
    `evaluate_model_promotion_gates`). Always re-fits fresh from the
    current history + shadow data and prints the full gate evaluation;
    only writes anything (a new model_versions row) when --promote is
    given AND every gate passes -- calling it with --promote IS the user
    approval, same convention `promote_proposal` already uses. Mirrors
    `promote`/`rollback`'s existing UX: it prints what to put in
    config.yaml rather than writing config.yaml itself."""
    conn = sqlite3.connect(f"file:{history_db}?mode=ro", uri=True)
    windows, _ = _history_windows(conn)
    bars = densify(load_paxg_bars(conn))
    half_life = 900.0
    gld = GldSeries(bars, half_life, settings.volatility.short_horizon_s)
    rows = build_study_rows(
        windows, gld, _history_quote_fetcher(conn, "close"), settings.model, delay_s=1.5
    )
    dev, hold = split_dev_holdout(rows, holdout_days)
    if not dev or not hold:
        print(f"not enough history for a dev/holdout split (dev={len(dev)}, holdout={len(hold)})")
        return

    holdout_result = fit_and_score(dev, hold, BASE, n_boot=1000)
    if holdout_result is None:
        print("could not score a holdout result (too little data)")
        return

    n_dev_windows = len({r.ticker for r in dev})
    model = fit_blend(dev)
    now = datetime.now(UTC)

    # Learned component #1 (CLAUDE.md): fit an isotonic calibrator on top of
    # the blend's OWN dev-set predictions (same split the blend itself was
    # fit on), then keep it only if it improves BOTH held-out Brier and
    # log-loss over the blend alone -- never applied on backtest-looking-good
    # alone, same discipline as everything else in this promotion path.
    dev_preds = model.predict(dev)
    dev_outcomes = [r.outcome for r in dev]
    holdout_preds = model.predict(hold)
    holdout_outcomes = [r.outcome for r in hold]
    calibrator = fit_isotonic_calibrator(dev_preds, dev_outcomes)
    calibrated_holdout_preds = calibrator.predict_many(holdout_preds)
    kept_calibrator = should_promote_calibrator(
        holdout_preds, calibrated_holdout_preds, holdout_outcomes
    )
    kind = "blend+isotonic" if kept_calibrator else "blend"
    print(
        f"isotonic calibrator: {'kept' if kept_calibrator else 'not kept'} "
        f"(must improve both held-out Brier and log-loss over the blend alone)"
    )

    artifact = make_artifact(
        model.weights, half_life, n_dev_windows, now,
        calibrator=calibrator if kept_calibrator else None,
    )
    artifact.save(Path(out))
    print(
        f"Fit on {n_dev_windows} dev windows ({len(dev)} rows); "
        f"weights={artifact.weights}  version={artifact.version_hash} -> {out}"
    )

    shadow_stats = load_shadow_stats(Path(shadow_db))
    proposal = build_model_proposal(
        kind, out, artifact.version_hash, BASE, n_dev_windows, holdout_result, shadow_stats, now
    )
    print()
    print(proposal.rationale)

    gate = evaluate_model_promotion_gates(
        proposal.to_gate_evidence(), settings.learning, user_approved=promote
    )
    print()
    if gate.passed:
        print("ALL GATES PASSED.")
    else:
        print("Gates NOT cleared:")
        for reason in gate.reasons:
            print(f"  - {reason}")

    if not promote:
        print("\n(run with --promote once you're ready to approve this -- nothing was written)")
        return

    # The promotion registry (config_versions/model_versions/proposals)
    # always lives in the recorder's main sqlite file, same as
    # promote_proposal/rollback_config -- history_db/shadow_db are only
    # inputs used to FIT and EVALUATE the candidate above.
    outcome = promote_model(settings.sqlite_path, proposal, settings.learning)
    if outcome.gate_result.passed:
        assert outcome.promoted_version is not None
        print(f"\nPromoted model version {outcome.promoted_version.version_hash}.")
        print("Set this in config.yaml's `learning:` section before the next `live` session:")
        print(f"  active_model_path: {out}")
    else:
        print("\nNOT promoted -- failed gate(s):")
        for reason in outcome.gate_result.reasons:
            print(f"  - {reason}")


def _run_rollback_model(settings: Settings) -> None:
    restored = rollback_model(settings.sqlite_path)
    if restored is None:
        print("No promoted model version to roll back from.")
        return
    print(f"Rolled back. Now active (if any): {restored.version_hash}")
    print(
        "Update config.yaml's `learning.active_model_path` accordingly (or unset it to run "
        "on the raw baseline)."
    )


def _run_train_blend(settings: Settings, db: str, half_life: float, out: str) -> None:
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    windows, _ = _history_windows(conn)
    gld = GldSeries(
        densify(load_paxg_bars(conn)), half_life, settings.volatility.short_horizon_s
    )
    rows = build_study_rows(
        windows, gld, _history_quote_fetcher(conn, "close"), settings.model, delay_s=1.5
    )
    model = fit_blend(rows)
    artifact = make_artifact(
        model.weights, half_life, len({r.ticker for r in rows}), datetime.now(UTC)
    )
    artifact.save(Path(out))
    a, b, c = artifact.weights
    print(
        f"Trained on {artifact.n_windows} windows / {len(rows)} rows (all history, so this "
        f"is a SHADOW candidate only -- its honest out-of-sample test is the live shadow).\n"
        f"logit(p) = {a:+.3f} + {b:.3f}*logit(market) + {c:+.3f}*logit(model)  "
        f"[half-life {half_life:.0f}s, version {artifact.version_hash}] -> {out}"
    )


async def _seed_prices(client: httpx.AsyncClient, prices: PriceState) -> int:
    end = datetime.now(UTC)
    rows = await _history_get(
        client,
        f"{COINBASE_BASE}/products/PAXG-USD/candles",
        {
            "granularity": 60,
            "start": (end - timedelta(minutes=150)).isoformat(),
            "end": end.isoformat(),
        },
    )
    bars = densify(parse_coinbase_candles(rows))
    for b in bars:
        prices.on_tick(b.close, b.timestamp + timedelta(seconds=60))
    for b in bars:
        prices.advance(b.timestamp + timedelta(seconds=60))
    return len(bars)


async def _shadow_feed(settings: Settings, prices: PriceState) -> None:
    client = ProxyFeedClient(settings.gold_proxy.ws_url, settings.gold_proxy.product_id)
    async for tick in client.ticks():
        prices.on_tick(tick.price, tick.publish_time)


async def _run_shadow(settings: Settings, artifact_path: str, db: str, threshold: float) -> None:
    artifact = BlendArtifact.load(Path(artifact_path))
    prices = PriceState(artifact.half_life_s, settings.volatility.short_horizon_s)
    store = ShadowStore(Path(db))
    engine = ShadowEngine(artifact, settings.model, settings.fees, threshold, prices, store)
    windows: dict[str, ShadowWindow] = {}
    print(
        f"SHADOW mode (no orders, no API keys). blend {artifact.version_hash} "
        f"weights={artifact.weights} threshold={threshold} -> {db}",
        flush=True,
    )
    async with httpx.AsyncClient(timeout=15.0, headers={"User-Agent": "Mozilla/5.0"}) as client:
        n = await _seed_prices(client, prices)
        print(f"seeded {n} minutes of PAXG history; sigma={prices.sigma}", flush=True)
        feed = asyncio.create_task(
            _supervised("shadow-feed", lambda: _shadow_feed(settings, prices))
        )
        last_boundary: datetime | None = None
        last_settle = datetime.now(UTC)
        try:
            while True:
                now = datetime.now(UTC)
                boundary = now.replace(second=0, microsecond=0)
                if boundary != last_boundary and (now - boundary).total_seconds() >= 1.5:
                    last_boundary = boundary
                    try:
                        prices.advance(boundary)
                        data = await _history_get(
                            client,
                            f"{KALSHI_BASE}/markets",
                            {"series_ticker": settings.kalshi.series_ticker,
                             "status": "open", "limit": 20},
                        )
                        for m in data.get("markets", []):
                            o = datetime.fromisoformat(m["open_time"].replace("Z", "+00:00"))
                            c = datetime.fromisoformat(m["close_time"].replace("Z", "+00:00"))
                            if not (o <= boundary < c):
                                continue
                            w = windows.setdefault(m["ticker"], ShadowWindow(m["ticker"], o, c))
                            msg = engine.on_boundary(w, boundary, parse_market_quote(m, now))
                            if msg:
                                print(f"{boundary:%H:%M} {msg}", flush=True)
                    except Exception as exc:  # noqa: BLE001 - keep the shadow alive
                        print(f"shadow step error ({exc!r}); continuing", flush=True)
                if (now - last_settle).total_seconds() >= 20:
                    last_settle = now
                    for ticker in store.unsettled_tickers():
                        try:
                            mk = (
                                await _history_get(client, f"{KALSHI_BASE}/markets/{ticker}", {})
                            ).get("market", {})
                        except Exception:  # noqa: BLE001
                            continue
                        if mk.get("result") in ("yes", "no"):
                            pnl = store.settle(ticker, mk["result"])
                            print(
                                f"settled {ticker}: {mk['result']}  shadow pnl {pnl:+.3f}",
                                flush=True,
                            )
                await asyncio.sleep(0.25)
        finally:
            feed.cancel()


def _run_shadow_report(settings: Settings, db: str) -> None:
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    obs = conn.execute(
        "SELECT COUNT(*), COUNT(DISTINCT ticker) FROM shadow_observations"
    ).fetchone()
    trades = conn.execute(
        "SELECT ticker, side, pnl, substr(ts,1,10) FROM shadow_trades WHERE pnl IS NOT NULL"
    ).fetchall()
    pending = conn.execute("SELECT COUNT(*) FROM shadow_trades WHERE pnl IS NULL").fetchone()[0]
    print(f"observations: {obs[0]} across {obs[1]} windows;  shadow trades settled: {len(trades)}"
          f"  pending: {pending}")
    if not trades:
        print("no settled shadow trades yet")
        return
    pnls = {t[0]: [t[2]] for t in trades}
    mean, lo, hi = _cluster_bootstrap(pnls, 2000, random.Random(0))
    wins = sum(1 for t in trades if t[2] > 0)
    days = len({t[3] for t in trades})
    need = settings.learning.min_proposal_trades
    print(
        f"win rate {wins / len(trades):.1%}   mean net P&L/contract ${mean:+.3f}   "
        f"95% CI [${lo:+.3f}, ${hi:+.3f}]   (backtest expected about +$0.08)"
    )
    print(
        f"promotion progress: {len(trades)}/{need} settled trades, {days}/"
        f"{settings.learning.shadow_sessions} shadow days, CI lower bound "
        f"{'above' if lo > 0 else 'NOT above'} zero.  Nothing is promoted automatically."
    )


def _run_study_history(
    settings: Settings,
    db: str,
    quote_mode: str,
    shift_min: int,
    half_lives: list[float],
    start: str | None = None,
    end: str | None = None,
) -> None:
    import math

    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    start_dt = datetime.fromisoformat(start).replace(tzinfo=UTC) if start else None
    end_dt = datetime.fromisoformat(end).replace(tzinfo=UTC) if end else None
    windows, truth = _history_windows(conn, start_dt, end_dt)
    bars = densify(load_paxg_bars(conn))
    if shift_min:
        # PLACEBO: delay the price series so it no longer lines up with the
        # windows. Any real information in the model must vanish; if profit
        # survives, it came from the quotes/selection, not from the model.
        bars = [PriceBar(b.timestamp + timedelta(minutes=shift_min), b.close) for b in bars]
    quote_at = _history_quote_fetcher(conn, quote_mode)
    print(
        f"{len(windows)} settled windows, {len(bars)} PAXG minute prices (all hours); "
        f"quote_mode={quote_mode} placebo_shift_min={shift_min}"
    )

    for half_life in half_lives:
        gld = GldSeries(bars, half_life, settings.volatility.short_horizon_s)
        compared, agree = proxy_agreement(windows, gld)
        pairs = [
            (truth[w.ticker], math.log(gld.price_at(w.close_time) / gld.price_at(w.open_time)))
            for w in windows
            if w.ticker in truth and gld.price_at(w.open_time) and gld.price_at(w.close_time)
        ]
        corr = float("nan")
        if len(pairs) > 2:
            xs, ys = zip(*pairs, strict=True)
            corr = float(np.corrcoef(xs, ys)[0, 1])
        print()
        print(f"##### vol half-life {half_life:.0f}s #####")
        print(f"PAXG proxy audit vs Kalshi's true S0/settlement: return correlation {corr:.3f}")
        rows = build_study_rows(windows, gld, quote_at, settings.model, delay_s=1.5)
        print(
            format_study(
                "raw model vs market (all windows)",
                compared,
                agree,
                compare_brier(rows, n_boot=500),
                [
                    hold_to_settlement(rows, settings.fees, th, n_boot=500)
                    for th in (0.03, 0.05, 0.08)
                ],
            )
        )
        wf = walk_forward_blend(rows, n_boot=500)
        print("\n=== learned blend of market + model ===")
        print(format_blend(wf))
        if wf is not None:
            print("hold-to-settlement using ONLY the held-out blend probabilities:")
            for th in (0.03, 0.05, 0.08):
                h = hold_to_settlement(wf.scored_rows, settings.fees, th, n_boot=500)
                if h.n_trades == 0:
                    print(f"  gap>={th:.2f}: no trades")
                else:
                    print(
                        f"  gap>={th:.2f}: trades={h.n_trades}  win_rate={h.win_rate:.1%}  "
                        f"mean_net_pnl/contract=${h.mean_pnl:+.3f}  "
                        f"95% CI [${h.ci_low:+.3f}, ${h.ci_high:+.3f}]"
                    )


def _run_study_proxies(settings: Settings, db: str, half_life: float) -> None:
    """Stage 4 experiment #1 (plan): is Kraken's PAXG-USD a BETTER free gold
    proxy than Coinbase's, or does a median-of-both consensus beat either
    alone? `proxy_agreement` against Kalshi's true settlement is the exact
    audit CLAUDE.md-style honesty requires before any new source gets
    anywhere near a feature -- this command only reports that audit, it
    never wires a new source into the blend by itself."""
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    windows, _ = _history_windows(conn)
    coinbase = densify(load_paxg_bars(conn))
    kraken = densify(load_kraken_paxg_bars(conn))
    consensus = merge_consensus(coinbase, kraken)
    print(
        f"{len(windows)} settled windows; Coinbase PAXG bars={len(coinbase)}  "
        f"Kraken PAXG bars={len(kraken)}  consensus (median) bars={len(consensus)}"
    )
    for label, bars in (("Coinbase PAXG", coinbase), ("Kraken PAXG", kraken),
                         ("consensus (median)", consensus)):
        if not bars:
            print(f"{label}: no data -- run `gold-edge backfill` first")
            continue
        gld = GldSeries(bars, half_life, settings.volatility.short_horizon_s)
        compared, agree = proxy_agreement(windows, gld)
        if compared == 0:
            print(f"{label}: no windows overlap this source's coverage")
            continue
        print(f"{label}: outcome agreement vs Kalshi settlement {agree}/{compared} "
              f"({agree / compared:.1%})")


async def _run_study_market(
    settings: Settings, start: str | None, end: str | None, delay_s: float, cache: str
) -> None:
    if not settings.sqlite_path.exists():
        print(f"No recorded data found at {settings.sqlite_path}.")
        sys.exit(1)
    cache_path = Path(cache)
    try:
        fresh = await fetch_yahoo_chart("GLD", "1m", "7d")
        save_gld_cache(cache_path, fresh)
        print(f"Fetched {len(fresh)} GLD 1m bars (free Yahoo); cache now at {cache_path}")
    except httpx.HTTPError as exc:
        print(f"Yahoo fetch failed ({exc}); using cached bars only")
    bars = load_gld_cache(cache_path)
    print(f"GLD 1m bars available: {len(bars)}")

    conn = sqlite3.connect(f"file:{settings.sqlite_path}?mode=ro", uri=True)
    start_dt = datetime.fromisoformat(start).replace(tzinfo=UTC) if start else None
    end_dt = datetime.fromisoformat(end).replace(tzinfo=UTC) if end else None
    windows = _load_settled_windows(conn, start_dt, end_dt)
    print(f"Settled Kalshi windows in range: {len(windows)}")
    quote_at = _quote_fetcher(conn)

    # Live half-life (matches the recorder's 1s ticks) vs a longer one that
    # actually spans several 1-minute bars; both shown, neither picked.
    for half_life in (settings.volatility.ewma_half_life_s, 900.0):
        gld = GldSeries(bars, half_life, settings.volatility.short_horizon_s)
        compared, agree = proxy_agreement(windows, gld)
        rows = build_study_rows(windows, gld, quote_at, settings.model, delay_s)
        print()
        print(
            format_study(
                f"vol half-life {half_life:.0f}s",
                compared,
                agree,
                compare_brier(rows),
                [hold_to_settlement(rows, settings.fees, th) for th in (0.03, 0.05, 0.08)],
            )
        )
    print(
        "\nNOTE: GLD stands in for spot only during NYSE hours, ~100 independent windows at "
        "best -- treat any result as a lead to confirm with live-recorded spot data, not proof."
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="gold-edge")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("record", help="Stream Pyth ticks + Kalshi orderbook and record them.")

    validate_parser = sub.add_parser(
        "validate-feed", help="Compare recorded Pyth candle closes to Kalshi settlements."
    )
    validate_parser.add_argument("--limit", type=int, default=20)

    sub.add_parser(
        "fair-value", help="Print live fair value vs Kalshi bid/ask for the current window."
    )

    live_parser = sub.add_parser(
        "live", help="Run the live dashboard server (feeds + engine + web UI)."
    )
    live_parser.add_argument("--host", default="127.0.0.1")
    live_parser.add_argument("--port", type=int, default=8000)

    backtest_parser = sub.add_parser(
        "backtest",
        help="Replay recorded data through the live engine with human-delay fills.",
    )
    backtest_parser.add_argument("--start", help="ISO datetime; only windows opening at/after this")
    backtest_parser.add_argument("--end", help="ISO datetime; only windows closing at/before this")
    backtest_parser.add_argument("--seed", type=int, default=None, help="Fill-delay RNG seed")
    backtest_parser.add_argument(
        "--bucket-width", type=float, default=0.1, help="Calibration bucket width"
    )
    backtest_parser.add_argument(
        "--sweep-param",
        action="append",
        dest="sweep_params",
        metavar="KEY=V1,V2,...",
        help="Sweep an EngineConfig field over these values (repeatable). Requires --train-end.",
    )
    backtest_parser.add_argument(
        "--train-end",
        help="ISO datetime splitting training (<=) from held-out (>) windows for --sweep-param",
    )

    learn_parser = sub.add_parser(
        "learn",
        help="Compute markouts/grades/opportunities/attribution/patterns and refit learned "
        "components on recorded data; never changes live behavior on its own.",
    )
    learn_parser.add_argument("--start", help="ISO datetime; only windows opening at/after this")
    learn_parser.add_argument("--end", help="ISO datetime; only windows closing at/before this")
    learn_parser.add_argument("--seed", type=int, default=None)

    review_parser = sub.add_parser(
        "review", help="Print a report card from already-graded data (run `learn` first)."
    )
    review_parser.add_argument("--start", help="ISO datetime lower bound")
    review_parser.add_argument("--end", help="ISO datetime upper bound")

    sub.add_parser("proposals", help="List pending/approved/rejected threshold-change proposals.")

    promote_parser = sub.add_parser(
        "promote", help="Approve a proposal if it clears every promotion gate."
    )
    promote_parser.add_argument("proposal_id")

    sub.add_parser("rollback", help="Restore the previously promoted config version.")

    history_parser = sub.add_parser(
        "analyze-history",
        help="Check the fair-value formula and volatility defaults against years of "
        "free historical GLD prices (calibration check only, not a trading backtest).",
    )
    history_parser.add_argument("--daily-symbol", default="GLD")
    history_parser.add_argument("--daily-range", default="20y", help="Yahoo range; keep <= ~20y")
    history_parser.add_argument("--intraday-symbol", default="GLD")
    history_parser.add_argument("--intraday-interval", default="1m")
    history_parser.add_argument("--intraday-range", default="7d", help="Yahoo cap for 1m bars")

    backfill_parser = sub.add_parser(
        "backfill",
        help="Download months of settled KXGOLD15M windows, minute bid/ask candles and "
        "PAXG prices from free public endpoints (no API keys) into data/history.sqlite.",
    )
    backfill_parser.add_argument("--days", type=int, default=60)
    backfill_parser.add_argument("--db", default="data/history.sqlite")

    history_study_parser = sub.add_parser(
        "study-history",
        help="Train/test on the backfilled history: proxy audit, model vs market, and a "
        "walk-forward learned blend (offline).",
    )
    history_study_parser.add_argument("--db", default="data/history.sqlite")
    history_study_parser.add_argument("--start", help="ISO datetime; only windows opening at/after")
    history_study_parser.add_argument("--end", help="ISO datetime; only windows opening before")
    history_study_parser.add_argument(
        "--quote-mode", choices=["next_open", "close"], default="close"
    )
    history_study_parser.add_argument("--shift-min", type=int, default=0, help="placebo shift")
    history_study_parser.add_argument(
        "--half-life", type=float, action="append", dest="half_lives"
    )

    proxies_parser = sub.add_parser(
        "study-proxies",
        help="Audit free gold-proxy sources (Coinbase PAXG, Kraken PAXG, and a median "
        "consensus of both) against Kalshi's true settlement, before trusting any of them "
        "in a feature.",
    )
    proxies_parser.add_argument("--db", default="data/history.sqlite")
    proxies_parser.add_argument("--half-life", type=float, default=900.0)

    features_parser = sub.add_parser(
        "study-features",
        help="Test richer blend features with a dev walk-forward and a one-shot holdout.",
    )
    features_parser.add_argument("--db", default="data/history.sqlite")
    features_parser.add_argument("--holdout-days", type=int, default=14)

    propose_model_parser = sub.add_parser(
        "propose-model",
        help="Fit the market+model blend fresh and evaluate it against every promotion gate "
        "(holdout Brier CI + live-shadow trade count/days/P&L CI). Prints the gate result; "
        "only writes anything with --promote, and only if every gate passes.",
    )
    propose_model_parser.add_argument("--history-db", default="data/history.sqlite")
    propose_model_parser.add_argument("--shadow-db", default="data/shadow.sqlite")
    propose_model_parser.add_argument("--holdout-days", type=int, default=14)
    propose_model_parser.add_argument("--out", default="data/blend_model.json")
    propose_model_parser.add_argument(
        "--promote", action="store_true", help="Approve and persist if every gate passes"
    )

    sub.add_parser("rollback-model", help="Restore the previously promoted model version.")

    train_parser = sub.add_parser(
        "train-blend", help="Fit the market+model blend on the backfilled history -> artifact."
    )
    train_parser.add_argument("--db", default="data/history.sqlite")
    train_parser.add_argument("--half-life", type=float, default=900.0)
    train_parser.add_argument("--out", default="data/blend_model.json")

    shadow_parser = sub.add_parser(
        "shadow",
        help="Run the blend LIVE in shadow mode: records what it would trade, never orders. "
        "Keyless (public Kalshi + Coinbase data).",
    )
    shadow_parser.add_argument("--artifact", default="data/blend_model.json")
    shadow_parser.add_argument("--db", default="data/shadow.sqlite")
    shadow_parser.add_argument("--threshold", type=float, default=0.05)

    report_parser = sub.add_parser("shadow-report", help="Summarise shadow results vs the gates.")
    report_parser.add_argument("--db", default="data/shadow.sqlite")

    study_parser = sub.add_parser(
        "study-market",
        help="Score the fair-value model against real recorded Kalshi prices on settled "
        "windows, using free GLD minute bars as the price proxy (offline, no API keys).",
    )
    study_parser.add_argument("--start", help="ISO datetime; only windows opening at/after this")
    study_parser.add_argument("--end", help="ISO datetime; only windows closing at/before this")
    study_parser.add_argument("--delay-s", type=float, default=1.5)
    study_parser.add_argument("--cache", default="data/gld_1m_cache.json")

    args = parser.parse_args(argv)
    settings = get_settings()

    if args.command == "record":
        asyncio.run(_run_record(settings))
    elif args.command == "validate-feed":
        asyncio.run(_run_validate_feed(settings, args.limit))
    elif args.command == "fair-value":
        asyncio.run(_run_fair_value(settings))
    elif args.command == "live":
        _run_live(settings, args.host, args.port)
    elif args.command == "backtest":
        _run_backtest(
            settings,
            args.start,
            args.end,
            args.seed,
            args.bucket_width,
            args.sweep_params,
            args.train_end,
        )
    elif args.command == "learn":
        _run_learn(settings, args.start, args.end, args.seed)
    elif args.command == "review":
        _run_review(settings, args.start, args.end)
    elif args.command == "proposals":
        _run_proposals(settings)
    elif args.command == "promote":
        _run_promote(settings, args.proposal_id)
    elif args.command == "rollback":
        _run_rollback(settings)
    elif args.command == "study-proxies":
        _run_study_proxies(settings, args.db, args.half_life)
    elif args.command == "study-features":
        asyncio.run(_run_study_features(settings, args.db, args.holdout_days))
    elif args.command == "propose-model":
        _run_propose_model(
            settings, args.history_db, args.shadow_db, args.holdout_days, args.out, args.promote
        )
    elif args.command == "rollback-model":
        _run_rollback_model(settings)
    elif args.command == "train-blend":
        _run_train_blend(settings, args.db, args.half_life, args.out)
    elif args.command == "shadow":
        asyncio.run(_run_shadow(settings, args.artifact, args.db, args.threshold))
    elif args.command == "shadow-report":
        _run_shadow_report(settings, args.db)
    elif args.command == "study-history":
        _run_study_history(
            settings,
            args.db,
            args.quote_mode,
            args.shift_min,
            args.half_lives or [settings.volatility.ewma_half_life_s, 900.0],
            args.start,
            args.end,
        )
    elif args.command == "backfill":
        asyncio.run(_run_backfill(settings, args.days, args.db))
    elif args.command == "study-market":
        asyncio.run(_run_study_market(settings, args.start, args.end, args.delay_s, args.cache))
    elif args.command == "analyze-history":
        asyncio.run(
            _run_analyze_history(
                settings,
                args.daily_symbol,
                args.daily_range,
                args.intraday_symbol,
                args.intraday_interval,
                args.intraday_range,
            )
        )


if __name__ == "__main__":
    main()
