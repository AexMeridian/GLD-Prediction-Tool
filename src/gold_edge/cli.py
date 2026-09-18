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
from datetime import UTC, datetime
from pathlib import Path

from gold_edge.backtest.replay import load_recorded_data, replay_all, run_sweep
from gold_edge.backtest.report import format_report, format_sweep_report
from gold_edge.config import Settings, get_settings
from gold_edge.feeds.kalshi_auth import KalshiSigner
from gold_edge.feeds.kalshi_rest import KalshiRestClient
from gold_edge.feeds.kalshi_ws import KalshiWsClient
from gold_edge.feeds.pyth import PythFeedClient
from gold_edge.learning.actions import promote_proposal, rollback_config
from gold_edge.learning.delay_profile import FillLatencySample
from gold_edge.learning.insights import session_report_card, weekly_report
from gold_edge.learning.opportunities import filter_scorecard
from gold_edge.learning.patterns import load_macro_events
from gold_edge.learning.pipeline import run_learning_pipeline
from gold_edge.learning.queries import load_proposals, load_session_data
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


async def _pyth_recording_loop(client: PythFeedClient, recorder: Recorder) -> None:
    count = 0
    async for tick in client.ticks():
        await recorder.record_tick(tick)
        count += 1
        if count % 10 == 0:
            print(f"[pyth]   {tick.publish_time.isoformat()}  price={tick.price:.2f}  (n={count})")


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
        pyth_task = asyncio.create_task(_pyth_recording_loop(pyth_client, recorder))

        current_window: Window | None = None
        kalshi_task: asyncio.Task | None = None
        try:
            while True:
                window = await get_current_window(rest, settings.kalshi.series_ticker)
                if window is None:
                    next_window = await get_next_window(rest, settings.kalshi.series_ticker)
                    if next_window is None:
                        print("No open or upcoming KXGOLD15M market found; retrying in 10s")
                        await asyncio.sleep(10)
                        continue
                    wait_s = max(0.0, (next_window.open_time - datetime.now(UTC)).total_seconds())
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

                await asyncio.sleep(5)
        finally:
            pyth_task.cancel()
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
    logged fills, joined against the signal that produced them."""
    conn = sqlite3.connect(sqlite_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT f.signal_id, f.action, s.reason, f.logged_at, s.created_at, s.status "
            "FROM fills f JOIN signals s ON f.signal_id = s.id WHERE f.signal_id IS NOT NULL"
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
    return samples


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


if __name__ == "__main__":
    main()
