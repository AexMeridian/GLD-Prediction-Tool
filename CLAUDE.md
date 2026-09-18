# Gold Edge — Live Intra-Window Signal Tool for Kalshi 15-Min Gold Markets

## What this project is
A live decision-support tool for Kalshi's 15-minute gold up/down markets
(series `KXGOLD15M`). During each live 15-minute window it tells the human
user what to do, in real time, as a sequence of signals:

  BUY YES → SELL YES → BUY NO → SELL NO → HOLD ...

Multiple round trips per window are expected. The tool does NOT predict
where gold "will go." It computes a fair-value probability for YES/NO from
the live Pyth gold price and compares it to the Kalshi orderbook after all
fees and spread. It signals an entry when the market is mispriced by more
than the full round-trip cost, and an exit when the mispricing is gone.

The human places orders manually on Kalshi. Automated order placement is
out of scope until explicitly requested (see "Live execution" below).

## Contract facts (verify before relying on them)
- Each window asks: is gold up over the 15 minutes?
- Resolves YES if the close of the 1-minute Pyth XAU/USD candle at window
  close is >= the close of the 1-minute candle at window open (ties = YES).
  Prices rounded to 2 decimals.
- Contracts pay $1.00 if correct, $0 otherwise. Price in cents ≈ probability.
- No settlement fee. Trading fee per order:
  `fee = ceil_to_cent(M * 0.07 * C * P * (1 - P))`
  where M = per-market fee multiplier, C = contracts, P = price in dollars.
  Maker (resting) orders also pay fees as of Aug 2026. Keep all fee
  parameters in `config.yaml`, never hardcoded.
- BEFORE implementing settlement logic: fetch a live KXGOLD15M market from
  the Kalshi API, read its rules text and strike/reference fields, and
  confirm how the reference price S0 is defined and when it becomes known.
  Record findings in `docs/contract_notes.md`. If anything above is wrong,
  update this file and tell the user.

## External data
- **Pyth Hermes** (price source = settlement source). Stream XAU/USD via
  Hermes SSE (`https://hermes.pyth.network/v2/updates/price/stream`). Look
  up the correct XAU/USD price feed ID from Pyth's official feed list — do
  not guess it. Parse price * 10^expo; also store confidence and publish_time.
- **Kalshi Trade API v2**. REST base and WebSocket URL from Kalshi's current
  docs (currently under `api.elections.kalshi.com/trade-api/`). Auth uses an
  API key ID + RSA private key signing (headers KALSHI-ACCESS-KEY,
  KALSHI-ACCESS-TIMESTAMP, KALSHI-ACCESS-SIGNATURE). Read the docs before
  writing the client; do not assume endpoint shapes. Use WebSocket
  orderbook deltas for live books and REST for market discovery,
  settlements, and (read-only) fills.
- Secrets live in `.env` (KALSHI_API_KEY_ID, KALSHI_PRIVATE_KEY_PATH).
  `.env` and key files must be in `.gitignore`. Never print secrets.

## Tech stack
- Python 3.12, `asyncio` throughout
- `httpx`, `websockets`, `cryptography` (RSA signing)
- `numpy`, `scipy` (norm.cdf), `pandas`, `pyarrow`
- SQLite (live recording) + Parquet (backtest datasets)
- `fastapi` + `uvicorn` backend; dashboard is a single static HTML/JS page
  receiving pushes over a WebSocket (no Streamlit — too slow for this)
- `pydantic` for config and message models
- `pytest`, `pytest-asyncio`; `ruff` for lint/format
- Package management with `uv`

## Project layout
```
gold-edge/
├── CLAUDE.md
├── config.yaml
├── .env.example
├── pyproject.toml
├── docs/contract_notes.md
├── src/gold_edge/
│   ├── config.py            # pydantic settings loaded from config.yaml + .env
│   ├── models.py            # Tick, BookSnapshot, Window, Signal, Position, Fill
│   ├── feeds/
│   │   ├── pyth.py          # Hermes SSE client, reconnect, stale detection
│   │   ├── kalshi_auth.py   # request signing
│   │   ├── kalshi_rest.py   # market discovery, rules, settlements, fills
│   │   └── kalshi_ws.py     # orderbook subscription, local book maintenance
│   ├── windows.py           # find current/next KXGOLD15M market, S0, time left
│   ├── model/
│   │   ├── volatility.py    # EWMA realized vol from Pyth ticks
│   │   ├── fair_value.py    # P(YES) given S, S0, sigma, time left
│   │   └── fees.py          # Kalshi fee formula, taker & maker
│   ├── engine/
│   │   ├── state_machine.py # FLAT / LONG_YES / LONG_NO transitions
│   │   ├── signals.py       # builds Signal objects with limits + expiry
│   │   └── risk.py          # limits, cooldowns, kill switch, no-trade zones
│   ├── recorder.py          # writes every tick/book/signal/settlement
│   ├── backtest/
│   │   ├── replay.py        # replays recorded data through the engine
│   │   ├── fills.py         # human-delay fill simulator
│   │   └── report.py        # metrics, calibration, parameter sweeps
│   ├── learning/
│   │   ├── markouts.py      # fair/market value at +5/+15/+30/+60s after events
│   │   ├── grader.py        # grades every signal, fill, exit, and skip
│   │   ├── opportunities.py # hindsight scan for missed trades + blocked losers
│   │   ├── attribution.py   # why: model error, delay, costs, filter, exit timing
│   │   ├── patterns.py      # aggregates grades by regime / time / gap size
│   │   ├── calibrator.py    # learned recalibration of fair value
│   │   ├── trade_filter.py  # learned P(entry is profitable after delay+fees)
│   │   ├── delay_profile.py # learns the user's real click latency
│   │   ├── proposer.py      # walk-forward param search -> Proposal objects
│   │   ├── shadow.py        # runs candidate configs silently alongside live
│   │   ├── registry.py      # versioned configs/models, promote, rollback
│   │   ├── drift.py         # live calibration / performance drift alerts
│   │   └── insights.py      # plain-language session & weekly reports
│   ├── server.py            # FastAPI app: runs feeds + engine, pushes to UI
│   └── cli.py               # record, live, backtest, validate-feed,
│                            # review, learn, proposals, promote, rollback
├── web/index.html           # dashboard: Live tab + Review tab + Learning tab
└── tests/
```

## Core model
Recompute on every Pyth tick and every Kalshi book update:

```
tau        = seconds_left / 60                      # minutes remaining
sigma      = EWMA per-minute vol of log returns     # floor at config min
fair_yes   = norm.cdf( ln(S / S0) / (sigma * sqrt(tau)) )
fair_no    = 1 - fair_yes
```
- Clamp fair values to [0.01, 0.99].
- Handle tau → 0 and S == S0 without division errors (ties resolve YES).
- Vol: EWMA of 1-second-sampled log returns scaled to per-minute, with a
  configurable half-life; also compute a short-horizon vol (e.g. 60s) to
  detect spikes.
- The model is a baseline. Additional features (order-flow imbalance,
  scheduled news times, DXY) may be added ONLY if they improve out-of-sample
  calibration and backtest P&L. Keep the baseline available for comparison.

## Trading engine (per window)
States: `FLAT`, `LONG_YES`, `LONG_NO`. One position at a time.

Definitions (all in dollars per contract):
```
entry_cost(side) = ask(side) + fee(ask(side))
exit_value(side) = bid(side) - fee(bid(side))
round_trip_est   = fee(ask) + fee_est(exit) + spread_est
gap(side)        = fair(side) - ask(side) - round_trip_est
```

### Entry (only from FLAT)
All must be true:
- `gap(side) > ENTER_EDGE` (default 0.03), persisting ≥ `PERSIST_S` (1.5s)
- not in cooldown (`COOLDOWN_S`, default 20s after any exit)
- `time_left > ENTRY_CUTOFF_S` (default 90s)
- Pyth data fresh (< `STALE_S`, default 3s) and Kalshi book fresh
- spread on that side ≤ `MAX_SPREAD` (default 0.04)
- short-horizon vol not above `VOL_SPIKE_LIMIT`
- round trips this window < `MAX_ROUND_TRIPS` (default 6)
- daily realized loss not beyond `DAILY_LOSS_STOP`

If both sides qualify (shouldn't happen), take neither and log it.

### Exit (from LONG_x) — first condition that fires
1. **Overshoot:** `exit_value > fair(side)` → SELL now
2. **Converged:** `exit_value >= fair(side) - CONVERGE_BAND` (0.01) → SELL
3. **Stop:** `bid(side) <= entry_price - STOP` (0.08) → SELL
4. **Time:** `time_left < EXIT_CUTOFF_S` (75s) → SELL, unless config
   `hold_to_settlement_when_itm: true` and fair(side) ≥ 0.90
5. **Data failure:** stale feed > `STALE_S` → SELL-advisory with warning

### Flip
LONG_YES → LONG_NO directly only if `gap(NO)` exceeds `ENTER_EDGE` plus the
cost of exiting YES. Otherwise exit to FLAT and apply cooldown.

### Hysteresis
Entry requires edge ≥ ENTER_EDGE; edge-decay alone does not exit until the
converged/overshoot rules fire. Never flip-flop on single ticks.

All thresholds live in `config.yaml`.

## Signals
Every signal is a pydantic model:
```
{id, window_ticker, action: BUY|SELL|HOLD, side: YES|NO,
 limit_price, size, fair, market_price, edge_after_costs,
 reason, created_at, expires_at}
```
- BUY limit = ask at signal time; SELL limit = bid at signal time.
- Signals expire after `SIGNAL_TTL_S` (default 6s) or when the book moves
  past the limit. Expired signals are marked `MISSED` and are never
  re-issued at a worse price ("don't chase").
- Engine position state changes only when the user logs a fill (or, later,
  when a fill is read from the API). Until confirmed, show "awaiting fill."

## Dashboard (web/index.html)
Must update within ~250ms of engine events. Layout top to bottom:
1. **Action card**: huge text (▲ BUY YES / ▼ BUY NO / ✓ SELL / HOLD),
   limit price, size, edge, countdown to expiry. Distinct audio tone per
   action type. Grey "MISSED" state.
2. **Fill buttons**: "I bought at __" / "I sold at __" (prefilled with limit,
   editable), plus "Skipped".
3. **Window bar**: ticker, S0, live S, seconds to close, entry cutoff marker.
4. **Chart**: live Pyth price vs S0 line for the current window.
5. **Fair vs market**: fair YES/NO next to bid/ask for both sides.
6. **Position**: side, size, entry, unrealized P&L after exit fees.
7. **Session log**: every signal, taken/skipped/missed, realized P&L, fees.
8. **Status**: feed health, data age, kill switch button.

Support dark mode. Must be usable on a laptop screen next to Kalshi.

## Recorder
Always on when `live` runs, and runnable alone via `record`. Store: Pyth
ticks (price, conf, publish_time, receive_time), Kalshi book snapshots and
deltas with receive_time, window metadata, S0, settlements, every signal,
every logged fill. Use receive timestamps for latency analysis.

## Backtesting
- Replay recorded data through the exact same engine code (no duplicated
  strategy logic).
- **Human delay is mandatory:** a signal fills only if, at
  `signal_time + delay` (sample delay uniformly from 1.0–2.0s, configurable),
  the book still offers the limit price or better. Otherwise it's MISSED.
- Apply full taker fees on both legs; positions still open at close settle
  at $1/$0 using the recorded settlement.
- Report per window and overall: signals, fills, misses, round trips, fees,
  gross and net P&L, max drawdown, P&L per round trip, and a comparison to
  a do-nothing baseline.
- Calibration report: bucket fair_yes predictions vs actual outcomes.
- Parameter sweeps (ENTER_EDGE, STOP, COOLDOWN, cutoffs) must be tuned on
  training days and reported on held-out days only.
- `validate-feed`: compare our Pyth candle closes to Kalshi's published
  settlement values; report agreement rate and all disagreements near S0.

## Self-learning system
The tool reviews every session, grades its own decisions, finds missed
opportunities, explains why things went right or wrong, and proposes
improvements. It learns from recorded data and graded outcomes, but it
**never changes live behavior on its own**: improvements go through
validation, shadow testing, and user approval.

### Principle: grade decisions, not just outcomes
In 15-minute gold markets luck dominates single trades. A trade can lose
money and still be a good decision (edge was real, price moved against it),
or make money and be a bad decision. Every grade separates:
- **Decision quality**: was the edge real at signal time? Measured by
  markouts — did the market move toward our fair value afterward?
- **Execution quality**: did the fill (user delay, slippage) preserve the edge?
- **Outcome**: net P&L after fees.

### Markouts (learning/markouts.py)
For every signal (taken, skipped, or missed), and every filter-blocked
candidate, record at +5s, +15s, +30s, +60s:
market mid and bid/ask for that side, fair value, and Pyth price.
- `edge_markout(t) = market_mid(t) - signal_market_price` for buys
  (direction-adjusted for sells).
- Positive markout = the market moved toward our fair value → edge was real.

### Grades (learning/grader.py)
Each entry+exit round trip gets exactly one primary grade and optional tags:

| Grade | Meaning |
|---|---|
| `GOOD_CALL` | positive markout AND positive net P&L |
| `GOOD_BUT_UNLUCKY` | positive markout, lost due to later price move |
| `LUCKY` | negative markout, made money anyway (do not reinforce) |
| `BAD_MODEL` | negative markout; market was right, fair value was wrong |
| `BAD_EXECUTION` | edge existed at signal time but was gone by the fill |
| `COSTS_ATE_EDGE` | gross positive, net negative after fees/spread |
| `EXIT_TOO_EARLY` | after exit, value kept moving our way by > EXIT_REGRET |
| `EXIT_TOO_LATE` | better exit_value was available earlier in the trade |
| `STOPPED_CORRECTLY` | stop fired and price kept moving against us |
| `STOPPED_WRONGLY` | stop fired and price reverted past entry |

Non-trades are also graded:

| Grade | Meaning |
|---|---|
| `MISSED_BY_USER` | signal issued, user skipped or signal expired, would have profited after delay+fees |
| `SKIP_WAS_RIGHT` | user skipped/missed and trade would have lost |
| `MISSED_BY_FILTER` | a filter blocked an entry that would have profited |
| `FILTER_SAVED_US` | a filter blocked an entry that would have lost |
| `MISSED_NO_SIGNAL` | hindsight opportunity with no candidate at all |

All grade thresholds (EXIT_REGRET, markout horizons, etc.) live in config.

### Missed-opportunity scanner (learning/opportunities.py)
After each window, scan recorded data for hindsight-profitable round trips:
- Enumerate candidate entries at each second where either side's ask was
  below fair value, simulate the user's learned delay, apply the current
  exit rules AND an oracle best-exit, and net out full fees.
- An opportunity counts only if it clears a minimum net profit
  (`MIN_OPPORTUNITY`, default 0.03/contract) under the realistic exit rules,
  not just the oracle exit. The oracle is reported separately as the
  theoretical ceiling.
- For each opportunity, record **why the engine didn't take it**:
  `below_enter_edge`, `persistence`, `cooldown`, `entry_cutoff`,
  `spread_filter`, `stale_data`, `vol_spike`, `max_round_trips`,
  `already_in_position`, `fair_value_disagreed`.
- Symmetrically, track every filter block and whether it saved money.
  A filter's value = losses avoided − profits missed. Report per filter.

### Attribution (learning/attribution.py)
For each losing or missed trade, assign the dominant cause with an estimated
dollar impact: model error, volatility misestimate, user delay, spread,
fees, exit rule, filter, stale data, or news/vol shock. Session reports roll
these up: "Of −$4.20 today, $2.60 was user delay, $1.10 fees, $0.50 model."

### Pattern mining (learning/patterns.py)
Aggregate grades and net P&L per round trip across dimensions:
minute-of-window, time of day / session (Asia/London/NY), gap size bucket,
side, spread, realized vol regime, distance |S−S0| in sigmas, time since
last trade, proximity to scheduled macro releases (maintain a simple
events calendar file), and day of week.
- Report only buckets with n ≥ `MIN_BUCKET_N` (default 30), with confidence
  intervals (bootstrap). Flag everything else "insufficient data."
- Apply a multiple-comparisons correction (Benjamini–Hochberg) before
  calling any pattern significant.

### Learned components
1. **Fair-value calibrator (calibrator.py):** isotonic regression mapping
   raw `fair_yes` → calibrated probability, fit on settled windows and on
   markout-implied values. Used only if it improves held-out log-loss/Brier.
2. **Volatility scaling:** learn a multiplier on sigma by vol regime and
   time of day that minimizes held-out log-loss.
3. **Trade filter (trade_filter.py):** a regularized logistic regression
   (optionally LightGBM later) predicting P(round trip is net profitable
   after learned delay and fees) from the pattern features. Acts as an
   extra entry gate: enter only if P ≥ `FILTER_MIN_PROB`. Must beat the
   no-filter engine on held-out data by `PROMOTION_MARGIN`.
4. **Delay profile (delay_profile.py):** from logged fills, learn the user's
   actual latency distribution (signal created_at → fill logged, and fill
   price vs limit). Replace the default 1–2s backtest delay with this
   profile once n ≥ 50 fills. Also learn which signal types the user tends
   to miss.
5. **Threshold proposals (proposer.py):** walk-forward optimization of
   ENTER_EDGE, PERSIST_S, COOLDOWN_S, STOP, CONVERGE_BAND, cutoffs, and
   MAX_SPREAD. Objective: net P&L per window with a drawdown penalty.
   Keep search grids small and bounded to sane ranges.

### Learning pipeline (runs after sessions, never during a live window)
`learn` command, also triggerable from the dashboard when no window is live:
1. Compute markouts, grades, opportunities, attribution for new data.
2. Refit learned components on a rolling training window.
3. Evaluate walk-forward: train on days 1..k, test on day k+1, roll forward.
   Report only out-of-sample results.
4. Emit `Proposal` objects: what changes, evidence (sample size, held-out
   delta P&L with CI, drawdown change, calibration change), and plain-
   language rationale tied to the grades that motivated it.

### Promotion gates (registry.py + shadow.py)
A proposal becomes live only if ALL pass:
- ≥ `MIN_PROPOSAL_TRADES` held-out round trips (default 200)
- held-out net P&L improvement ≥ `PROMOTION_MARGIN` with the CI lower bound
  above zero
- max drawdown not worse by more than `MAX_DD_WORSEN`
- **Shadow mode:** runs silently alongside the live config for
  `SHADOW_SESSIONS` (default 10) sessions, generating its own hypothetical
  signals, graded with the same grader; must still beat live config
- **User approval** in the Learning tab (or `promote` CLI)

Registry rules:
- Every config and model is versioned (hash + timestamp + evidence snapshot).
- Every recorded signal stores the config/model version that produced it.
- `rollback` restores any prior version in one step.
- Maximum one promotion per day; no change mid-session.
- Anti-overfitting: never evaluate on data used for fitting; if a proposal
  was rejected, don't re-propose near-identical params within N days.

### Drift monitoring (drift.py)
Rolling live checks: calibration error, markout sign rate, net P&L per
round trip, delay profile. If metrics degrade beyond thresholds vs the
promoted version's validation results, show a warning banner, suggest
raising ENTER_EDGE or pausing, and offer rollback. Drift alerts never
change config automatically.

### Insights reports (insights.py)
Deterministic, template-generated, numbers-first text (no LLM required):
- **Session report card:** trades, grades breakdown, net P&L, fees,
  best call, worst call, top 3 missed opportunities with reasons,
  attribution rollup, filter scorecard.
- **Weekly report:** significant patterns (with n and CI), calibration
  plot, delay profile trend, pending proposals, drift status.
- Every claim shows its sample size. Small samples are labeled.

### Storage
New SQLite tables: `markouts`, `grades`, `opportunities`, `filter_events`,
`attribution`, `pattern_stats`, `model_versions`, `config_versions`,
`proposals`, `shadow_signals`, `drift_events`, `user_fills_latency`.

### Dashboard additions
- **Review tab:** pick a session/window; replay chart with S0, fair value,
  bid/ask; markers for signals taken (colored by grade), skipped, missed,
  filter-blocked, and hindsight opportunities. Click a marker to see its
  markouts, grade, attribution, and reason. Session report card at top.
- **Learning tab:** calibration curve, filter scorecard, pattern tables,
  delay profile, pending proposals with evidence and Approve/Reject,
  shadow-mode comparison, version history with Rollback, drift status.
- **Live tab:** small "current version" badge and drift banner only.
  No learning computation in the live path.

## Live execution (NOT in initial scope)
Do not build automated order placement unless the user explicitly asks.
If requested later: separate module, `live_trading: false` default,
requires config flag + CLI confirmation + per-order size cap, and all
risk limits still apply.

## Engineering rules
- Tests first for `fees.py`, `fair_value.py`, `state_machine.py`, `risk.py`,
  and `backtest/fills.py`. Include edge cases: tau≈0, S==S0, empty book,
  one-sided book, stale feeds, fee rounding at 1¢/99¢/50¢.
- Tests for learning: grader on hand-built scenarios for every grade,
  opportunity scanner on synthetic windows with known answers, walk-forward
  splitter proving no train/test leakage, registry promote/rollback, and
  promotion gates rejecting proposals that fail any single gate.
- State machine must be pure and deterministic (inputs → state + signals)
  so it can be unit tested and replayed.
- Feeds must auto-reconnect with backoff and mark data stale while down.
- Use `Decimal` or integer cents for prices and fees; floats only in the
  model math.
- Log structured JSON. No secrets in logs.
- Keep functions small; type hints everywhere; `ruff` clean.
- After each phase: run tests, summarize what was built, list open
  questions, and stop for user review before the next phase.

## Honesty requirements
- The dashboard footer states that signals are model output, not advice,
  and contracts can lose their full value.
- Backtest reports must show net-of-fees, human-delayed results prominently.
  If results are negative, say so plainly; do not tune on test data to
  make them positive.
