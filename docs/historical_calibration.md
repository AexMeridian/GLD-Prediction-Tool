# Historical GLD calibration check (`gold-edge analyze-history`)

## What it is

A model "instinct" check that runs the exact production fair-value formula
(`model/fair_value.py`) and volatility tracker (`model/volatility.py`)
against years of real, free, keyless GLD (SPDR Gold Shares ETF) price
history pulled from Yahoo Finance's public chart endpoint
(`query1.finance.yahoo.com/v8/finance/chart/GLD`). It reports:

- Realized volatility (overall and by weekday), to compare against
  `config.yaml`'s `volatility.min_sigma_per_minute` / `ewma_half_life_s`.
- A calibration table: bucket the model's predicted `fair_yes` against how
  often gold actually finished up over that window, at two horizons:
  - **5-trading-day window** (~20 years of daily bars, `range=20y&interval=1d`)
    — the shortest window daily-only data can test without the bug below.
  - **15-minute window** (~7 days of 1-minute bars during NYSE hours,
    `range=7d&interval=1m`) — exactly the horizon Kalshi's windows use.

  Each window uses its first bar as the fixed open (S0) and its last bar
  only to determine the outcome; the model is evaluated once per bar
  strictly BETWEEN open and close, using that bar's price as the live
  "current" price and the actual remaining time as tau — mirroring how the
  live engine continuously recomputes fair value as a window runs. A window
  with no interior bar (e.g. a 1-day window built from daily-only data) has
  nothing to evaluate, which is why the daily check uses a 5-day window
  instead of 1.
- Whether an isotonic recalibration (the same one `learning/calibrator.py`
  would fit on live data) would improve held-out Brier/log-loss on this
  much larger sample than live recording alone can offer yet.
- **Multiple rounds of walk-forward analysis**, not one single train/test
  split: the daily dataset is split into one round per calendar year, the
  intraday dataset into one round per trading day. Each round fits a
  calibrator only on every STRICTLY EARLIER round pooled together
  (expanding window, "train on days/years 1..k, test on k+1, roll
  forward," per CLAUDE.md) and scores it on that round alone, never on data
  used to fit it. `run_walkforward_rounds`/`format_walkforward_report` in
  `historical_gold.py`. This is what actually answers "does this hold up,"
  not just "did it work once" — a single 80/20 split can be a lucky draw;
  a recalibration that helps in most rounds across 20 separate years is a
  much stronger, repeatable signal.
- **A fitted sigma multiplier by (volatility regime, time-of-day session)**
  — CLAUDE.md's learned component #2 (volatility scaling), fit on real GLD
  history via the exact same `fit_vol_multipliers` and session/regime
  bucket definitions the live pattern-mining path (`learning/patterns.py`)
  uses, so a label here means the same thing it would in a live report.
  This is the most literal "learn behaviors of GLD" output: e.g. a bucket
  showing `multiplier=0.70x` means that in low-volatility conditions during
  the NY session, GLD moved noticeably less than the raw model's sigma
  estimate implied, over real historical data. Buckets without at least
  `min_bucket_n` observations aren't shown (not enough data to trust a
  number) rather than defaulting to a misleading 1.0x.

## What it is NOT

This is **not a trading backtest** and produces no P&L, fill rate, or
`Proposal`. There is no historical Kalshi orderbook, spread, or fee data
for past dates — only `record` + `backtest`/`learn` on live-recorded data
can produce that, and CLAUDE.md's promotion gates require real recorded
round trips. This tool only checks the underlying probability model against
real price history; it cannot validate execution, fees, or fills, and its
output never feeds the promotion pipeline automatically. Any change it
suggests (e.g. "raise `min_sigma_per_minute`") is something a human decides
to apply to `config.yaml` by hand, the same as any other engineering change
— it does not go through `promote`.

## Known limitations (read before trusting a number here)

- **GLD tracks spot gold, but isn't spot gold.** It's an ETF that only
  trades NYSE hours (~9:30am–4:00pm ET) and can gap on open relative to
  overnight spot moves. Kalshi's contracts settle on a much closer-to-24/5
  feed. Every window is bounded by `max_window_minutes` specifically so an
  overnight/weekend gap in this NYSE-hours-only series never gets treated
  as "15 minutes of diffusion" — but it also means the intraday check says
  nothing about calibration during hours GLD doesn't trade and Kalshi's
  windows still run.
- **Using the window's own close as the "current" price would be a
  tautology, not a calibration check** — the model would just be told the
  answer and asked to repeat it back. `build_calibration_points` only
  evaluates the model at bars strictly between a window's open and close
  for exactly this reason; this was a real bug in the first draft of this
  tool (every calibration bucket showed exactly 0% or 100% actual win
  rate, a dead giveaway of a lookahead leak), caught by reading the output
  rather than trusting it.
- **Yahoo's free `interval=1d` endpoint silently downsamples very long
  ranges.** `range=max` returned ~263 bars across ~22 years (roughly
  monthly, not daily) when this was verified live. Bounded ranges (`5y`,
  `10y`, `20y`) return true one-bar-per-trading-day data — confirmed live
  before shipping this tool. Keep `--daily-range` at or under ~20y.
- **Daily-horizon calibration is the wrong horizon.** It has real
  statistical power (thousands of days) but says nothing directly about
  15-minute calibration; a formula can be well-calibrated at one horizon
  and not another. Treat the daily result as a coarse sanity check on the
  volatility scaling, and the intraday result (smaller sample, right
  horizon) as the more relevant one for Kalshi's actual contracts.
- **No fitted parameters in the raw formula**, so its full-sample
  Brier/log-loss isn't "in-sample" in the overfitting sense. The isotonic
  recalibration comparison IS trained/held-out split (80/20, chronological)
  specifically to avoid that leakage for the one component that does get
  fit to data.
