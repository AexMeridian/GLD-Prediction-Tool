# Contract & API research notes

Researched 2026-09-16 by fetching live endpoints directly (curl + web search
against official docs). Sources are inline. **Two findings below contradict
CLAUDE.md and need a decision before Phase 1/2 code is written — see
"⚠️ Contradictions" at the top.**

## ⚠️ Contradictions with CLAUDE.md

### 1. Settlement source is NOT the standard XAU/USD feed
CLAUDE.md says to stream Pyth's `XAU/USD` feed as both price source and
settlement source. Live data shows Kalshi's `KXGOLD15M` series actually
settles on a **different Pyth symbol**:

- Fetched `GET /trade-api/v2/series/KXGOLD15M` → `settlement_sources: [{"name":
  "Pyth - Gold", "url": ".../explore/Metal.Index.GOLD%2FUSD"}]`.
- Pyth Hermes has two distinct gold products:
  | Symbol | Feed ID | Behavior |
  |---|---|---|
  | `Metal.XAU/USD` | `765d2ba906dbc32ca17cc11f5310a89e9ee1f6420508c63861f2f8ba4ee34bb2` | Standard spot gold. Has real market hours — closes daily ~17:00–18:00 America/New_York and around holidays (`market_hours.is_open` was `false` when checked at 17:30 ET on a weekday). |
  | `Metal.Index.GOLD/USD` | `fa0f57505be633c026896e15afef2c7ce2cf8ff9a45349d1da737f4f01266b01` | "PYTH PRICE IN USD FOR GOLD 24/7" — always open, no daily halt. |
- Kalshi's `KXGOLD15M` windows run 24/7 on weekdays, including through the
  standard feed's daily halt (the open market we fetched has
  `open_time: 2026-09-16T21:30:00Z` = 5:30pm ET, inside the XAU/USD daily
  break). That's only possible if Kalshi is settling on the 24/7 index feed,
  which matches the `settlement_sources` URL exactly.
- **Recommendation:** `pyth.py` / `windows.py` should track
  `Metal.Index.GOLD/USD` (id above) as the settlement-relevant price, not
  `Metal.XAU/USD`. Do not hardcode the feed ID — resolve it at startup via
  Hermes's `/v2/price_feeds?query=...` lookup or a config value, and log a
  loud warning if the series' `settlement_sources` field ever points
  somewhere else, because:
- **Update 2026-09-17 — the announced change happened, live, exactly as
  predicted.** Kalshi's series data now reads: *"Weekend trading now
  available... These markets now use a new gold price feed from Pyth,
  Metal.Index.1OZGOLD/USD (feed ID 3712), for settlement."* The runtime
  check in `windows.py` (`check_settlement_source`) caught this itself the
  first time `live` was run today — it logged a mismatch warning because
  `config.yaml` still pointed at `Metal.Index.GOLD/USD`. Confirmed the new
  feed via `GET /v2/price_feeds?query=1OZGOLD` →
  `Metal.Index.1OZGOLD/USD`, id
  `7fd2c87083dd8fd5af2486a43f40b8e444acd77fda4e96263a53421cd079c387`
  ("PYTH PRICE IN USD FOR 1-OUNCE GOLD 24/7" — same 24/7 behavior as the
  old feed, just a different underlying product). `config.yaml` has been
  updated to this new symbol/id. This is exactly the failure mode the
  runtime check exists for — expect it to fire again if Kalshi changes
  the settlement source a third time.

### 2. Pyth Hermes now requires an API key (breaking change, effective Aug 2026)
CLAUDE.md assumes anonymous SSE streaming from `hermes.pyth.network`. As of
the **Pyth Core upgrade on 2026-08-26**, Hermes requires authentication:
- Old host `https://hermes.pyth.network` still resolves but now returns
  `401 unauthorized` on price-update routes (confirmed live: `curl
  .../v2/updates/price/latest?ids[]=...` → `401 unauthorized` for both the
  XAU/USD and index feed IDs).
- New canonical host: `https://pyth.dourolabs.app/hermes/` — "routes and
  response shapes are unchanged, drop-in replacement" per Pyth's own
  migration guide, plus an `Authorization: Bearer $PYTH_API_KEY` header on
  every request, **including the SSE stream endpoint**.
- Streaming path becomes:
  `https://pyth.dourolabs.app/hermes/v2/updates/price/stream?ids[]=<hex_id>`
  with header `Authorization: Bearer <PYTH_API_KEY>`.
- Getting a key: sign up at Pyth Terminal and copy the key from "View your
  API key". Now resolved by direct testing: **the free key works fine for
  standard feeds (e.g. `Metal.XAU/USD`) but not for "Index" feeds.**

### 3. Confirmed 2026-09-17: Pyth "Index" feeds need a separate paid entitlement
This directly affects Gold Edge because Kalshi settles `KXGOLD15M` on an
Index feed (`Metal.Index.1OZGOLD/USD`, see #1 above), not a standard feed.
Tested live with the user's real (free-tier) `PYTH_API_KEY`:
- `GET .../v2/updates/price/latest?ids[]=<Metal.XAU/USD id>` → `200 OK`,
  real price data.
- `GET .../v2/updates/price/latest?ids[]=<Metal.Index.1OZGOLD/USD id>` →
  `403 Forbidden`: *"Not entitled: feed ... (no grant accepts this gated
  feed; it requires access to one of the following groups:
  ["pyth-indices"])"*.

So the free tier can stream `Metal.XAU/USD` (or any other standard feed)
but not the Index family. This reopens the decision in #1: either (a) the
user upgrades their Pyth plan to get the `pyth-indices` grant so the tool
tracks the exact feed Kalshi settles on, or (b) the tool runs on the free
`Metal.XAU/USD` feed as an approximation — accepting that it will diverge
from Kalshi's actual settlement source, particularly during XAU/USD's
daily trading halt (see #1's table), which Kalshi's 24/7 windows trade
straight through. Option (b) is a real accuracy compromise, not just a
technicality — CLAUDE.md's core principle is "price source = settlement
source." Whichever the user picks, `config.yaml`'s `pyth.price_feed_symbol`
/ `price_feed_id` should be updated accordingly, and if (b), the settlement
source mismatch warning from `windows.py` will fire continuously by design
(it's now telling the truth: the live feed doesn't match the settlement
feed) — that warning should not be "fixed" by silencing it, only by
picking option (a).

**Decision (user, 2026-09-17): option (b).** Running on the free
`Metal.XAU/USD` feed for now rather than paying for the Pyth Indices
upgrade. `config.yaml` set accordingly. Revisit if live results look
meaningfully worse than backtests, especially for windows that open/close
near XAU/USD's daily halt.
- **Action needed:** add `PYTH_API_KEY` to `.env` / `.env.example` and
  `config.py` settings, alongside the existing `KALSHI_API_KEY_ID` /
  `KALSHI_PRIVATE_KEY_PATH`. `feeds/pyth.py` must send the bearer header on
  both REST and SSE calls.

**Correction (2026-09-18) — the closure is weekly, not a ~1hr daily halt.**
Live recording caught `Metal.XAU/USD` frozen for 3+ hours straight (same
price, same `publish_time`, both the SSE stream and the REST `/latest`
endpoint) starting Friday ~17:00 ET and still closed as of Friday 19:58 ET.
Fetching `/v2/price_feeds?query=XAU` mid-freeze shows why — the feed's
`schedule` field (`America/New_York;0000-1700&1800-2400,...,0000-1700,C,
1800-2400`) gives Friday as `0000-1700` (no evening reopen) and Saturday as
`C` (fully closed all day); `market_hours.next_open` resolves to Sunday
~18:00 ET. So the real gap is **Friday ~17:00 ET through Sunday ~18:00 ET —
essentially the whole weekend**, not a short daily dip. Since Kalshi's
KXGOLD15M now trades weekends (see the 2026-09-17 update above), this is
the single biggest hole in the free-tier setup, bigger than originally
scoped here.
- **Mitigation shipped 2026-09-18:** `feeds/gold_proxy.py` streams PAXG
  (tokenized, physically-redeemable gold) from Coinbase's free, keyless
  public WebSocket as a 24/7 fallback, basis-corrected against real Pyth
  spot via `model/basis.py` whenever both are live. `live` mode
  (`server.py`) switches to it automatically once `Metal.XAU/USD` is
  confirmed stale/closed; `record` captures both streams continuously,
  tagged by source, so recorded data is never ambiguous about which one
  produced a given tick. This does not close the accuracy gap versus the
  paid `pyth-indices` entitlement Kalshi actually settles on — PAXG is a
  correlated proxy, not the settlement source — but it's a real
  improvement over having zero live price for ~49 hours a week.
- Both `Crypto.PAXG/USD` and `Crypto.XAUT/USD` were checked directly against
  this Pyth account and returned `"Not entitled... asset type 'crypto'"` —
  the free plan doesn't grant Pyth's own crypto feeds either, which is why
  the fallback goes to Coinbase/Kraken's exchange APIs directly instead.

## Kalshi Trade API v2 (confirmed live)

- **REST base (production, confirmed working via direct request):**
  `https://api.elections.kalshi.com/trade-api/v2`
  - Public/unauthenticated endpoints (market discovery, series, rules) work
    with no auth headers — confirmed by fetching `/series/KXGOLD15M` and
    `/markets?series_ticker=KXGOLD15M` with a bare `curl`.
- **WebSocket (production):** `wss://api.elections.kalshi.com/trade-api/ws/v2`
  - Subscribe message shape: `{"id": 1, "cmd": "subscribe", "params":
    {"channels": ["orderbook_delta"], "market_ticker": "..."}}`.
  - First response after subscribing is `orderbook_snapshot` (full book),
    then incremental `orderbook_delta` messages.
  - WS connections also authenticate at handshake time with the same three
    headers as REST (see below), signing method `GET` and path
    `/trade-api/ws/v2`.
- **Authentication (REST and WS):** headers `KALSHI-ACCESS-KEY`,
  `KALSHI-ACCESS-TIMESTAMP` (ms since epoch), `KALSHI-ACCESS-SIGNATURE`.
  - Signed message = `timestamp_ms + HTTP_METHOD + path` (path includes the
    `/trade-api/v2` prefix, excludes query string).
  - Signature = RSA-PSS, SHA-256 for both hash and MGF1, salt length = digest
    length (32 bytes), base64-encoded.
  - This matches CLAUDE.md exactly — no change needed.
- Other hostnames turned up in secondary sources (`external-api.kalshi.com`,
  `trading-api.kalshi.com`, `external-api.demo.kalshi.co` for demo) were
  **not verified live**; `api.elections.kalshi.com` is the one confirmed to
  return real data and is what `kalshi_rest.py` / `kalshi_ws.py` should use.
  If auth calls to it fail once we have real keys, check whether Kalshi has
  since moved to one of these alternates.

## KXGOLD15M contract facts (confirmed against a live market)

Fetched `GET /trade-api/v2/series/KXGOLD15M` and one open market
(`KXGOLD15M-26SEP161745-45`, window 2026-09-16 21:30–21:45 UTC):

- `fee_multiplier: 1`, `fee_type: "quadratic"` → confirms
  `fee = ceil_to_cent(1 * 0.07 * C * P * (1-P))` for this series, matching
  CLAUDE.md's formula with `M = 1`. (Verify `M` per-market going forward —
  CLAUDE.md is right to keep it in config rather than hardcoding, since
  other series can carry a different multiplier.)
- No settlement fee, and maker (resting-order) fees are live as of
  2026-08-19 23:59 ET — matches CLAUDE.md.
- `rules_primary` (verbatim from the live market): *"If the close price of
  the 1-minute candlestick for Gold on Sep 16, 2026 at 5:45 PM EDT is at
  least the close price of the 1-minute Pyth GOLD candlestick at 5:30 PM
  EDT on September 16, 2026 ..., then the market resolves to Yes."*
  `strike_type: "greater_or_equal"` → confirms ties resolve YES, matching
  CLAUDE.md.
- `rules_secondary` adds a subtlety CLAUDE.md doesn't mention: *"the close
  price for the 1-minute candlestick at a given time is the price at the
  end of the immediately preceding one-minute interval"* — e.g. the
  candle timestamped 4:59 PM covers 4:59:00–4:59:59 and closes at 5:00:00
  PM. `windows.py` / settlement replication must align candle boundaries
  this way, not naively bucket by wall-clock minute.
- Fallback rule (not in CLAUDE.md): *"If no data is published by the
  specified source agency for the specified time, then the most recently
  available published data will be used."* Relevant to the "data failure"
  exit rule and to settlement replication in backtesting.
- `custom_strike.round_digits: "2"` and settlement value rounded to nearest
  2 decimal places — matches CLAUDE.md.
- Reference price (`S0`) is published on the market itself as `floor_strike`
  (e.g. `4271.08`) and mirrored in `yes_sub_title` ("Target Price:
  $4,271.08") — known at market open, before the window starts trading in
  earnest. `windows.py` should read `floor_strike`, not try to derive S0
  itself from the first Pyth tick.
- `open_time` / `close_time` confirm exactly a 15-minute window
  (`21:30:00Z` → `21:45:00Z`).
- `settlement_timer_seconds: 1` — settlement is essentially immediate after
  close.
- Price tick structure (`price_level_structure: "tapered_deci_cent"`) is
  finer near the edges than CLAUDE.md's examples assume: steps of `0.001`
  for prices in `[0, 0.10]` and `[0.90, 1.00]`, `0.01` in between. Relevant
  to spread/gap rounding in `fees.py` / `signals.py`.

## Settlement value is public (confirmed live) — enables `validate-feed`

Fetching `GET /markets?series_ticker=KXGOLD15M&status=settled&limit=1` (no
auth needed — same public endpoint as market discovery) on a finalized
market returns `expiration_value: "4272.59"` — the actual 1-minute candle
close price Kalshi used to settle — alongside `floor_strike` (the open/S0),
`result` (`"yes"`/`"no"`), and `settlement_ts`. This means `validate-feed`
can compare our own recorded Pyth candle closes against Kalshi's published
`expiration_value` for every settled window using only public REST calls —
no need for the authenticated `/portfolio/settlements` endpoint (which only
covers positions the account actually held).

## Kalshi orderbook mechanics (confirmed via docs.kalshi.com)

Kalshi's book stores **bids only**, for both YES and NO — there is no
separate ask side. Asks are derived from the reciprocal relationship in a
binary market: `yes_ask = 1 - no_bid`, `no_ask = 1 - yes_bid` (confirmed
against the live market fetched above: `yes_bid=0.37, yes_ask=0.39,
no_bid=0.61, no_ask=0.63` → `1 - 0.61 = 0.39` ✓, `1 - 0.37 = 0.63` ✓).
`kalshi_ws.py` maintains two price→size maps (yes bids, no bids) from
`orderbook_snapshot` + `orderbook_delta`, and derives all four
bid/ask/size values `BookSnapshot` needs from those two maps. The exact
delta wire format (field names, whether the delta is a signed increment
vs. an absolute size) was **not confirmed against official docs** — only
reconstructed from a third-party source — so treat the parsing in
`kalshi_ws.py` as provisional until it's checked against a real
authenticated connection (needs your API key) in early Phase 1 testing.
Sequence numbers are present on deltas; the client should track them and
force a reconnect/resnapshot on any gap.

## Pyth Hermes (confirmed live)

- Feed lookup: `GET https://hermes.pyth.network/v2/price_feeds?asset_type=metal&query=<term>`
  still works unauthenticated (metadata-only endpoint, not a price-update
  route) and is how the two feed IDs above were found — don't hand-guess
  IDs, resolve them this way at startup or pin them with a comment showing
  how they were obtained.
- Price-update routes (`/v2/updates/price/latest`, `/v2/updates/price/stream`)
  require the bearer key as described above, on both the legacy
  `hermes.pyth.network` host and the new `pyth.dourolabs.app/hermes` host.
- SSE stream auto-closes after 24h; client must reconnect (already covered
  by CLAUDE.md's reconnect/backoff requirement).

## Open questions for the user

1. Do you already have (or want to sign up for) a Pyth Terminal account to
   get `PYTH_API_KEY`? Needed before Phase 1's live feed work; `record`
   can't pull real ticks without it.
2. OK to track `Metal.Index.GOLD/USD` (24/7 index) instead of
   `Metal.XAU/USD` as the live price source, matching Kalshi's stated
   settlement source?
3. Kalshi's own notice says the settlement feed changes again tomorrow
   (2026-09-17 05:15 ET). Want me to re-check `settlement_sources` after
   that time before we lock in Phase 1, or proceed now with the
   read-it-at-runtime approach above so it self-adjusts?
