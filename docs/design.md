# trade-rl design

This document describes how trade-rl turns market data into scored,
reproducible experiments, and why each piece works the way it does. For the
rules a change must keep, see [AGENTS.md](../AGENTS.md).

```text
trl download ──► CSV files ──► trl bars ──► tradedb ──► features ──► environment ──► agents
(trl-binance)     data/...     (import)     (store)     (causal)     (costs)          │
                                                                                      ▼
           trade-rl viewer ◄── runs/<id>/ (manifest, metrics, equity, learning, policy) ◄── experiment
```

## 1. Storage: series in trade-db

All bars live in one trade-db file (default `data/bars.tradedb`). A series has
a name such as `ETHUSDT/1m` and is stored as:

- **A fixed-width symbol**, `symbol_id("trade-rl/bars/<name>")`. Records are
  keyed by bar open time (epoch ms) and hold 72 bytes:

  ```text
  0 open   8 high   16 low   24 close   32 volume   40 quote   48 buy_volume
  56 trades:i32   60 filled:i32   64 session:i64          (all little-endian)
  ```

  Fixed-width symbols are stored as columns, so `Store::load` reads a series
  with one copy per leaf (`read_columns`).
- **A catalog entry** in the variable-width symbol
  `symbol_id("trade-rl/catalog")`, keyed by the series' symbol id. It holds
  `key=value` lines: name, kind, symbol, interval, market metadata, bar count,
  first and last time, and the data hash.

`Store::save` replaces a series in **one transaction**: drop the symbol,
redefine it, put every bar, overwrite the catalog entry and set the format
key. A reader in another process (the viewer during an import, say) sees the
old series or the new one, never a mixture. `Store::load` reads the catalog
and the bars from one registered snapshot. The database records
`trade-rl/format = 1` in the trade-db metadata, and a file written by an
incompatible version is refused.

A series is **Binance** (continuous 24/7 bars with taker flow; sessions are
UTC days) or **OHLCV** (generic; sessions come from the producer). The data
hash is FNV-1a over symbol, interval and the 72-byte bars, plus, for OHLCV,
the market metadata and sessions.

## 2. Import

**Binance trades** are aggregated into `interval`-wide slots:

- open is the slot's first trade and close its last; high and low are its
  extremes;
- volume and quote are summed, and `buy_volume` counts taker buys;
- an empty slot between two traded ones becomes a *filled* bar carrying the
  last close, unless the gap exceeds 400,000 bars, which marks a hole in the
  archive rather than a quiet market;
- a file that starts inside data already read is refused, since it typically
  overlaps a monthly archive with a daily one.

**Klines** merge into bars at whole multiples of their own interval, which is
also the default. A kline is never split between bars, and a kline that does
not follow the previous one is refused. Timestamps at or above 10¹⁴ are read
as microseconds (Binance spot from 2025). As milliseconds that would be the
year 5138, so the magnitude is unambiguous.

**OHLCV CSV** must satisfy, row by row:

- finite prices with `low > 0` and `low <= open, close <= high`;
- times at least one interval apart;
- nondecreasing sessions, equal to the UTC day when the calendar is `utc`;
- no invented volume, flow or bars.

## 3. Features

Every indicator is a streaming object fed one bar at a time, so none can read
ahead. Two profiles exist:

- **`ohlc-v1`**, for OHLCV series. 11 columns: returns over 1, 5, 15, 60 and
  300 bars (each divided by its horizon's volatility), fast and slow trend
  spreads in ATR units, the volatility ratio, the position in the 300-bar
  range, and the position and distance relative to the previous session's
  range and high.
- **`binance-v1`**. 20 columns: the price features above plus taker flow,
  activity, time of day (two harmonics), day of week, and activity against
  each hour's own causal norm.

The EMA spans are 12, 48 and 192; the volatility windows 60 and 1,440 bars.
Most columns are z-scored with expanding (Welford) statistics and clamped to
±5. Previous-session levels are clamped to ±2 ranges. The warmup is 1,740
bars, extended until a full day (Binance) or the first previous session
(OHLCV) exists. `trl audit` probes eight bars after the warmup: it
recomputes the features over the series truncated at each probe and requires
identical values.

## 4. Environment

The state is the feature row plus the agent's position, its unrealised pnl in
trailing-ATR units (clamped to ±10) and `ln(1 + bars held) / 5`.

- **Timing:** the action at bar `t` is decided at `t`'s close and earns
  `position × ln(close[t+1] / close[t])`.
- **Cost:** each unit of position change costs `cost_bps / 2 + slippage_bps`
  basis points. With `--cost-activity k` (Binance only), the cost is scaled
  by `(trailing trade rate / this bar's trades)^k`, clamped to 0.5–4.
- **Reward:** gross return − cost − `hold_penalty × |position|`.
- **Episode end:** the position is liquidated and the exit is charged. The
  exit enters the reward only for OHLCV series.
- **Shuffled control:** returns are Fisher–Yates shuffled. For OHLCV series
  they are shuffled only within groups of equal elapsed time, so an overnight
  gap never lands inside a session.

Per-bar data shared by every environment (returns, ATR, cost scale) is
computed once per experiment in `EnvData`.

## 5. Agents and the network

`flat`, `long` and `short` hold a constant position. `random` flips with
probability 0.02 per bar. `crossover` goes long or short when `trend_fast` is
beyond ±0.5, and holds inside the band. `breakout` is long while the close is
above the previous session's high.

`qtable` uses 225 states: trend buckets × 60-bar return buckets × volatility
buckets × position. The learning rate is 0.1, the discount 0.99 and the reward
scale 10⁴, and ε decays from 1 to 0.05 over 200,000 steps.

`dqn` is a double DQN:

- hidden layers 64,64 with ReLU and a linear output;
- Adam at 3e-4, with optional decay per epoch;
- 200,000 replay transitions, a 5,000-step warmup, batches of 32 trained every
  4 steps, and a target network synced every 2,000 steps;
- Huber loss with δ = 1;
- an optional `switch_margin`, which requires the Q edge to pay for a move.

The **network** (`mlp.rs`) is a small multilayer perceptron:

- binary32 weights, laid out per layer as `[out][in]` followed by biases;
- He-uniform initialisation for ReLU layers and Glorot-uniform for the linear
  output, from SplitMix64 seeded with `mix64(seed)` folded to a nonzero `u32`;
- gradients summed over the batch;
- an Adam step in binary64 on the mean gradient, clipped to a global norm of
  10.

Dot products always run in eight lanes, combined in a fixed pairwise order;
there is no FMA and no parallelism, so the bits are identical everywhere.
Manifests record this as `dqn_numerics`.

The **ensemble** is a majority vote of the seeds' greedy actions. A tie holds
the current position, since changing on a split vote would manufacture
turnover.

## 6. Experiments

**Folds:** train, gap, an optional validation window followed by another gap,
then test. Folds advance by the test length. Seeds are
`base + i × 7919`, and all seeds train side by side within each fold.

Within each fold, every seed is trained for `epochs` passes of
`episode`-length episodes, then scored greedily on the test window. That
includes the validation window, the cost-curve variants and the shuffled
control. The ensemble is then scored the same way. Everything is evaluated
inside the fold loop, so all variants see the policy at the same stage of
learning.

`metrics.csv` has one row per variant, seed, cost and fold, where fold `-1`
is the pooled result:

- the agent per seed, at the experiment cost and at each cost-curve level;
- `<agent>:valid`;
- `control:shuffled` per seed;
- `ensemble` at every cost;
- `control:ensemble-shuffled`;
- `control:ensemble-rotated`: the median by Sharpe of eight rotations
  (offsets `n × k / 9`). It keeps the position sequence, exposure and costs,
  and destroys only its alignment with returns.
- the five baselines.

The console adds the break-even round-trip cost of each variant, a table by
calendar year (or month) with a rotation inside each period, and the band of
net returns across seeds, which is the noise floor. A walk **exits 1** when
the shuffled ensemble's Sharpe beats the real one and exceeds 0.5.

**Artifacts** are assembled in memory and published by renaming a staged
directory, so `runs/` never holds a partial run:

| File | Contents |
|------|----------|
| `manifest.txt` | `key=value`: the contract, every setting, the data hash, the store and series |
| `metrics.csv` | `variant,seed,cost_bps,fold,` + 17 metrics |
| `equity.csv` | The headline's per-bar time, price, position, gross, cost and cumulative net |
| `learning.csv` | Seed 0's mean reward, ε and loss per episode (learners only) |
| `policy.csv` / `weights.bin` | Seed 0's Q-table or network (`TRLMLP01`, little-endian) |

The **contract** is everything results depend on beyond the settings: bar
format, feature profile, schema, columns and warmup, normalisation, decision
time, horizon, annualisation, and for OHLCV series the market metadata and
control definitions. `trl verify`:

1. reloads the series and checks its data hash;
2. checks the contract and the DQN numerics;
3. rebuilds the configuration from the manifest;
4. re-runs it under `runs/verify`;
5. requires every recorded artifact to be byte-identical.

## 7. Patterns

`trl patterns` buckets each bar's forward return by:

- hour of day and weekday (Binance only);
- distance to the previous session's high, low and close, and position in its
  range;
- participant proxies (Binance only): activity surprise, trade size, close
  against VWAP, and impact per unit volume.

It reports mean, standard error, t and both halves of the sample for drift and
for one-bar momentum.

The null is the 95th percentile of the largest |t| over permutations of whole
blocks of returns, with bucket membership held fixed. The block is a day of
bars, or `--block`; OHLCV series must pass `--block`. A null that never moves
is reported as UNTESTED. **PAYS** needs both a |t| beyond the null and a mean
position run longer than the bars needed to cover the round trip.

## 8. Acquisition

`trl-binance` builds `https://data.binance.vision/data/{spot|futures/um|futures/cm}/...`
names and validates the symbol, interval and date before any request. Each
archive goes through these steps:

1. **GET over rustls** with webpki roots: no redirects, no cookies, no
   retries, a body limit and a ten-minute ceiling.
2. **Mandatory `.CHECKSUM`:** an exact `<hex>␠␠name` format and the SHA-256
   of the compressed bytes. A missing sidecar is a checksum failure, so
   `--skip-missing` cannot bypass it.
3. **Strict ZIP:**
   - exactly one entry, the expected name, local header at offset 0;
   - local and central records agree, and the central directory fills its
     space;
   - stored or deflated only; no encryption, links or directories; ZIP64 and
     data descriptors understood;
   - the declared size is limited, and the output must match its size and CRC.
4. **Typed CSV** (klines and trades):
   - the exact column count, and an optional known header;
   - nonnegative JSON-number lexemes, exact u64 ids and counts, strict
     booleans;
   - the timestamp unit chosen by market and archive date;
   - every row inside the archive's own period, and a row limit.
5. **Publication:** write and `fsync` `<name>.part` (created exclusively),
   then hard-link it into place, which never replaces a name. `--force` first
   moves the old file to `.previous`, and never overwrites an existing
   `.previous`.

## 9. Viewer

`trade-rl` keeps a read-only model: the series, plus a run's positions and
equity matched to bars by open time. With a run, the view opens on the span
of bars the run scored; without one, on the first 20,000 bars. `render.rs`
draws the chart as pixels. Every screen column aggregates its bars: the
range, the last close, the majority position (long wins ties, then short),
and the last equity. A minimap reduces the whole series to one column per
pixel, rebuilt only when the width changes.

The raster is rendered at physical resolution (the window's scale factor) and
shown through an iced canvas as horizontal tiles under 2 MiB. iced uploads
images that size synchronously but larger ones on a worker thread, which
would leave the chart blank while it changes. Tiles are cached until the view
changes. `--render` draws the same raster to a PNG with no window; CI uses it.

## 10. Determinism

Results must be bit-identical across platforms, compilers and optimisation
levels:

- transcendental functions come from `libm` (a musl port), never the
  platform's maths library;
- randomness is SplitMix64, seeded explicitly;
- reductions run in a fixed order;
- floats are formatted with explicit precision.

CI runs identical walks on Windows, Linux and macOS and diffs the artifact
checksums. The project was checked the same way between Windows (stable) and
Linux (Rust 1.88).
