# trade-rl guide

How to install trade-rl, get data in, run experiments and read the results.
[design.md](design.md) explains how it works inside.

## Install

You need Rust 1.88 or newer. trade-rl builds
[trade-db](https://github.com/ZacWalk/trade-db) from source, so check both
repositories out side by side:

```sh
git clone https://github.com/ZacWalk/trade-db
git clone https://github.com/ZacWalk/trade-rl
cd trade-rl
cargo install --path crates/trl-cli    # trl, the command line
cargo install --path crates/trl-app    # trade-rl, the viewer
```

On Linux, the viewer needs the usual windowing development packages
(`libxkbcommon-dev`, `libwayland-dev`, `pkg-config`).

## Commands

| Command | What it does |
|---------|--------------|
| `trl download` | Fetches checksum-verified Binance archives. The only network command. |
| `trl bars` | Imports Binance trade/kline CSVs or an OHLCV CSV (`--format ohlcv`). |
| `trl series list\|show\|export\|remove` | Manages stored series. `export` writes CSV; OHLCV exports re-import exactly. |
| `trl synth` | Stores a deterministic synthetic series: `trend`, `session` or `walk`. |
| `trl walk` | Trains and evaluates an agent walk-forward, reporting baselines, the cost curve, controls and a period table. |
| `trl audit` | Shift audit: recomputes features over truncated series and requires identical values. |
| `trl patterns` | Bucket study: tests whether hour, weekday, previous-session levels or flow proxies carry forward return, against a block-permutation null. |
| `trl report` / `trl list` | Shows one run's manifest and pooled results, or lists all runs. |
| `trl verify` | Re-runs a recorded run under `runs/verify` and compares every artifact byte for byte. |

Run `trl <command> --help` for options. Every command takes `--db`
(default `data/bars.tradedb`). Commands that read a series take `--series`,
which may be omitted when the database holds only one.

Exit codes: **0** success; **1** a control, the audit or a reproduction
failed; **2** bad input or an error.

## Getting data

### Binance

```sh
trl download --symbol ETHUSDT --kind klines --source 1m --from 2024-01-01 --to 2024-06-30
trl bars --in data/klines/ETHUSDT/1m          # stored as ETHUSDT/1m, ~260k bars
```

For bars finer than a minute, or with exact taker flow, download `--kind trades`
(daily archives, much larger) into `data/raw` and import them with
`trl bars --in data/raw --interval 10s`.

Every archive must match its `.CHECKSUM` sidecar (SHA-256) before its ZIP is
opened. The ZIP must hold exactly one member with the expected name, which is
checked against its declared size and CRC. Klines and trades are then
validated row by row. A file appears under its final name only once complete.
Existing data is never replaced unless you pass `--force`, which keeps the old
file as `.previous`. `--fixtures DIR` serves archives from a directory
instead, which is how the tests run offline.

Kline files aggregate to any whole multiple of their own interval. Trades
aggregate to any interval, and quiet intervals become filled bars carrying the
last close.

### Your own OHLC data

```sh
trl bars --format ohlcv --in prices.csv --symbol SPX --interval 15m \
         --calendar nyse-sessions --currency USD --source-name vendor
```

The CSV has the header `time,open,high,low,close[,volume][,session]`, with
columns in any order:

- `time` is the bar's open in UTC epoch milliseconds.
- With `--calendar utc`, sessions are UTC days. Any other calendar needs an
  integer `session` column from the data producer, so session-based features
  never guess a trading day.
- Gaps stay gaps: no bars, volume or order flow are invented.

These series use the price-only `ohlc-v1` feature profile, and their controls
respect elapsed time.

### Synthetic data

`trl synth --kind trend|session|walk` stores a deterministic series that
needs no network, for trying everything out:

```sh
trl synth --kind trend --bars 16000
trl audit
trl walk --agent breakout --seeds 2 --train 5000 --test 2000 --gap 300 --episode 1000 --epochs 1
```

## Agents

| Agent | Policy |
|-------|--------|
| `flat`, `long`, `short` | A constant position: the baseline every learner must beat. |
| `random` | Flips position with probability 0.02 per bar: the real cost floor. |
| `crossover` | Long or short on the fast/slow trend spread, with a dead band. |
| `breakout` | Long above the previous session's high, as a hand-coded rule. |
| `qtable` | A 225-state tabular Q-learner, small enough to read in full (`policy.csv`). |
| `dqn` | Double DQN with replay, a target network and Huber loss. The policy is saved as `weights.bin`. |

Each learner is trained on several seeds, and the seeds' vote is the headline
result. Every result appears beside a cost curve (1, 2, 5 and 10 bps), five
baselines, a shuffled-returns control and a rotated-positions control, broken
down by calendar period. A walk exits 1 when the shuffled control beats the
real result.

## The viewer

`trade-rl` is read-only: it never trains and never writes the database.

```sh
trade-rl [--db data/bars.tradedb] [--series NAME] [--run runs/<id>]
trade-rl --run runs/<id> --render chart.png --size 1400x900   # headless, no window
```

It draws price, the position band (green long, red short, grey flat), equity
with its drawdown shading, and a minimap of the whole series. With `--run` it
opens on the bars the run traded, and finds its database and series from the
run, so it can be given alone. Drag to pan, use the wheel to zoom, Shift+wheel
to scroll, and click the minimap to jump; Home, End and the arrow keys also
work. Each screen column summarises the bars that fall in it, so a million
bars draw as fast as a thousand.

## Workspace

```text
crates/trl-core     engine: bars and the tradedb store, import, features, environment,
                    agents, MLP, experiments, metrics, runs, patterns, audit
crates/trl-binance  acquisition: names, HTTPS (rustls), checksum, strict ZIP, CSV validation
crates/trl-cli      `trl`, the command line
crates/trl-app      `trade-rl`, the iced viewer and its headless renderer
```

The tests use synthetic data and fixture archives only. CI runs the same
walks on Windows, Linux and macOS and requires byte-identical artifacts. See
[AGENTS.md](../AGENTS.md) for the invariants a change must keep.
