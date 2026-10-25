# trade-rl — notes for agents and contributors

trade-rl is an offline research bench: an agent chooses short, flat or long
after each bar closes, and walk-forward experiments score it against baselines
and controls after costs. Bars are stored in
[trade-db](https://github.com/ZacWalk/trade-db) (crate `tradedb`), which is
checked out beside this repository (`tradedb = { path = "../trade-db" }`).

## Layout

```text
crates/trl-core     engine (no I/O beyond the bar store and run directories)
  bars.rs           Bar, Series, Market, the data hash, validation
  store.rs          series in tradedb: fixed-width symbol + catalog entry, atomic replace
  import.rs         OHLCV CSV, Binance trade/kline aggregation
  features.rs       streaming indicators, ohlc-v1 / binance-v1 profiles
  audit.rs          shift audit
  env.rs            trading environment, costs, shuffled-returns control
  mlp.rs            the network
  agents.rs         constant, random, crossover, breakout, qtable, dqn, ensemble
  experiment.rs     folds, training, evaluation, controls, manifest, verify
  metrics.rs, run.rs, patterns.rs, synthetic.rs, rng.rs, math.rs, text.rs, time.rs
crates/trl-binance  the only network code: names, HTTPS, checksum, ZIP, CSV, publication
crates/trl-cli      `trl` (clap); integration tests in tests/cli.rs
crates/trl-app      `trade-rl` (iced 0.14): model.rs, render.rs (pure raster), app.rs
```

## Invariants a change must keep

1. **Causality.** A decision at bar `t` uses only bars up to `t`'s close and
   earns the return from `t` to `t + 1`. Features are streaming objects that
   never hold the series. Standardisation uses expanding statistics. Fills are
   at the decision close plus slippage, never inside the bar's range.
   `trl audit` and `audit::tests` must keep passing.
2. **Determinism, bit for bit, on every platform.** Runs are reproduced by
   `trl verify` and compared across Windows, Linux and macOS in CI.
   - Use `trl_core::math` (`libm`) for `ln`, `exp`, `sin`, `cos` and `pow`,
     never the `f64` methods, whose results differ by platform. `sqrt` is
     exact and is fine.
   - Randomness comes only from `trl_core::rng::Rng` (SplitMix64), seeded from
     the run's seeds. Never use a `rand` crate, time or addresses.
   - Keep the MLP's reductions in their fixed order. Do not use `mul_add` or
     parallelism in anything an artifact depends on.
   - Never let `HashMap` iteration order reach an output.
   - Artifacts format numbers with explicit precision (`text::fixed`,
     `text::general`, `text::exact`).
   - A change that alters any artifact's bytes changes results. Note it in
     CHANGELOG.md, and change `DQN_NUMERICS` if the network's numerics moved.
3. **Honest scoring.** Every headline appears beside the cost curve,
   baselines, the shuffled and rotated controls, and the period table. The
   test windows are scored once and never searched over. Parameters are chosen
   on `--valid`. The saved policy is seed 0's, fixed before any test window was
   scored.
4. **Exit codes.** 0 success; 1 a control, the audit or a reproduction failed;
   2 bad input or an error. A run that fails validation writes no run
   directory, because runs are published atomically.
5. **Network isolation.** Only `trl-binance` touches the network, and only for
   `trl download`. Checksums are mandatory and there is no bypass. Tests use
   `--fixtures` or `FixtureFetch`, never the network.
6. **The viewer is read-only.** `trade-rl` opens the database read-only and
   never writes runs.
7. **Generic OHLC series** never invent flow, volume or bars. Sessions come
   from the data producer unless the calendar is `utc`, and
   `--cost-activity` is refused for them.

## Commands

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo run -p trl-cli -- --help
cargo run -p trl-app -- --db data/bars.tradedb --render chart.png   # headless check
```

`.\run.ps1` starts the viewer on Windows. `-SmokeTest out.png` opens the
window, captures it and exits.

## Conventions

- Edition 2024 with MSRV 1.88, in step with trade-db, trade-up and list-zero.
  CI tests the MSRV with `RUSTUP_TOOLCHAIN=1.88`, because
  `rust-toolchain.toml` would otherwise override it.
- `unsafe_code = "deny"` across the workspace.
- No GPL or LGPL dependencies (`deny.toml`).
- Error messages name the flag to change and the value that was refused.
- Comments explain why, not what.
