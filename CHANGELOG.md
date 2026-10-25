# Changelog

## 0.1.0 — 2026-10-25

First release.

- Bars stored in [trade-db](https://github.com/ZacWalk/trade-db): one
  fixed-width symbol per series plus a catalog entry, replaced atomically.
- Imports: Binance trades and klines (millisecond and microsecond archives),
  and generic OHLC(V) CSV with producer sessions.
- `ohlc-v1` and `binance-v1` causal feature profiles, with the shift audit.
- Agents `flat`, `long`, `short`, `random`, `crossover`, `breakout`,
  `qtable` and `dqn`, plus the seed-ensemble vote.
- Walk-forward experiments with a cost curve, baselines, shuffled and rotated
  controls, a calendar period table, and byte-identical `verify`.
- `patterns` bucket study with a block-permutation null.
- Checksum-verified Binance acquisition with a strict ZIP reader, typed CSV
  validation, size limits, fixtures and no-replace publication.
- `trade-rl` iced viewer, with a headless `--render` and a `--smoke-test`
  window capture.
- DQN numerics `trl-mlp-f32-lanes8-he-splitmix64-v1`: runs reproduce
  byte for byte across Windows, Linux and macOS.
