//! The trade-rl research engine.
//!
//! An agent picks a target position (short, flat or long) after each bar
//! closes; walk-forward experiments compare it with baselines and controls
//! after costs. Everything here is deterministic: the same bars, settings and
//! seed produce byte-identical artifacts on every platform, which is what
//! `trl verify` checks.
//!
//! Bars live in a [`tradedb`] database ([`store::Store`]), one fixed-width
//! symbol per series, so many series share one crash-safe file that a
//! viewer can read while an import writes.

pub mod agents;
pub mod audit;
pub mod bars;
pub mod env;
pub mod error;
pub mod experiment;
pub mod features;
pub mod import;
pub mod math;
pub mod metrics;
pub mod mlp;
pub mod patterns;
pub mod rng;
pub mod run;
pub mod store;
pub mod synthetic;
pub mod text;
pub mod time;

pub use bars::{Bar, Market, Series, SeriesKind};
pub use error::{Error, Result};
pub use store::{SeriesInfo, Store};
