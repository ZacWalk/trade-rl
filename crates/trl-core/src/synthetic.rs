//! Deterministic synthetic series for tests, demos and smoke checks, so no
//! test ever needs the network.

use crate::bars::{Bar, Market, Series, SeriesKind};
use crate::math;
use crate::rng::Rng;
use crate::time::{MS_PER_DAY, days_from_civil};

/// 2020-01-01 00:00 UTC.
pub const START_MS: i64 = 1_577_836_800_000;

/// A trending, oscillating 15-minute Binance-style series with flow.
pub fn trend_series(n: usize) -> Series {
    let mut bars = Vec::with_capacity(n);
    let mut last = 1000.0;
    for i in 0..n {
        let x = i as f64;
        let close = 1000.0 + x * 0.01 + 20.0 * math::sin(x * 0.017) + 5.0 * math::sin(x * 0.31);
        let volume = (100 + i % 37) as f64;
        bars.push(Bar {
            time: START_MS + i as i64 * 900_000,
            open: last,
            high: last.max(close) + 0.5,
            low: last.min(close) - 0.5,
            close,
            volume,
            quote: close * volume,
            buy_volume: (40 + i % 31) as f64,
            trades: (20 + i % 17) as i32,
            filled: 0,
        });
        last = close;
    }
    Series::binance("ETHUSDT", 900_000, bars)
}

/// The generic OHLC fixture: 26 fifteen-minute bars per session from
/// 14:30 UTC, sessions numbered from 1, two missing days after every five,
/// no volume.
pub fn session_series(n: usize) -> Series {
    let start = days_from_civil(2020, 1, 2) * MS_PER_DAY + 14 * 3_600_000 + 30 * 60_000;
    let mut bars = Vec::with_capacity(n);
    let mut sessions = Vec::with_capacity(n);
    let mut previous = 100.0;
    for i in 0..n {
        let day = (i / 26) as i64;
        let time = start + (day + 2 * (day / 5)) * MS_PER_DAY + 900_000 * (i % 26) as i64;
        let x = i as f64;
        let close = 100.0 + 0.001 * x + 2.0 * math::sin(x * 0.017) + 0.5 * math::sin(x * 0.31);
        bars.push(Bar {
            time,
            open: previous,
            high: previous.max(close) + 0.05,
            low: previous.min(close) - 0.05,
            close,
            ..Bar::default()
        });
        sessions.push(day + 1);
        previous = close;
    }
    Series {
        symbol: "TEST".into(),
        interval_ms: 900_000,
        kind: SeriesKind::Ohlcv,
        market: Market {
            source: "synthetic".into(),
            currency: "USD".into(),
            calendar: "fixture-sessions".into(),
            price_basis: "raw".into(),
            volume_available: false,
        },
        bars,
        sessions,
    }
}

/// [`session_series`] as the CSV `trl bars --format ohlcv` imports.
pub fn session_csv(n: usize) -> String {
    let s = session_series(n);
    let mut out = String::from("time,open,high,low,close,session\n");
    for (b, session) in s.bars.iter().zip(&s.sessions) {
        out += &format!("{},{},{},{},{},{}\n", b.time, b.open, b.high, b.low, b.close, session);
    }
    out
}

/// A seeded random walk of one-minute Binance-style bars with noisy flow:
/// no edge by construction, for checking that controls stay quiet.
pub fn random_walk(n: usize, seed: u64) -> Series {
    let mut rng = Rng::new(seed);
    let mut bars = Vec::with_capacity(n);
    let mut price: f64 = 100.0;
    for i in 0..n {
        let open = price;
        let mut high = open;
        let mut low = open;
        for _ in 0..4 {
            price *= math::exp(0.0005 * rng.normal());
            high = high.max(price);
            low = low.min(price);
        }
        let volume = 50.0 + 50.0 * rng.uniform();
        bars.push(Bar {
            time: START_MS + i as i64 * 60_000,
            open,
            high,
            low,
            close: price,
            volume,
            quote: volume * price,
            buy_volume: volume * rng.uniform(),
            trades: 10 + rng.index(40) as i32,
            filled: 0,
        });
    }
    Series::binance("WALK", 60_000, bars)
}
