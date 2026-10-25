//! Causal features.
//!
//! Every indicator here is a streaming object: it takes one value at a time
//! and never holds the series, so it cannot read forwards. Standardisation
//! uses expanding (trailing) statistics for the same reason; `trl audit`
//! proves it by recomputing features over truncated series.

use std::collections::VecDeque;

use crate::bars::Series;
use crate::error::{Result, invalid};
use crate::math::{self, clamp, finite};
use crate::time::MS_PER_DAY;

const RETURN_LAGS: [usize; 5] = [1, 5, 15, 60, 300];
const EMA_FAST: usize = 12;
const EMA_MID: usize = 48;
const EMA_SLOW: usize = 192;
const VOL_SHORT: usize = 60;
const VOL_LONG: usize = 1440;
const RANGE_WINDOW: usize = 300;
const FLOW_WINDOW: usize = 60;

/// Bars before the slowest indicator is meaningful.
pub const WARMUP_BARS: usize = VOL_LONG + RANGE_WINDOW;
const CLAMP: f64 = 5.0;
/// Previous-day distances, in day ranges.
const LEVEL_CLAMP: f64 = 2.0;
const TAU: f64 = std::f64::consts::TAU;

/// The name of the normalisation contract, recorded in manifests.
pub const NORMALISATION: &str = "causal-expanding-clamped-v1";

/// Columns of the price-only profile used for generic OHLC series.
pub const OHLC_NAMES: [&str; 11] = [
    "ret_1",
    "ret_5",
    "ret_15",
    "ret_60",
    "ret_300",
    "trend_fast",
    "trend_slow",
    "vol_ratio",
    "range_pos",
    "pd_range_pos",
    "pd_high_dist",
];

/// Columns of the profile for Binance bars, which know flow and run 24/7.
pub const BINANCE_NAMES: [&str; 20] = [
    "ret_1",
    "ret_5",
    "ret_15",
    "ret_60",
    "ret_300",
    "trend_fast",
    "trend_slow",
    "vol_ratio",
    "range_pos",
    "flow",
    "activity",
    "tod_sin",
    "tod_cos",
    "pd_range_pos",
    "pd_high_dist",
    "activity_surprise",
    "tod_sin2",
    "tod_cos2",
    "dow_sin",
    "dow_cos",
];

fn safe_divide(a: f64, b: f64) -> f64 {
    if b > 1e-12 { a / b } else { 0.0 }
}

/// Exponential moving average seeded with its first value.
#[derive(Clone, Debug)]
pub struct Ema {
    alpha: f64,
    value: f64,
    started: bool,
}

impl Ema {
    pub fn new(span: usize) -> Ema {
        Ema {
            alpha: 2.0 / (span as f64 + 1.0),
            value: 0.0,
            started: false,
        }
    }

    pub fn push(&mut self, x: f64) -> f64 {
        if self.started {
            self.value += self.alpha * (x - self.value);
        } else {
            self.value = x;
            self.started = true;
        }
        self.value
    }

    pub fn value(&self) -> f64 {
        self.value
    }
}

/// Sum of the last `window` values.
#[derive(Clone, Debug)]
pub struct RollingSum {
    buffer: Vec<f64>,
    index: usize,
    count: usize,
    sum: f64,
}

impl RollingSum {
    pub fn new(window: usize) -> RollingSum {
        RollingSum {
            buffer: vec![0.0; window.max(1)],
            index: 0,
            count: 0,
            sum: 0.0,
        }
    }

    pub fn push(&mut self, x: f64) -> f64 {
        self.sum += x - self.buffer[self.index];
        self.buffer[self.index] = x;
        self.index = (self.index + 1) % self.buffer.len();
        self.count = (self.count + 1).min(self.buffer.len());
        self.sum
    }

    pub fn sum(&self) -> f64 {
        self.sum
    }

    pub fn mean(&self) -> f64 {
        if self.count > 0 {
            self.sum / self.count as f64
        } else {
            0.0
        }
    }

    pub fn full(&self) -> bool {
        self.count >= self.buffer.len()
    }
}

/// Rolling minimum and maximum over monotonic deques: O(1) amortised.
#[derive(Clone, Debug)]
pub struct RollingExtremes {
    window: i64,
    step: i64,
    min: VecDeque<(i64, f64)>,
    max: VecDeque<(i64, f64)>,
}

impl RollingExtremes {
    pub fn new(window: usize) -> RollingExtremes {
        RollingExtremes {
            window: window.max(1) as i64,
            step: 0,
            min: VecDeque::new(),
            max: VecDeque::new(),
        }
    }

    pub fn push(&mut self, x: f64) {
        while self.min.back().is_some_and(|b| b.1 >= x) {
            self.min.pop_back();
        }
        while self.max.back().is_some_and(|b| b.1 <= x) {
            self.max.pop_back();
        }
        self.min.push_back((self.step, x));
        self.max.push_back((self.step, x));
        let cutoff = self.step - self.window + 1;
        while self.min.front().is_some_and(|f| f.0 < cutoff) {
            self.min.pop_front();
        }
        while self.max.front().is_some_and(|f| f.0 < cutoff) {
            self.max.pop_front();
        }
        self.step += 1;
    }

    pub fn min(&self) -> f64 {
        self.min.front().map_or(0.0, |f| f.1)
    }

    pub fn max(&self) -> f64 {
        self.max.front().map_or(0.0, |f| f.1)
    }
}

/// The last `depth + 1` values; slots never written read as 0.
#[derive(Clone, Debug)]
pub struct History {
    buffer: Vec<f64>,
    index: usize,
    count: usize,
}

impl History {
    pub fn new(depth: usize) -> History {
        History {
            buffer: vec![0.0; depth.max(1) + 1],
            index: 0,
            count: 0,
        }
    }

    pub fn push(&mut self, x: f64) {
        self.index = (self.index + 1) % self.buffer.len();
        self.buffer[self.index] = x;
        self.count = (self.count + 1).min(self.buffer.len());
    }

    /// `ago(0)` is the value just pushed.
    pub fn ago(&self, n: usize) -> f64 {
        let size = self.buffer.len();
        let want = n.min(size - 1);
        self.buffer[(self.index + size - want) % size]
    }

    pub fn has(&self, n: usize) -> bool {
        self.count > n
    }
}

/// Tracks the session. The previous session's levels only become readable
/// once it has closed, so only a rollover can publish them.
#[derive(Clone, Debug)]
pub struct DailyLevels {
    day: i64,
    started: bool,
    has_previous: bool,
    high: f64,
    low: f64,
    close: f64,
    previous_high: f64,
    previous_low: f64,
    previous_close: f64,
}

impl Default for DailyLevels {
    fn default() -> DailyLevels {
        DailyLevels {
            day: i64::MIN,
            started: false,
            has_previous: false,
            high: 0.0,
            low: 0.0,
            close: 0.0,
            previous_high: 0.0,
            previous_low: 0.0,
            previous_close: 0.0,
        }
    }
}

impl DailyLevels {
    pub fn push(&mut self, high: f64, low: f64, close: f64, session: i64) {
        if session != self.day {
            if self.started {
                self.previous_high = self.high;
                self.previous_low = self.low;
                self.previous_close = self.close;
                self.has_previous = true;
            }
            self.day = session;
            self.high = high;
            self.low = low;
            self.started = true;
        } else {
            self.high = self.high.max(high);
            self.low = self.low.min(low);
        }
        self.close = close;
    }

    pub fn ready(&self) -> bool {
        self.has_previous
    }

    pub fn previous_high(&self) -> f64 {
        self.previous_high
    }

    pub fn previous_low(&self) -> f64 {
        self.previous_low
    }

    pub fn previous_close(&self) -> f64 {
        self.previous_close
    }

    /// Includes the bar just pushed, which is known at that bar's close.
    pub fn day_high(&self) -> f64 {
        self.high
    }

    pub fn day_low(&self) -> f64 {
        self.low
    }
}

/// Trailing mean and variance over everything seen so far (Welford).
#[derive(Clone, Copy, Debug, Default)]
pub struct ExpandingStats {
    count: i64,
    mean: f64,
    m2: f64,
}

impl ExpandingStats {
    pub fn push(&mut self, x: f64) {
        self.count += 1;
        let delta = x - self.mean;
        self.mean += delta / self.count as f64;
        self.m2 += delta * (x - self.mean);
    }

    pub fn mean(&self) -> f64 {
        self.mean
    }

    pub fn sd(&self) -> f64 {
        if self.count > 1 {
            (self.m2 / (self.count - 1) as f64).sqrt()
        } else {
            0.0
        }
    }

    pub fn count(&self) -> i64 {
        self.count
    }
}

/// Features of every bar, row-major.
#[derive(Clone, Debug)]
pub struct FeatureMatrix {
    pub columns: usize,
    /// Rows before this index are not usable.
    pub warmup: usize,
    pub profile: &'static str,
    pub names: &'static [&'static str],
    pub data: Vec<f64>,
}

impl FeatureMatrix {
    pub fn row(&self, i: usize) -> &[f64] {
        &self.data[i * self.columns..(i + 1) * self.columns]
    }

    pub fn rows(&self) -> usize {
        self.data.len().checked_div(self.columns).unwrap_or(0)
    }

    pub fn column(&self, name: &str) -> Result<usize> {
        self.names
            .iter()
            .position(|n| *n == name)
            .ok_or_else(|| invalid(format!("feature profile {} lacks {name}", self.profile)))
    }

    /// The column names, comma separated: the feature contract of a run.
    pub fn schema(&self) -> String {
        self.names.join(",")
    }
}

/// Streams bars into base observations, before the environment appends the
/// agent's own position.
#[derive(Clone, Debug)]
pub struct FeatureBuilder {
    price_only: bool,
    warmup: usize,
    seen: usize,
    closes: History,
    ema_fast: Ema,
    ema_mid: Ema,
    ema_slow: Ema,
    sq_return_short: RollingSum,
    sq_return_long: RollingSum,
    true_range: RollingSum,
    volume: RollingSum,
    buy_volume: RollingSum,
    trade_count: RollingSum,
    trade_rate: Ema,
    range: RollingExtremes,
    levels: DailyLevels,
    hourly_activity: [ExpandingStats; 24],
    stats: Vec<ExpandingStats>,
    previous_close: f64,
}

impl FeatureBuilder {
    pub fn new(interval_ms: i64, price_only: bool) -> FeatureBuilder {
        let interval_ms = if interval_ms > 0 { interval_ms } else { 1000 };
        let mut warmup = WARMUP_BARS;
        // The previous day's levels need a rollover before they exist, which
        // at a fine interval takes longer than the indicator warmup.
        let per_day = (MS_PER_DAY / interval_ms) as usize;
        if !price_only && per_day + 1 > warmup {
            warmup = per_day + 1;
        }
        FeatureBuilder {
            price_only,
            warmup,
            seen: 0,
            closes: History::new(RETURN_LAGS[RETURN_LAGS.len() - 1] + 2),
            ema_fast: Ema::new(EMA_FAST),
            ema_mid: Ema::new(EMA_MID),
            ema_slow: Ema::new(EMA_SLOW),
            sq_return_short: RollingSum::new(VOL_SHORT),
            sq_return_long: RollingSum::new(VOL_LONG),
            true_range: RollingSum::new(EMA_MID),
            volume: RollingSum::new(FLOW_WINDOW),
            buy_volume: RollingSum::new(FLOW_WINDOW),
            trade_count: RollingSum::new(FLOW_WINDOW),
            trade_rate: Ema::new(VOL_LONG),
            range: RollingExtremes::new(RANGE_WINDOW),
            levels: DailyLevels::default(),
            hourly_activity: [ExpandingStats::default(); 24],
            stats: vec![ExpandingStats::default(); Self::names(price_only).len()],
            previous_close: 0.0,
        }
    }

    pub fn names(price_only: bool) -> &'static [&'static str] {
        if price_only { &OHLC_NAMES } else { &BINANCE_NAMES }
    }

    pub fn warmup(&self) -> usize {
        self.warmup
    }

    /// Bounded or already-periodic columns are left out of the z-scoring.
    fn standardised(&self, column: usize) -> bool {
        column < 8 || (!self.price_only && matches!(column, 9 | 10 | 15))
    }

    fn log_return(&self, close: f64, past: f64) -> f64 {
        if past <= 0.0 {
            0.0
        } else if self.price_only {
            math::ln(close) - math::ln(past)
        } else {
            math::ln(close / past)
        }
    }

    /// Compute the observation of `bar` (known at its close) into `out`.
    pub fn push(&mut self, bar: &crate::Bar, session: i64, out: &mut [f64]) -> Result<()> {
        out.fill(0.0);
        let close = bar.close;
        let previous = if self.seen == 0 { close } else { self.previous_close };
        let r1 = self.log_return(close, previous);

        self.closes.push(close);
        self.range.push(close);
        let fast = self.ema_fast.push(close);
        let mid = self.ema_mid.push(close);
        let slow = self.ema_slow.push(close);

        let short_vol = (self.sq_return_short.push(r1 * r1) / VOL_SHORT as f64).sqrt();
        let long_vol = (self.sq_return_long.push(r1 * r1) / VOL_LONG as f64).sqrt();
        let range = safe_divide(bar.high - bar.low, close);
        let atr = safe_divide(self.true_range.push(range), EMA_MID as f64) * close;

        // Each return is divided by its horizon's volatility, so a quiet hour
        // and a violent one give comparable numbers.
        for (i, &lag) in RETURN_LAGS.iter().enumerate() {
            let raw = self.log_return(close, self.closes.ago(lag));
            out[i] = safe_divide(raw, short_vol * (lag as f64).sqrt());
        }
        out[5] = safe_divide(fast - mid, atr);
        out[6] = safe_divide(mid - slow, atr);
        out[7] = if short_vol > 0.0 && long_vol > 0.0 {
            math::ln(short_vol / long_vol)
        } else {
            0.0
        };
        let (low, high) = (self.range.min(), self.range.max());
        out[8] = if high > low {
            (close - low) / (high - low) - 0.5
        } else {
            0.0
        };

        self.levels.push(bar.high, bar.low, bar.close, session);
        if self.price_only && !self.levels.ready() {
            self.warmup = self.warmup.max(self.seen + 1);
        }
        let span = if self.levels.ready() {
            self.levels.previous_high() - self.levels.previous_low()
        } else {
            0.0
        };
        if span > 0.0 {
            let (range_at, high_at) = if self.price_only { (9, 10) } else { (13, 14) };
            out[range_at] = clamp(
                (close - self.levels.previous_low()) / span - 0.5,
                -LEVEL_CLAMP,
                LEVEL_CLAMP,
            );
            out[high_at] = clamp((close - self.levels.previous_high()) / span, -LEVEL_CLAMP, LEVEL_CLAMP);
        }

        if !self.price_only {
            let volume = self.volume.push(bar.volume);
            let buys = self.buy_volume.push(bar.buy_volume);
            let trades = self.trade_count.push(f64::from(bar.trades));
            let rate = self.trade_rate.push(f64::from(bar.trades));
            out[9] = safe_divide(2.0 * buys - volume, volume);
            out[10] = math::ln((trades + 1.0) / (rate * FLOW_WINDOW as f64 + 1.0));
            let day_fraction = bar.time.rem_euclid(MS_PER_DAY) as f64 / MS_PER_DAY as f64;
            out[11] = math::sin(day_fraction * TAU);
            out[12] = math::cos(day_fraction * TAU);

            // Remove this hour's own causal activity norm, not a whole-file average.
            let hour = (bar.time.div_euclid(3_600_000)).rem_euclid(24) as usize;
            let norm = self.hourly_activity[hour].mean();
            if self.hourly_activity[hour].count() > 24 && norm > 0.0 {
                out[15] = math::ln((f64::from(bar.trades) + 1.0) / (norm + 1.0));
            }
            self.hourly_activity[hour].push(f64::from(bar.trades));

            out[16] = math::sin(day_fraction * 2.0 * TAU);
            out[17] = math::cos(day_fraction * 2.0 * TAU);
            // Monday is 0.
            let weekday = (bar.time.div_euclid(MS_PER_DAY) + 3).rem_euclid(7) as f64 / 7.0;
            out[18] = math::sin(weekday * TAU);
            out[19] = math::cos(weekday * TAU);
        }

        for (i, value) in out.iter_mut().enumerate() {
            if !finite(*value) {
                if self.price_only {
                    return Err(invalid(format!("nonfinite OHLC feature {}", OHLC_NAMES[i])));
                }
                *value = 0.0;
            }
            if !self.standardised(i) {
                continue;
            }
            let stats = &mut self.stats[i];
            stats.push(*value);
            let sd = stats.sd();
            *value = if sd > 1e-12 {
                clamp((*value - stats.mean()) / sd, -CLAMP, CLAMP)
            } else {
                0.0
            };
        }

        self.previous_close = close;
        self.seen += 1;
        Ok(())
    }
}

/// Features of every bar of `series`: the `ohlc-v1` profile for generic
/// OHLC series, `binance-v1` for Binance bars.
pub fn build_features(series: &Series) -> Result<FeatureMatrix> {
    let price_only = series.generic();
    let names = FeatureBuilder::names(price_only);
    let columns = names.len();
    let mut data = vec![0.0; series.len() * columns];
    let mut builder = FeatureBuilder::new(series.interval_ms, price_only);
    for (i, row) in data.chunks_exact_mut(columns).enumerate() {
        builder.push(&series.bars[i], series.session(i), row)?;
    }
    Ok(FeatureMatrix {
        columns,
        warmup: builder.warmup(),
        profile: if price_only { "ohlc-v1" } else { "binance-v1" },
        names,
        data,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synthetic;

    #[test]
    fn rolling_helpers() {
        let mut s = RollingSum::new(3);
        for x in [1.0, 2.0, 3.0, 4.0] {
            s.push(x);
        }
        assert_eq!(s.sum(), 9.0);
        assert!(s.full());
        let mut e = RollingExtremes::new(3);
        for x in [5.0, 1.0, 3.0, 4.0, 2.0] {
            e.push(x);
        }
        assert_eq!((e.min(), e.max()), (2.0, 4.0));
        let mut h = History::new(3);
        for x in [1.0, 2.0, 3.0] {
            h.push(x);
        }
        assert_eq!((h.ago(0), h.ago(2), h.ago(3)), (3.0, 1.0, 0.0));
        assert!(h.has(2) && !h.has(3));
    }

    #[test]
    fn profiles_and_warmup() {
        let binance = build_features(&synthetic::trend_series(3000)).unwrap();
        assert_eq!((binance.profile, binance.columns), ("binance-v1", 20));
        assert_eq!(binance.warmup, WARMUP_BARS);
        assert!(binance.data.iter().all(|v| v.is_finite() && v.abs() <= 5.0));

        let generic = build_features(&synthetic::session_series(3000)).unwrap();
        assert_eq!((generic.profile, generic.columns), ("ohlc-v1", 11));
        assert!(generic.warmup >= WARMUP_BARS);
        assert_eq!(generic.column("pd_high_dist").unwrap(), 10);
        assert!(generic.column("flow").is_err());
    }

    #[test]
    fn one_second_bars_wait_for_a_whole_day() {
        let b = FeatureBuilder::new(1000, false);
        assert_eq!(b.warmup(), 86_401);
    }
}
