//! The pattern study: does hour of day, day of week, distance to the
//! previous session's levels, or (for Binance bars) a participant proxy
//! carry forward return? Each bucket is reported over both halves of the
//! sample against a block-permutation null that accounts for testing every
//! bucket, and flagged PAYS only when the implied position also outlasts its
//! own round-trip cost.

use std::io::Write;
use std::path::PathBuf;

use crate::bars::Series;
use crate::error::{Error, Result, invalid};
use crate::features::{DailyLevels, ExpandingStats};
use crate::math;
use crate::rng::Rng;
use crate::text::general;
use crate::time::{MS_PER_DAY, format_time_ms};

/// So 2K + 1 distance buckets, the ends open.
const DISTANCE_HALF_BUCKETS: i64 = 4;

/// Buckets below this are reported but left out of the multiple-testing
/// scan: a handful of samples produces a meaningless t.
const MIN_BUCKET: i64 = 200;

#[derive(Clone, Debug, PartialEq)]
pub struct PatternsConfig {
    /// Level bucket width, in previous-session ranges.
    pub bucket_width: f64,
    pub shuffles: usize,
    /// Permutation block in bars; 0 means one day of bars (Binance only).
    pub block: usize,
    /// Round trip, for the tradeable verdict.
    pub cost_bps: f64,
    pub seed: u64,
}

impl Default for PatternsConfig {
    fn default() -> PatternsConfig {
        PatternsConfig {
            bucket_width: 0.25,
            shuffles: 200,
            block: 0,
            cost_bps: 5.0,
            seed: 1,
        }
    }
}

fn console(source: std::io::Error) -> Error {
    Error::Io {
        path: PathBuf::from("<output>"),
        source,
    }
}

macro_rules! say {
    ($out:expr) => { writeln!($out).map_err(console)? };
    ($out:expr, $($arg:tt)*) => { writeln!($out, $($arg)*).map_err(console)? };
}

#[derive(Clone, Copy, Debug, Default)]
struct Bucket {
    n: i64,
    sum: f64,
    sum_sq: f64,
}

impl Bucket {
    fn add(&mut self, r: f64) {
        self.n += 1;
        self.sum += r;
        self.sum_sq += r * r;
    }

    fn mean(&self) -> f64 {
        if self.n > 0 { self.sum / self.n as f64 } else { 0.0 }
    }

    fn sd(&self) -> f64 {
        if self.n < 2 {
            return 0.0;
        }
        let m = self.mean();
        let variance = (self.sum_sq - self.n as f64 * m * m) / (self.n - 1) as f64;
        if variance > 0.0 { variance.sqrt() } else { 0.0 }
    }

    fn standard_error(&self) -> f64 {
        if self.n > 1 {
            self.sd() / (self.n as f64).sqrt()
        } else {
            0.0
        }
    }

    fn t_stat(&self) -> f64 {
        let se = self.standard_error();
        if se > 0.0 { self.mean() / se } else { 0.0 }
    }
}

/// One way of slicing the samples. `index` is parallel to the returns; -1
/// means the sample is in no bucket.
struct Grouping {
    name: String,
    labels: Vec<String>,
    index: Vec<i32>,
}

impl Grouping {
    fn new(name: &str, labels: Vec<String>) -> Grouping {
        Grouping {
            name: name.into(),
            labels,
            index: Vec::new(),
        }
    }
}

fn accumulate(group: &Grouping, values: &[f64], from: usize, to: usize) -> Vec<Bucket> {
    let mut out = vec![Bucket::default(); group.labels.len()];
    for (&b, &v) in group.index[from..to].iter().zip(&values[from..to]) {
        if b >= 0 {
            out[b as usize].add(v);
        }
    }
    out
}

fn max_abs_t(buckets: &[Bucket]) -> f64 {
    buckets
        .iter()
        .filter(|b| b.n >= MIN_BUCKET)
        .fold(0.0, |worst, b| f64::max(worst, b.t_stat().abs()))
}

/// The 95th percentile of the largest |t| any bucket reaches when whole
/// blocks of returns are permuted with bucket membership held fixed: the
/// threshold that accounts for every bucket being a separate test. Blocks,
/// not bars, because membership persists for hours and volatility clusters,
/// so a bar-level null is far too narrow. `None` when the shuffle never moves
/// the statistic and so cannot test it.
fn null_thresholds(groups: &[&Grouping], values: &[f64], shuffles: usize, block: usize, seed: u64) -> Vec<Option<f64>> {
    let mut rng = Rng::new(seed);
    let n = values.len();
    let width = block.max(1);
    let count = n / width;
    if count < 2 {
        return vec![None; groups.len()];
    }
    let mut order: Vec<usize> = (0..count).collect();
    let mut permuted = vec![0.0; n];
    permuted[count * width..].copy_from_slice(&values[count * width..]);
    let mut maxima = vec![Vec::with_capacity(shuffles); groups.len()];
    // One permutation serves every grouping: each grouping's null is still a
    // valid draw from the same shuffled series.
    for _ in 0..shuffles {
        rng.shuffle(&mut order);
        for (b, &source) in order.iter().enumerate() {
            permuted[b * width..(b + 1) * width].copy_from_slice(&values[source * width..(source + 1) * width]);
        }
        for (g, group) in groups.iter().enumerate() {
            maxima[g].push(max_abs_t(&accumulate(group, &permuted, 0, n)));
        }
    }
    groups
        .iter()
        .zip(maxima)
        .map(|(group, mut m)| {
            let observed = max_abs_t(&accumulate(group, values, 0, n));
            let moved = m.iter().any(|v| (v - observed).abs() > 1e-9 * observed.max(1.0));
            m.sort_by(f64::total_cmp);
            moved.then(|| m[(m.len() as f64 * 0.95) as usize])
        })
        .collect()
}

/// How long the implied position lasts, not how long the bucket lasts: a
/// momentum bucket can persist all day while its position flips every bar,
/// and it is the position that pays the round trip.
fn mean_position_run(group: &Grouping, side: &[i8]) -> Vec<f64> {
    let k = group.labels.len();
    let (mut runs, mut total) = (vec![0i64; k], vec![0i64; k]);
    let (mut previous, mut previous_side) = (-1i32, 0i8);
    for (&b, &s) in group.index.iter().zip(side) {
        if b >= 0 && s != 0 {
            total[b as usize] += 1;
            if b != previous || s != previous_side {
                runs[b as usize] += 1;
            }
        }
        previous = b;
        previous_side = s;
    }
    runs.iter()
        .zip(&total)
        .map(|(&r, &t)| if r > 0 { t as f64 / r as f64 } else { 0.0 })
        .collect()
}

fn report(
    out: &mut dyn Write,
    group: &Grouping,
    values: &[f64],
    side: &[i8],
    threshold: Option<f64>,
    cost_bps: f64,
    statistic: &str,
) -> Result<()> {
    let total = values.len();
    let half = total / 2;
    let full = accumulate(group, values, 0, total);
    let first = accumulate(group, values, 0, half);
    let second = accumulate(group, values, half, total);
    let pooled = full.iter().fold(Bucket::default(), |p, b| Bucket {
        n: p.n + b.n,
        sum: p.sum + b.sum,
        sum_sq: p.sum_sq + b.sum_sq,
    });
    say!(out, "\n{statistic} {}", group.name);
    say!(
        out,
        "  bucket              n    mean       se       t      1st      2nd     hold    need"
    );
    let runs = mean_position_run(group, side);
    for (b, f) in full.iter().enumerate() {
        if f.n == 0 {
            continue;
        }
        let bps = f.mean() * 10_000.0;
        let need = if bps.abs() > 1e-9 { cost_bps / bps.abs() } else { 1e9 };
        let real = threshold.is_some_and(|t| f.t_stat().abs() > t) && f.n >= MIN_BUCKET;
        say!(
            out,
            "  {:<14} {:>7} {:>7.3}  {:>7.3}  {:>6.2}  {:>7.3}  {:>7.3} {:>8.1} {:>7.0}{}",
            group.labels[b],
            f.n,
            bps,
            f.standard_error() * 10_000.0,
            f.t_stat(),
            first[b].mean() * 10_000.0,
            second[b].mean() * 10_000.0,
            runs[b],
            need,
            if real && runs[b] > need { "  PAYS" } else { "" }
        );
    }
    // The permutation null preserves the pooled mean, so it can only find a
    // difference between buckets; whether the whole sample sits away from
    // zero is a separate question with its own line.
    say!(
        out,
        "  {:<14} {:>7} {:>7.3}  {:>7.3}  {:>6.2}",
        "ALL",
        pooled.n,
        pooled.mean() * 10_000.0,
        pooled.standard_error() * 10_000.0,
        pooled.t_stat()
    );
    let observed = max_abs_t(&full);
    match threshold {
        None => say!(
            out,
            "  largest |t| {observed:.2}  ->  UNTESTED: the block shuffle never moved it; whole-day blocks keep\n  \
             every bar's hour and fewer than two blocks keep everything, so pass a --block of part of a day"
        ),
        Some(t) => say!(
            out,
            "  largest |t| {observed:.2} against a shuffled 95% of {t:.2}  ->  {}",
            if observed > t {
                "SURVIVES the permutation test"
            } else {
                "no better than chance"
            }
        ),
    }
    Ok(())
}

fn distance_bucket(distance: f64, width: f64) -> i32 {
    let raw = (distance / width).floor() as i64 + DISTANCE_HALF_BUCKETS;
    raw.clamp(0, 2 * DISTANCE_HALF_BUCKETS) as i32
}

fn distance_labels(width: f64) -> Vec<String> {
    let half = DISTANCE_HALF_BUCKETS;
    (0..=2 * half)
        .map(|b| {
            let low = (b - half) as f64 * width;
            if b == 0 {
                format!("< {:+.2}", low + width)
            } else if b == 2 * half {
                format!(">= {low:+.2}")
            } else {
                format!("{low:+.2}..{:+.2}", low + width)
            }
        })
        .collect()
}

/// Run the study over `series`, printing its report.
pub fn run_patterns(series: &Series, config: &PatternsConfig, out: &mut dyn Write) -> Result<()> {
    if !math::positive(config.bucket_width) || !config.bucket_width.is_finite() {
        return Err(invalid("--width must be positive"));
    }
    // The threshold is the 95th percentile of the shuffled maxima: fewer than
    // 20 draws cannot place it.
    if config.shuffles < 20 {
        return Err(invalid("--shuffles must be at least 20 to place a 95% threshold"));
    }
    if !math::nonnegative(config.cost_bps) {
        return Err(invalid("--cost-bps must be at least 0"));
    }
    let generic = series.generic();
    if generic && config.block == 0 {
        return Err(invalid(
            "OHLCV patterns require an explicit --block in observed bars; a session is not necessarily 24 hours",
        ));
    }
    if series.len() < 5000 {
        return Err(invalid("not enough bars; build a longer range first"));
    }
    let count = series.len();
    say!(
        out,
        "bars      {count} x {}ms  {} .. {}",
        series.interval_ms,
        format_time_ms(series.bars[0].time),
        format_time_ms(series.bars[count - 1].time)
    );

    let width = config.bucket_width;
    let distance = || distance_labels(width);
    let mut hour = Grouping::new("by hour of day (UTC)", (0..24).map(|h| format!("{h:02}:00")).collect());
    let mut weekday = Grouping::new(
        "by day of week",
        [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
        .iter()
        .map(|d| d.to_string())
        .collect(),
    );
    let level = if generic { "session" } else { "day" };
    let mut to_high = Grouping::new(
        &format!("by distance to previous {level} high (previous {level} ranges)"),
        distance(),
    );
    let mut to_low = Grouping::new(
        &format!("by distance to previous {level} low (previous {level} ranges)"),
        distance(),
    );
    let mut to_close = Grouping::new(
        &format!("by distance to previous {level} close (previous {level} ranges)"),
        distance(),
    );
    let mut in_range = Grouping::new(
        &format!("by position in the previous {level} range (0 = low, 1 = high)"),
        distance(),
    );
    let mut surprise = Grouping::new("by activity against this hour's own norm (log ratio)", distance());
    let mut trade_size = Grouping::new("by average trade size against its norm (log ratio)", distance());
    let mut vwap_side = Grouping::new("by close against the bar's own vwap (bar ranges)", distance());
    let mut impact = Grouping::new(
        "by price impact per unit volume against its norm (log ratio)",
        distance(),
    );
    if generic {
        say!(
            out,
            "UNAVAILABLE: UTC seasonality and trade-flow/activity/VWAP/impact studies in the OHLCV profile"
        );
    }

    // Everything is streamed in bar order, so no statistic can read a bar
    // later than the one it is attributed to.
    let mut levels = DailyLevels::default();
    let mut hourly_activity = [ExpandingStats::default(); 24];
    let (mut size_norm, mut impact_norm) = (ExpandingStats::default(), ExpandingStats::default());
    let mut returns = Vec::new();
    // The payoff of holding the previous bar's direction.
    let mut momentum = Vec::new();
    let (mut drift_side, mut momentum_side) = (Vec::new(), Vec::new());
    let log_return = |to: f64, from: f64| {
        if generic {
            math::ln(to) - math::ln(from)
        } else {
            math::ln(to / from)
        }
    };

    for i in 0..count - 1 {
        let b = &series.bars[i];
        levels.push(b.high, b.low, b.close, series.session(i));
        let h = b.time.div_euclid(3_600_000).rem_euclid(24) as usize;
        let norm = hourly_activity[h].mean();
        let seen = hourly_activity[h].count();
        hourly_activity[h].push(f64::from(b.trades));
        if i == 0 || !levels.ready() || b.close <= 0.0 {
            continue;
        }
        // The previous session's own range is the natural scale for its
        // levels; an intraday ATR puts almost every sample in the tails.
        let span = levels.previous_high() - levels.previous_low();
        if span <= 0.0 {
            continue;
        }
        let previous = series.bars[i - 1].close;
        let next = series.bars[i + 1].close;
        if previous <= 0.0 || next <= 0.0 {
            continue;
        }
        let forward = log_return(next, b.close);
        let behind = log_return(b.close, previous);
        returns.push(forward);
        momentum.push(if behind > 0.0 {
            forward
        } else if behind < 0.0 {
            -forward
        } else {
            0.0
        });
        // Drift implies a position held while the bucket holds; momentum one
        // that flips with the previous bar.
        drift_side.push(1i8);
        momentum_side.push(if behind > 0.0 {
            1
        } else if behind < 0.0 {
            -1
        } else {
            0
        });

        hour.index.push(h as i32);
        // Epoch day 0 was a Thursday; Monday is 0.
        weekday
            .index
            .push((b.time.div_euclid(MS_PER_DAY) + 3).rem_euclid(7) as i32);
        to_high
            .index
            .push(distance_bucket((b.close - levels.previous_high()) / span, width));
        to_low
            .index
            .push(distance_bucket((b.close - levels.previous_low()) / span, width));
        to_close
            .index
            .push(distance_bucket((b.close - levels.previous_close()) / span, width));
        in_range
            .index
            .push(distance_bucket((b.close - levels.previous_low()) / span - 0.5, width));

        // The norm is built only from earlier bars, so this is a surprise an
        // agent could have known about at the time.
        surprise.index.push(if seen > 50 && norm > 0.0 {
            distance_bucket(math::ln((f64::from(b.trades) + 1.0) / (norm + 1.0)), width)
        } else {
            -1
        });
        // Participant proxies: average trade size separates institutional
        // from retail flow; close against the bar's vwap says which side was
        // aggressive; impact per unit volume stands in for depth.
        let size = if b.trades > 0 {
            b.volume / f64::from(b.trades)
        } else {
            0.0
        };
        let log_size = if size > 0.0 { math::ln(size) } else { 0.0 };
        trade_size.index.push(if size > 0.0 && size_norm.count() > 500 {
            distance_bucket(log_size - size_norm.mean(), width)
        } else {
            -1
        });
        if size > 0.0 {
            size_norm.push(log_size);
        }
        let bar_range = b.high - b.low;
        let vwap = if b.volume > 0.0 { b.quote / b.volume } else { 0.0 };
        vwap_side.index.push(if bar_range > 0.0 && vwap > 0.0 {
            distance_bucket((b.close - vwap) / bar_range, width)
        } else {
            -1
        });
        let depth = if b.volume > 0.0 {
            math::ln((behind.abs() + 1e-9) / b.volume.sqrt())
        } else {
            0.0
        };
        impact.index.push(if b.volume > 0.0 && impact_norm.count() > 500 {
            distance_bucket(depth - impact_norm.mean(), width)
        } else {
            -1
        });
        if b.volume > 0.0 {
            impact_norm.push(depth);
        }
    }

    if returns.len() < 1000 {
        return Err(invalid("too few usable bars after the first session"));
    }
    say!(
        out,
        "samples   {}  (mean and se are basis points per bar)",
        returns.len()
    );
    say!(
        out,
        "control   {} permutations, bucket assignment held fixed",
        config.shuffles
    );
    say!(out, "\ndrift     mean forward return: does the bucket say which way");
    say!(
        out,
        "momentum  mean of the previous bar's direction held one bar: positive is\n          \
         trend, negative is reversion, and either is tradeable"
    );

    let groups: Vec<&Grouping> = if generic {
        vec![&to_high, &to_low, &to_close, &in_range]
    } else {
        vec![
            &hour,
            &weekday,
            &to_high,
            &to_low,
            &to_close,
            &in_range,
            &surprise,
            &trade_size,
            &vwap_side,
            &impact,
        ]
    };
    let block = if config.block > 0 {
        config.block
    } else {
        (MS_PER_DAY / series.interval_ms.max(1)).max(1) as usize
    };
    say!(out, "block     {block} bars per permutation block");

    let drift_null = null_thresholds(&groups, &returns, config.shuffles, block, config.seed);
    for (g, group) in groups.iter().enumerate() {
        report(
            out,
            group,
            &returns,
            &drift_side,
            drift_null[g],
            config.cost_bps,
            "drift",
        )?;
    }
    let momentum_null = null_thresholds(&groups, &momentum, config.shuffles, block, config.seed.wrapping_add(1));
    for (g, group) in groups.iter().enumerate() {
        report(
            out,
            group,
            &momentum,
            &momentum_side,
            momentum_null[g],
            config.cost_bps,
            "momentum",
        )?;
    }
    say!(
        out,
        "\nhold is how long the implied position lasts, need is how many bars it must be\n\
         held to cover a {} bps round trip. PAYS needs hold > need AND a t\n\
         beyond the permutation threshold. Significance without persistence is not an edge.",
        general(config.cost_bps, 6)
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synthetic;

    #[test]
    fn labels_and_buckets() {
        let labels = distance_labels(0.25);
        assert_eq!(labels.len(), 9);
        assert_eq!(labels[0], "< -0.75");
        assert_eq!(labels[4], "+0.00..+0.25");
        assert_eq!(labels[8], ">= +1.00");
        assert_eq!(distance_bucket(-10.0, 0.25), 0);
        assert_eq!(distance_bucket(0.1, 0.25), 4);
        assert_eq!(distance_bucket(10.0, 0.25), 8);
    }

    #[test]
    fn binance_study_runs_every_grouping() {
        let mut out = Vec::new();
        let config = PatternsConfig {
            shuffles: 20,
            ..PatternsConfig::default()
        };
        run_patterns(&synthetic::random_walk(12_000, 5), &config, &mut out).unwrap();
        let text = String::from_utf8(out).unwrap();
        assert!(text.contains("drift by hour of day"), "{text}");
        assert!(text.contains("momentum by price impact"), "{text}");
        assert!(text.contains("largest |t|"), "{text}");
    }

    #[test]
    fn generic_study_needs_a_block() {
        let s = synthetic::session_series(7000);
        let mut config = PatternsConfig {
            shuffles: 20,
            ..PatternsConfig::default()
        };
        let e = run_patterns(&s, &config, &mut Vec::new()).unwrap_err();
        assert!(e.to_string().contains("--block"), "{e}");
        config.block = 26;
        let mut out = Vec::new();
        run_patterns(&s, &config, &mut out).unwrap();
        let text = String::from_utf8(out).unwrap();
        assert!(
            text.contains("UNAVAILABLE") && text.contains("previous session"),
            "{text}"
        );
        assert!(!text.contains("by hour of day"), "{text}");
    }
}
