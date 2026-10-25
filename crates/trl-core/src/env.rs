//! The trading environment.
//!
//! The decision at bar `t` is made from information up to bar `t`'s close and
//! earns the return from `t` to `t + 1`. That ordering is the whole causality
//! guarantee. A position change fills at the decision bar's close plus
//! slippage; no fill price is ever read from the bar's high or low, because
//! choosing a price inside the range would read the bar to pick the fill.

use crate::bars::Series;
use crate::error::{Result, invalid};
use crate::features::{Ema, FeatureMatrix};
use crate::math::{self, clamp};
use crate::rng::Rng;
use crate::time::MS_PER_DAY;

pub const ACTION_COUNT: usize = 3;

/// Actions are 0 short, 1 flat, 2 long.
pub fn action_to_position(action: usize) -> i32 {
    action as i32 - 1
}

pub fn position_to_action(position: i32) -> usize {
    (position + 1) as usize
}

pub fn action_name(action: usize) -> &'static str {
    match action {
        0 => "short",
        2 => "long",
        _ => "flat",
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct EnvConfig {
    /// Round trip, so one way is half of this.
    pub cost_bps: f64,
    /// One way, on top of cost: a fill is never exactly at the close.
    pub slippage_bps: f64,
    /// Above 0 scales cost by how thin the bar is; 1 is inverse-proportional.
    pub cost_activity: f64,
    pub hold_penalty: f64,
    pub episode_bars: usize,
}

impl Default for EnvConfig {
    fn default() -> EnvConfig {
        EnvConfig {
            cost_bps: 5.0,
            slippage_bps: 0.0,
            cost_activity: 0.0,
            hold_penalty: 0.0,
            episode_bars: 3600,
        }
    }
}

/// Log return of each bar's close to the next.
pub fn bar_returns(series: &Series) -> Vec<f64> {
    series
        .bars
        .windows(2)
        .map(|w| {
            let (a, b) = (w[0].close, w[1].close);
            if a > 0.0 && b > 0.0 {
                if series.generic() {
                    math::ln(b) - math::ln(a)
                } else {
                    math::ln(b / a)
                }
            } else {
                0.0
            }
        })
        .collect()
}

/// Per-bar data every environment over one series shares: computed once per
/// experiment rather than once per environment (a walk builds dozens).
#[derive(Clone, Debug)]
pub struct EnvData {
    /// Log return of each bar's close to the next.
    returns: Vec<f64>,
    /// Trailing range, for scaling unrealised pnl.
    atr: Vec<f64>,
    /// Per bar: 1 unless `cost_activity` is set.
    cost_scale: Vec<f64>,
    cost_activity: f64,
}

impl EnvData {
    pub fn new(series: &Series, cost_activity: f64) -> Result<EnvData> {
        if series.generic() && cost_activity != 0.0 {
            return Err(invalid(
                "--cost-activity requires trade counts, unavailable in the OHLCV profile",
            ));
        }
        if !math::nonnegative(cost_activity) {
            return Err(invalid("--cost-activity must be at least 0"));
        }
        let n = series.len();
        let mut atr = Vec::with_capacity(n);
        let mut running = 0.0;
        for (i, b) in series.bars.iter().enumerate() {
            let range = b.high - b.low;
            running = if i == 0 {
                range
            } else {
                running + (range - running) / 48.0
            };
            atr.push(if running > 1e-9 { running } else { 1e-9 });
        }

        // A flat cost across every hour is wrong: activity swings by hour of
        // day and the thin hours are the expensive ones. The trailing norm
        // only sees the past, and this bar's own trade count is known at its
        // close, which is when the fill happens.
        let mut cost_scale = vec![1.0; n];
        if cost_activity > 0.0 {
            let per_day = MS_PER_DAY / series.interval_ms.max(1);
            let mut norm = Ema::new(per_day as usize);
            for (scale, b) in cost_scale.iter_mut().zip(&series.bars) {
                let here = f64::from(b.trades);
                let rate = norm.push(here);
                let ratio = if here > 0.0 && rate > 0.0 { rate / here } else { 1.0 };
                *scale = clamp(math::pow(ratio, cost_activity), 0.5, 4.0);
            }
        }
        Ok(EnvData {
            returns: bar_returns(series),
            atr,
            cost_scale,
            cost_activity,
        })
    }

    /// Log return of each bar's close to the next.
    pub fn returns(&self) -> &[f64] {
        &self.returns
    }
}

pub struct Env<'a> {
    series: &'a Series,
    features: &'a FeatureMatrix,
    data: &'a EnvData,
    config: EnvConfig,
    override_returns: Option<&'a [f64]>,
    state: Vec<f64>,
    t: usize,
    end: usize,
    position: i32,
    bars_held: u64,
    entry_price: f64,
    cost_per_unit: f64,
    last_cost: f64,
    last_units: f64,
    last_gross: f64,
}

impl<'a> Env<'a> {
    pub fn new(
        series: &'a Series,
        features: &'a FeatureMatrix,
        data: &'a EnvData,
        config: &EnvConfig,
    ) -> Result<Env<'a>> {
        if series.len() < 2 || features.rows() != series.len() || data.returns.len() + 1 != series.len() {
            return Err(invalid(
                "the environment needs at least two bars and a feature row per bar",
            ));
        }
        if config.cost_activity != data.cost_activity {
            return Err(invalid("environment data was prepared for a different --cost-activity"));
        }
        for (flag, value) in [
            ("--cost-bps", config.cost_bps),
            ("--slippage-bps", config.slippage_bps),
            ("--hold-penalty", config.hold_penalty),
        ] {
            // A negative cost is a rebate and a negative penalty a bonus;
            // neither is the cost model every reported return must carry.
            if !math::nonnegative(value) {
                return Err(invalid(format!("{flag} must be at least 0")));
            }
        }
        Ok(Env {
            series,
            features,
            data,
            config: config.clone(),
            override_returns: None,
            state: vec![0.0; features.columns + 3],
            t: 0,
            end: 0,
            position: 0,
            bars_held: 0,
            entry_price: 0.0,
            cost_per_unit: config.cost_bps / 2.0 / 10_000.0 + config.slippage_bps / 10_000.0,
            last_cost: 0.0,
            last_units: 0.0,
            last_gross: 0.0,
        })
    }

    /// Control: substitute a different return stream, leaving costs intact.
    pub fn set_return_override(&mut self, returns: Option<&'a [f64]>) {
        self.override_returns = returns;
    }

    pub fn config(&self) -> &EnvConfig {
        &self.config
    }

    pub fn state(&self) -> &[f64] {
        &self.state
    }

    pub fn state_size(&self) -> usize {
        self.state.len()
    }

    pub fn index(&self) -> usize {
        self.t
    }

    pub fn position(&self) -> i32 {
        self.position
    }

    pub fn last_cost(&self) -> f64 {
        self.last_cost
    }

    /// Position units charged this step, scaled like the cost, liquidation included.
    pub fn last_units(&self) -> f64 {
        self.last_units
    }

    pub fn last_gross(&self) -> f64 {
        self.last_gross
    }

    pub fn cost_per_unit(&self) -> f64 {
        self.cost_per_unit
    }

    pub fn step_return(&self, t: usize) -> f64 {
        match self.override_returns {
            Some(r) => r[t],
            None => self.data.returns[t],
        }
    }

    /// The real (not overridden) return of bar `t`.
    pub fn market_return(&self, t: usize) -> f64 {
        self.data.returns[t]
    }

    pub fn first_bar(&self) -> usize {
        self.features.warmup
    }

    pub fn last_bar(&self) -> usize {
        self.series.len() - 2
    }

    pub fn reset(&mut self, start: usize, end: usize) {
        self.t = start.max(self.features.warmup).min(self.last_bar());
        self.end = end.min(self.last_bar()).max(self.t);
        self.position = 0;
        self.bars_held = 0;
        self.entry_price = self.series.bars[self.t].close;
        self.last_cost = 0.0;
        self.last_units = 0.0;
        self.last_gross = 0.0;
        self.observe();
    }

    fn observe(&mut self) {
        let columns = self.features.columns;
        self.state[..columns].copy_from_slice(self.features.row(self.t));
        let close = self.series.bars[self.t].close;
        let unrealised = if self.position == 0 || self.entry_price <= 0.0 {
            0.0
        } else {
            f64::from(self.position) * (close - self.entry_price) / self.data.atr[self.t]
        };
        self.state[columns] = f64::from(self.position);
        self.state[columns + 1] = clamp(unrealised, -10.0, 10.0);
        self.state[columns + 2] = math::ln_1p(self.bars_held as f64) / 5.0;
    }

    /// Take `action` at the current bar's close. Returns the reward and
    /// whether the episode ended; an episode ends flat, paying the exit.
    pub fn step(&mut self, action: usize) -> (f64, bool) {
        let target = action_to_position(action);
        let previous = self.position;
        let change = f64::from((target - previous).abs());
        let scale = self.data.cost_scale[self.t];
        self.last_cost = self.cost_per_unit * scale * change;
        self.last_units = scale * change;
        if target != previous {
            self.entry_price = self.series.bars[self.t].close;
            self.bars_held = 0;
        } else {
            self.bars_held += 1;
        }
        self.position = target;
        self.last_gross = f64::from(target) * self.step_return(self.t);
        let mut reward = self.last_gross - self.last_cost - self.config.hold_penalty * f64::from(target.abs());

        self.t += 1;
        let done = self.t >= self.end;
        if done {
            // Episodes finish flat, so no unrealised position can be banked.
            let held = f64::from(self.position.abs());
            let exit_cost = self.cost_per_unit * self.data.cost_scale[self.t - 1] * held;
            self.last_cost += exit_cost;
            self.last_units += self.data.cost_scale[self.t - 1] * held;
            if self.series.generic() {
                reward -= exit_cost;
            }
            self.position = 0;
        }
        self.observe();
        (reward, done)
    }
}

/// A Fisher–Yates shuffle of `source`: the timing control. For generic OHLC
/// series, returns only trade places with returns of the same elapsed
/// duration, so a weekend gap never lands inside a session.
pub fn shuffled_returns(source: &[f64], series: &Series, seed: u64) -> Result<Vec<f64>> {
    let mut out = source.to_vec();
    let mut rng = Rng::new(seed);
    if !series.generic() {
        rng.shuffle(&mut out);
        return Ok(out);
    }
    if source.len() + 1 != series.len() {
        return Err(invalid("return series does not match OHLCV bars"));
    }
    let mut groups: std::collections::BTreeMap<i64, Vec<usize>> = Default::default();
    for i in 0..source.len() {
        groups
            .entry(series.bars[i + 1].time - series.bars[i].time)
            .or_default()
            .push(i);
    }
    for indices in groups.values() {
        for i in (1..indices.len()).rev() {
            let j = rng.index(i + 1);
            out.swap(indices[i], indices[j]);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::build_features;
    use crate::synthetic;

    #[test]
    fn costs_and_rewards() {
        let s = synthetic::trend_series(2500);
        let f = build_features(&s).unwrap();
        let data = EnvData::new(&s, 0.0).unwrap();
        let config = EnvConfig {
            cost_bps: 10.0,
            slippage_bps: 1.0,
            ..EnvConfig::default()
        };
        let mut env = Env::new(&s, &f, &data, &config).unwrap();
        assert_eq!(env.state_size(), 23);
        env.reset(0, 5000);
        assert_eq!(env.index(), f.warmup);
        let t = env.index();
        let (reward, done) = env.step(2);
        let per_unit = 10.0 / 2.0 / 1e4 + 1.0 / 1e4;
        assert!(!done);
        assert_eq!(env.last_cost(), per_unit);
        assert_eq!(reward, env.step_return(t) - per_unit);
        env.step(0);
        assert_eq!(env.last_cost(), 2.0 * per_unit);
        assert_eq!(env.position(), -1);
        // Holding the episode to its end liquidates and charges the exit.
        let mut last = (0.0, false);
        while !last.1 {
            last = env.step(0);
        }
        assert_eq!(env.position(), 0);
        assert_eq!(env.index(), s.len() - 2);
    }

    #[test]
    fn generic_rejects_activity_costs() {
        let s = synthetic::session_series(2500);
        assert!(EnvData::new(&s, 1.0).is_err());
        let binance = synthetic::trend_series(2500);
        let f = build_features(&binance).unwrap();
        let data = EnvData::new(&binance, 1.0).unwrap();
        // Data prepared for one activity scaling cannot serve another.
        assert!(Env::new(&binance, &f, &data, &EnvConfig::default()).is_err());
        let negative = EnvConfig {
            cost_bps: -1.0,
            cost_activity: 1.0,
            ..EnvConfig::default()
        };
        assert!(Env::new(&binance, &f, &data, &negative).is_err());
    }

    #[test]
    fn shuffles_respect_durations() {
        let s = synthetic::session_series(600);
        let r = bar_returns(&s);
        let shuffled = shuffled_returns(&r, &s, 3).unwrap();
        let mut a = r.clone();
        let mut b = shuffled.clone();
        a.sort_by(f64::total_cmp);
        b.sort_by(f64::total_cmp);
        assert_eq!(a, b);
        assert_ne!(r, shuffled);
        for (i, value) in shuffled.iter().enumerate() {
            let gap = s.bars[i + 1].time - s.bars[i].time;
            if gap != 900_000 {
                // Only overnight and weekend returns can land here.
                let source = r.iter().position(|x| x == value).unwrap();
                assert_ne!(s.bars[source + 1].time - s.bars[source].time, 900_000);
            }
        }
    }
}
