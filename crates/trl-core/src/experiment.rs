//! Walk-forward experiments and their reproduction.
//!
//! Each fold trains on a window, skips a purge gap (and an optional
//! validation window inside its own gaps), then scores greedily on the next
//! held-out test window; folds advance by the test length. Several seeds train
//! side by side so their vote — the headline — is scored on the same windows.
//! Every result is reported after costs, beside a cost curve, constant and
//! rule baselines, and two controls: shuffled returns (does the alignment
//! with returns matter?) and rotated positions (is the exposure alone enough?).

use std::io::Write;
use std::path::{Path, PathBuf};

use crate::agents::{AGENT_NAMES, Agent, DqnConfig, Ensemble, TabularConfig, make_agent};
use crate::bars::Series;
use crate::env::{Env, EnvConfig, EnvData, action_to_position, shuffled_returns};
use crate::error::{Error, Result, invalid};
use crate::features::{FeatureMatrix, NORMALISATION, build_features};
use crate::math;
use crate::metrics::{Metrics, TradeLog, breakeven_cost_bps, compute_metrics};
use crate::mlp::{DQN_NUMERICS, network_seed};
use crate::run::{Artifacts, COMPARED, Manifest, make_run_id, same_bytes};
use crate::store::Store;
use crate::text::{general, hex64};
use crate::time::{MS_PER_DAY, civil_from_days, format_time_ms};

/// The agents every walk is compared with.
pub const BASELINES: [&str; 5] = ["flat", "long", "random", "crossover", "breakout"];

/// Seeds are `base + i * SEED_STRIDE`.
pub const SEED_STRIDE: u64 = 7919;

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

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Fold {
    pub train_start: usize,
    pub train_end: usize,
    pub valid_start: usize,
    pub valid_end: usize,
    pub test_start: usize,
    pub test_end: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExperimentConfig {
    /// The bar database, as given; recorded so `verify` can reload the series.
    pub store: String,
    /// The series name within the store.
    pub series: String,
    pub agent: String,
    pub runs_root: PathBuf,
    pub train_bars: usize,
    /// 0 disables the validation window.
    pub valid_bars: usize,
    pub test_bars: usize,
    pub gap_bars: usize,
    pub episode_bars: usize,
    pub epochs: usize,
    pub seeds: usize,
    pub base_seed: u64,
    pub cost_bps: f64,
    pub slippage_bps: f64,
    pub cost_activity: f64,
    pub hold_penalty: f64,
    pub cost_curve: Vec<f64>,
    pub dqn: DqnConfig,
    pub tabular: TabularConfig,
}

impl Default for ExperimentConfig {
    fn default() -> ExperimentConfig {
        ExperimentConfig {
            store: String::new(),
            series: String::new(),
            agent: "dqn".into(),
            runs_root: PathBuf::from("runs"),
            train_bars: 40_000,
            valid_bars: 0,
            test_bars: 10_000,
            gap_bars: 2000,
            episode_bars: 3600,
            epochs: 2,
            seeds: 3,
            base_seed: 1,
            cost_bps: 5.0,
            slippage_bps: 0.0,
            cost_activity: 0.0,
            hold_penalty: 0.0,
            cost_curve: vec![1.0, 2.0, 5.0, 10.0],
            dqn: DqnConfig::default(),
            tabular: TabularConfig::default(),
        }
    }
}

impl ExperimentConfig {
    pub fn validate(&self) -> Result<()> {
        if self.train_bars == 0
            || self.test_bars <= 1
            || self.episode_bars <= 1
            || self.epochs == 0
            || self.seeds == 0
            || self.seeds > 1000
        {
            return Err(invalid("invalid walk-forward windows, epochs or seed count"));
        }
        // A negative cost is a rebate and a negative penalty a bonus; neither
        // is the cost model every reported return must carry.
        for (flag, value) in [
            ("--cost-bps", self.cost_bps),
            ("--slippage-bps", self.slippage_bps),
            ("--cost-activity", self.cost_activity),
            ("--hold-penalty", self.hold_penalty),
        ] {
            if !math::nonnegative(value) {
                return Err(invalid(format!("{flag} must be at least 0")));
            }
        }
        if !self.cost_curve.iter().all(|&c| math::nonnegative(c)) {
            return Err(invalid("cost-curve levels must be at least 0"));
        }
        if !AGENT_NAMES.contains(&self.agent.as_str()) {
            return Err(invalid(format!(
                "unknown agent '{}'; choose one of {}",
                self.agent,
                AGENT_NAMES.join(", ")
            )));
        }
        Ok(())
    }

    /// The `i`th seed.
    pub fn seed(&self, i: usize) -> u64 {
        self.base_seed.wrapping_add(i as u64 * SEED_STRIDE)
    }

    fn hidden_text(&self) -> String {
        self.dqn
            .hidden
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(",")
    }

    fn record(&self, a: &mut Artifacts) {
        a.set("store", self.store.clone());
        a.set("series", self.series.clone());
        a.set("base_seed", self.base_seed.to_string());
        a.set("policy_seed", self.seed(0).to_string());
        a.set("seeds", self.seeds.to_string());
        a.set("train_bars", self.train_bars.to_string());
        a.set("valid_bars", self.valid_bars.to_string());
        a.set("test_bars", self.test_bars.to_string());
        a.set("gap_bars", self.gap_bars.to_string());
        a.set("episode_bars", self.episode_bars.to_string());
        a.set("epochs", self.epochs.to_string());
        a.set_real("cost_bps", self.cost_bps);
        a.set_real("slippage_bps", self.slippage_bps);
        a.set_real("cost_activity", self.cost_activity);
        a.set_real("hold_penalty", self.hold_penalty);
        a.set(
            "cost_curve",
            self.cost_curve
                .iter()
                .map(|c| general(*c, 10))
                .collect::<Vec<_>>()
                .join(","),
        );
        a.set("hidden", self.hidden_text());
        a.set_real("lr", self.dqn.learning_rate);
        a.set_real("lr_decay", self.dqn.lr_decay);
        a.set_real("gamma", self.dqn.discount);
        a.set("epsilon_decay", self.dqn.epsilon_decay_steps.to_string());
        a.set("batch", self.dqn.batch_size.to_string());
        a.set("target_sync", self.dqn.target_sync.to_string());
        a.set_real("switch_margin", self.dqn.switch_margin);
        a.set_real("q_lr", self.tabular.learning_rate);
    }

    /// Every setting a recorded run took from its command line. A setting
    /// that does not come back makes the re-run a different experiment
    /// wearing the same name, so a malformed value is an error.
    pub fn from_manifest(m: &Manifest, runs_root: PathBuf) -> Result<ExperimentConfig> {
        let d = ExperimentConfig::default();
        let bad = |key: &str| invalid(format!("manifest value {key}='{}' is malformed", m.get(key)));
        let whole = |key: &str, fallback: usize| -> Result<usize> {
            match m.get(key) {
                "" => Ok(fallback),
                v => v.parse().map_err(|_| bad(key)),
            }
        };
        let real = |key: &str, fallback: f64| -> Result<f64> {
            match m.get(key) {
                "" => Ok(fallback),
                v => crate::text::parse_finite(v).ok_or_else(|| bad(key)),
            }
        };
        let hidden = match m.get("hidden") {
            "" => d.dqn.hidden.clone(),
            v => parse_hidden(v)?,
        };
        let cost_curve = match m.get("cost_curve") {
            "" => d.cost_curve.clone(),
            v => v
                .split(',')
                .map(|c| crate::text::parse_finite(c).ok_or_else(|| bad("cost_curve")))
                .collect::<Result<_>>()?,
        };
        let discount = real("gamma", d.dqn.discount)?;
        let epsilon_decay = match m.get("epsilon_decay") {
            "" => d.dqn.epsilon_decay_steps,
            v => v.parse().map_err(|_| bad("epsilon_decay"))?,
        };
        Ok(ExperimentConfig {
            store: m.get("store").into(),
            series: m.get("series").into(),
            agent: m.get("agent").into(),
            runs_root,
            train_bars: whole("train_bars", d.train_bars)?,
            valid_bars: whole("valid_bars", d.valid_bars)?,
            test_bars: whole("test_bars", d.test_bars)?,
            gap_bars: whole("gap_bars", d.gap_bars)?,
            episode_bars: whole("episode_bars", d.episode_bars)?,
            epochs: whole("epochs", d.epochs)?,
            seeds: whole("seeds", d.seeds)?,
            base_seed: match m.get("base_seed") {
                "" => d.base_seed,
                v => v.parse().map_err(|_| bad("base_seed"))?,
            },
            cost_bps: real("cost_bps", d.cost_bps)?,
            slippage_bps: real("slippage_bps", d.slippage_bps)?,
            cost_activity: real("cost_activity", d.cost_activity)?,
            hold_penalty: real("hold_penalty", d.hold_penalty)?,
            cost_curve,
            dqn: DqnConfig {
                hidden,
                learning_rate: real("lr", d.dqn.learning_rate)?,
                lr_decay: real("lr_decay", d.dqn.lr_decay)?,
                discount,
                epsilon_decay_steps: epsilon_decay,
                batch_size: whole("batch", d.dqn.batch_size)?,
                target_sync: whole("target_sync", d.dqn.target_sync)?,
                switch_margin: real("switch_margin", d.dqn.switch_margin)?,
                ..d.dqn.clone()
            },
            tabular: TabularConfig {
                learning_rate: real("q_lr", d.tabular.learning_rate)?,
                discount,
                epsilon_decay_steps: epsilon_decay,
                ..d.tabular.clone()
            },
        })
    }
}

/// `"64,64"` into layer widths; anything else is an error, so a typo cannot
/// quietly train the default network.
pub fn parse_hidden(text: &str) -> Result<Vec<usize>> {
    let bad = || invalid(format!("bad --hidden '{text}'; expected positive widths such as 64,64"));
    let sizes: Vec<usize> = text
        .split(',')
        .map(|p| {
            crate::text::trim(p)
                .parse::<usize>()
                .ok()
                .filter(|&v| v > 0 && v <= 1 << 20)
        })
        .collect::<Option<_>>()
        .ok_or_else(bad)?;
    if sizes.is_empty() { Err(bad()) } else { Ok(sizes) }
}

/// The research contract a run's numbers depend on, beyond its settings:
/// recorded in every manifest and required unchanged by `verify`.
pub fn contract(series: &Series, features: &FeatureMatrix) -> Vec<(&'static str, String)> {
    let mut c: Vec<(&'static str, String)> = vec![
        ("bar_format", series.format_name().into()),
        ("prediction", "short-flat-long".into()),
        ("feature_profile", features.profile.into()),
        ("feature_schema", features.schema()),
        ("feature_columns", features.columns.to_string()),
        ("warmup_bars", features.warmup.to_string()),
        ("normalisation", NORMALISATION.into()),
        ("decision_time", "bar-close".into()),
        ("horizon", "next-observed-bar".into()),
        (
            "annualisation",
            if series.generic() {
                "scored-elapsed-time"
            } else {
                "continuous-interval"
            }
            .into(),
        ),
    ];
    if series.generic() {
        let m = &series.market;
        c.extend([
            ("source", m.source.clone()),
            ("currency", m.currency.clone()),
            ("calendar", m.calendar.clone()),
            ("price_basis", m.price_basis.clone()),
            ("volume_available", u8::from(m.volume_available).to_string()),
            ("flow_available", "0".into()),
            ("exposure_basis", "observed-bars".into()),
            ("execution", "decision-close-plus-slippage".into()),
            ("carry_costs", "not-modelled".into()),
            ("reward_costs", "entry-switch-and-terminal".into()),
            ("shuffled_control", "equal-elapsed-duration".into()),
            ("rotated_control", "bar-exposure-and-recorded-cost-matched".into()),
        ]);
    }
    c
}

pub fn make_folds(config: &ExperimentConfig, first_bar: usize, last_bar: usize) -> Vec<Fold> {
    let mut folds = Vec::new();
    let mut start = first_bar;
    loop {
        let train_end = start.saturating_add(config.train_bars);
        // Validation sits inside its own purge on both sides, so no feature
        // window is shared with the training or the test data.
        let valid_start = train_end.saturating_add(config.gap_bars);
        let valid_end = valid_start.saturating_add(config.valid_bars);
        let test_start = valid_end.saturating_add(if config.valid_bars > 0 { config.gap_bars } else { 0 });
        let test_end = test_start.saturating_add(config.test_bars);
        if test_end > last_bar {
            break;
        }
        folds.push(Fold {
            train_start: start,
            train_end,
            valid_start,
            valid_end,
            test_start,
            test_end,
        });
        start += config.test_bars;
    }
    folds
}

/// Train where the agent learns: episodes of `episode_bars` across the
/// fold's training window, `epochs` times.
pub fn train_on_fold(
    learner: &mut dyn Agent,
    env: &mut Env,
    fold: &Fold,
    config: &ExperimentConfig,
    mut artifacts: Option<&mut Artifacts>,
    episodes: &mut u64,
) -> Result<()> {
    if !learner.learns() {
        return Ok(());
    }
    let span = fold.train_end - fold.train_start;
    if span <= 1 {
        return Ok(());
    }
    let episode = if config.episode_bars > 0 {
        config.episode_bars
    } else {
        span
    };
    let mut state = Vec::with_capacity(env.state_size());
    for _ in 0..config.epochs {
        let mut offset = 0;
        while offset + 2 < span {
            let start = fold.train_start + offset;
            let stop = (start + episode).min(fold.train_end);
            offset += episode;
            if stop - start < 2 {
                continue;
            }
            env.reset(start, stop);
            learner.begin_episode();
            let (mut total, mut steps) = (0.0, 0u64);
            loop {
                state.clear();
                state.extend_from_slice(env.state());
                let action = learner.act(&state, false)?;
                let (reward, done) = env.step(action);
                learner.observe(&state, action, reward, env.state(), done)?;
                total += reward;
                steps += 1;
                if done {
                    break;
                }
            }
            *episodes += 1;
            if let Some(a) = artifacts.as_deref_mut().filter(|_| steps > 0) {
                a.append_learning(
                    *episodes,
                    total / steps as f64,
                    learner.exploration(),
                    learner.recent_loss(),
                );
            }
        }
        learner.end_epoch()?;
    }
    Ok(())
}

/// Score `learner` greedily over `start..end`, appending to `log`.
pub fn evaluate(
    learner: &mut dyn Agent,
    env: &mut Env,
    series: &Series,
    start: usize,
    end: usize,
    log: &mut TradeLog,
) -> Result<()> {
    if end < start + 2 {
        return Ok(());
    }
    env.reset(start, end);
    learner.begin_episode();
    loop {
        let index = env.index();
        let action = learner.act(env.state(), true)?;
        let (_, done) = env.step(action);
        let elapsed = if series.generic() {
            series.bars[index + 1].time - series.bars[index].time
        } else {
            0
        };
        log.add(
            series.bars[index].time,
            series.bars[index].close,
            action_to_position(action),
            env.last_gross(),
            env.last_cost(),
            elapsed,
            env.step_return(index),
            env.last_units(),
        );
        if done {
            return Ok(());
        }
    }
}

fn append_log(into: &mut TradeLog, from: &TradeLog) {
    for i in 0..from.len() {
        into.add(
            from.time[i],
            from.price[i],
            from.position[i],
            from.gross[i],
            from.cost[i],
            from.elapsed_ms[i],
            from.market_return[i],
            from.units[i],
        );
    }
}

fn slice_log(log: &TradeLog, from: usize, to: usize) -> TradeLog {
    let mut out = TradeLog::default();
    for i in from..to {
        out.add(
            log.time[i],
            log.price[i],
            log.position[i],
            log.gross[i],
            log.cost[i],
            log.elapsed_ms[i],
            log.market_return[i],
            log.units[i],
        );
    }
    out
}

/// The turnover-matched control: the same position sequence rotated in time.
/// It keeps exposure, holding periods and cost and destroys only the
/// alignment with returns. For OHLCV logs the returns counter-rotate instead,
/// keeping position transitions and recorded costs in place.
pub fn rotate_positions(log: &TradeLog, offset: usize) -> TradeLog {
    let mut out = TradeLog::default();
    let n = log.len();
    if n < 3 {
        return out;
    }
    if log.elapsed_ms.first().is_some_and(|&e| e > 0) {
        for i in 0..n {
            let j = (i + n - offset % n) % n;
            let r = log.market_return[j];
            out.add(
                log.time[i],
                log.price[i],
                log.position[i],
                f64::from(log.position[i]) * r,
                log.cost[i],
                log.elapsed_ms[i],
                r,
                log.units[i],
            );
        }
        return out;
    }
    for i in 0..n - 1 {
        let j = (i + offset) % (n - 1);
        let (a, b) = (log.price[i], log.price[i + 1]);
        let r = if a > 0.0 && b > 0.0 { math::ln(b / a) } else { 0.0 };
        out.add(
            log.time[i],
            a,
            log.position[j],
            f64::from(log.position[j]) * r,
            log.cost[j],
            0,
            r,
            log.units[j],
        );
    }
    out
}

/// The median by Sharpe ratio.
pub fn median_by_sharpe(mut values: Vec<Metrics>) -> Metrics {
    if values.is_empty() {
        return Metrics::default();
    }
    values.sort_by(|a, b| a.sharpe.total_cmp(&b.sharpe));
    values[values.len() / 2]
}

/// The median of eight rotations, so one unlucky offset cannot decide it.
fn rotated_control(log: &TradeLog, interval_ms: i64) -> Metrics {
    median_by_sharpe(
        (1..=8)
            .map(|k| compute_metrics(&rotate_positions(log, log.len() * k / 9), interval_ms))
            .collect(),
    )
}

fn bps_cell(bps: Option<f64>) -> String {
    match bps {
        Some(b) => format!("{b:>8.1}"),
        None => format!("{:>8}", "-"),
    }
}

fn print_header(out: &mut dyn Write) -> Result<()> {
    say!(
        out,
        "  {:<22} {:>9} {:>8} {:>9} {:>8} {:>8} {:>8} {:>8}",
        "",
        "net",
        "sharpe",
        "maxdd",
        "turnover",
        "annual",
        "b/e bps",
        "cost%"
    );
    Ok(())
}

fn print_row(out: &mut dyn Write, label: &str, m: &Metrics, slippage_bps: f64) -> Result<()> {
    // A share of a gross that is not positive is undefined, not "no cost".
    let share = if m.gross_return > 0.0 {
        format!("{:>7.0}%", m.cost_fraction * 100.0)
    } else {
        format!("{:>8}", "-")
    };
    say!(
        out,
        "  {:<22} {:>9.4} {:>8.2} {:>9.4} {:>8.3} {:>8.4} {} {}",
        label,
        m.net_return,
        m.sharpe,
        m.max_drawdown,
        m.turnover,
        m.annual_return,
        bps_cell(breakeven_cost_bps(m, slippage_bps)),
        share
    );
    Ok(())
}

/// The headline by calendar year, or by month when it sits in one year. A
/// result carried by one period, or fading across them, is the commonest
/// false positive; rotating inside each period asks whether timing earned it.
fn print_periods(
    out: &mut dyn Write,
    log: &TradeLog,
    interval_ms: i64,
    cost_bps: f64,
    slippage_bps: f64,
) -> Result<()> {
    let (mut years, mut months) = (Vec::with_capacity(log.len()), Vec::with_capacity(log.len()));
    for &t in &log.time {
        let (y, m, _) = civil_from_days(t.div_euclid(MS_PER_DAY));
        years.push(y);
        months.push(y * 12 + i64::from(m) - 1);
    }
    let runs = |keys: &[i64]| {
        if keys.is_empty() {
            0
        } else {
            1 + keys.windows(2).filter(|w| w[0] != w[1]).count()
        }
    };
    let monthly = runs(&years) < 2;
    let key = if monthly { &months } else { &years };
    if runs(key) < 2 {
        return Ok(());
    }
    say!(
        out,
        "\nBy calendar {} (UTC), ensemble at {} bps; rotated is a timing control inside each period",
        if monthly { "month" } else { "year" },
        general(cost_bps, 6)
    );
    say!(
        out,
        "  {:<10} {:>8} {:>9} {:>9} {:>9} {:>8}",
        "period",
        "bars",
        "market",
        "net",
        "rotated",
        "b/e bps"
    );
    let mut from = 0;
    while from < log.len() {
        let mut to = from;
        while to < log.len() && key[to] == key[from] {
            to += 1;
        }
        let part = slice_log(log, from, to);
        let m = compute_metrics(&part, interval_ms);
        let label = if monthly {
            format!("{:04}-{:02}", key[from].div_euclid(12), key[from].rem_euclid(12) + 1)
        } else {
            format!("{:04}", key[from])
        };
        say!(
            out,
            "  {:<10} {:>8} {:>9.4} {:>9.4} {:>9.4} {}",
            label,
            m.bars,
            m.market_return,
            m.net_return,
            rotated_control(&part, interval_ms).net_return,
            bps_cell(breakeven_cost_bps(&m, slippage_bps))
        );
        from = to;
    }
    Ok(())
}

/// One seed's learner, environments and logs.
struct SeedState<'a> {
    seed: u64,
    learner: Box<dyn Agent>,
    env: Env<'a>,
    control_env: Env<'a>,
    alt_envs: Vec<Env<'a>>,
    alt_logs: Vec<TradeLog>,
    log: TradeLog,
    valid_log: TradeLog,
    control_log: TradeLog,
    episodes: u64,
}

/// What a walk concluded.
#[derive(Clone, Debug)]
pub struct Outcome {
    pub directory: PathBuf,
    pub ensemble: Metrics,
    pub control: Metrics,
    pub rotated: Metrics,
    /// The shuffled-return ensemble beat the real one with a Sharpe above
    /// 0.5: whatever the agent found is not in the alignment with returns.
    pub control_failed: bool,
}

/// Run a walk-forward experiment over `series` and publish its run directory.
pub fn run_experiment(config: &ExperimentConfig, series: &Series, out: &mut dyn Write) -> Result<Outcome> {
    config.validate()?;
    if config.agent == "dqn" {
        say!(out, "DQN numerics {DQN_NUMERICS}");
    }
    if series.len() < if series.generic() { 2 } else { 5000 } {
        return Err(invalid("not enough bars; build a longer range first"));
    }
    say!(
        out,
        "bars      {} x {}ms  {} .. {}",
        series.len(),
        series.interval_ms,
        format_time_ms(series.bars[0].time),
        format_time_ms(series.bars[series.len() - 1].time)
    );
    if series.generic() {
        let m = &series.market;
        say!(
            out,
            "market    {} {}, {}, {}, {}",
            series.symbol,
            m.currency,
            m.calendar,
            m.price_basis,
            m.source
        );
        say!(
            out,
            "limits    no funding, borrow, dividends, contract rolls or short-sale constraints modelled"
        );
    }

    let features = build_features(series)?;
    let data = EnvData::new(series, config.cost_activity)?;
    let base_env = EnvConfig {
        cost_bps: config.cost_bps,
        slippage_bps: config.slippage_bps,
        cost_activity: config.cost_activity,
        hold_penalty: config.hold_penalty,
        episode_bars: config.episode_bars,
    };
    let make_env = |cost_bps: f64| {
        Env::new(
            series,
            &features,
            &data,
            &EnvConfig {
                cost_bps,
                ..base_env.clone()
            },
        )
    };
    let probe = make_env(config.cost_bps)?;
    let folds = make_folds(config, probe.first_bar(), probe.last_bar());
    if folds.is_empty() {
        return Err(invalid("no walk-forward folds fit; reduce --train or --test"));
    }
    say!(
        out,
        "features  {} base + 3 position, warmup {} bars",
        features.columns,
        features.warmup
    );
    let valid = if config.valid_bars > 0 {
        format!("  valid {}", config.valid_bars)
    } else {
        String::new()
    };
    say!(
        out,
        "folds     {}  train {}  gap {}{valid}  test {}\n",
        folds.len(),
        config.train_bars,
        config.gap_bars,
        config.test_bars
    );

    let alt_costs: Vec<f64> = config
        .cost_curve
        .iter()
        .copied()
        .filter(|&c| c != config.cost_bps)
        .collect();
    let shuffled: Vec<Vec<f64>> = (0..config.seeds)
        .map(|i| shuffled_returns(data.returns(), series, config.seed(i)))
        .collect::<Result<_>>()?;
    let ensemble_shuffled = shuffled_returns(data.returns(), series, config.base_seed)?;

    // The learner once per seed, with its controls. Seeds train side by side
    // so each fold's ensemble is scored on the windows the seeds saw.
    let mut seeds = Vec::with_capacity(config.seeds);
    for (i, returns) in shuffled.iter().enumerate() {
        let seed = config.seed(i);
        let mut control_env = make_env(config.cost_bps)?;
        control_env.set_return_override(Some(returns.as_slice()));
        seeds.push(SeedState {
            seed,
            learner: make_agent(&config.agent, &features, &config.dqn, &config.tabular, seed)?,
            env: make_env(config.cost_bps)?,
            control_env,
            alt_envs: alt_costs.iter().map(|&c| make_env(c)).collect::<Result<_>>()?,
            alt_logs: vec![TradeLog::default(); alt_costs.len()],
            log: TradeLog::default(),
            valid_log: TradeLog::default(),
            control_log: TradeLog::default(),
            episodes: 0,
        });
    }

    let mut artifacts = Artifacts::new();
    let mut ensemble_env = make_env(config.cost_bps)?;
    let mut ensemble_valid_env = make_env(config.cost_bps)?;
    let mut ensemble_control_env = make_env(config.cost_bps)?;
    ensemble_control_env.set_return_override(Some(ensemble_shuffled.as_slice()));
    let mut ensemble_alt_envs: Vec<Env> = alt_costs.iter().map(|&c| make_env(c)).collect::<Result<_>>()?;
    let mut ensemble_alt_logs = vec![TradeLog::default(); alt_costs.len()];
    let (mut ensemble_log, mut ensemble_valid_log, mut ensemble_control_log) =
        (TradeLog::default(), TradeLog::default(), TradeLog::default());
    let interval = series.interval_ms;

    for (i, f) in folds.iter().enumerate() {
        for (s, st) in seeds.iter_mut().enumerate() {
            let writer = if s == 0 { Some(&mut artifacts) } else { None };
            train_on_fold(st.learner.as_mut(), &mut st.env, f, config, writer, &mut st.episodes)?;
            // Every variant is scored inside the fold loop, so the cost curve
            // and the control see the policy at the headline's stage of learning.
            let mut fold_log = TradeLog::default();
            evaluate(
                st.learner.as_mut(),
                &mut st.env,
                series,
                f.test_start,
                f.test_end,
                &mut fold_log,
            )?;
            artifacts.append_metrics(
                &config.agent,
                st.seed,
                config.cost_bps,
                i as i64,
                &compute_metrics(&fold_log, interval),
            );
            append_log(&mut st.log, &fold_log);
            // Parameters are chosen here; the test window is scored once and
            // never selected on.
            if config.valid_bars > 0 {
                let (from, to) = (f.valid_start, f.valid_end);
                evaluate(st.learner.as_mut(), &mut st.env, series, from, to, &mut st.valid_log)?;
            }
            for (env, log) in st.alt_envs.iter_mut().zip(&mut st.alt_logs) {
                evaluate(st.learner.as_mut(), env, series, f.test_start, f.test_end, log)?;
            }
            let (from, to) = (f.test_start, f.test_end);
            evaluate(
                st.learner.as_mut(),
                &mut st.control_env,
                series,
                from,
                to,
                &mut st.control_log,
            )?;
        }

        let mut ensemble = Ensemble::new(seeds.iter_mut().map(|s| s.learner.as_mut() as &mut dyn Agent).collect());
        let mut fold_log = TradeLog::default();
        evaluate(
            &mut ensemble,
            &mut ensemble_env,
            series,
            f.test_start,
            f.test_end,
            &mut fold_log,
        )?;
        artifacts.append_metrics(
            "ensemble",
            0,
            config.cost_bps,
            i as i64,
            &compute_metrics(&fold_log, interval),
        );
        append_log(&mut ensemble_log, &fold_log);
        if config.valid_bars > 0 {
            let (from, to) = (f.valid_start, f.valid_end);
            evaluate(
                &mut ensemble,
                &mut ensemble_valid_env,
                series,
                from,
                to,
                &mut ensemble_valid_log,
            )?;
        }
        for (env, log) in ensemble_alt_envs.iter_mut().zip(&mut ensemble_alt_logs) {
            evaluate(&mut ensemble, env, series, f.test_start, f.test_end, log)?;
        }
        let (from, to) = (f.test_start, f.test_end);
        evaluate(
            &mut ensemble,
            &mut ensemble_control_env,
            series,
            from,
            to,
            &mut ensemble_control_log,
        )?;
    }

    let mut learner_results = Vec::new();
    let mut control_results = Vec::new();
    let mut valid_results = Vec::new();
    let (mut net_low, mut net_high) = (0.0f64, 0.0f64);
    for (s, st) in seeds.iter().enumerate() {
        let m = compute_metrics(&st.log, interval);
        artifacts.append_metrics(&config.agent, st.seed, config.cost_bps, -1, &m);
        say!(
            out,
            "seed {}  net {}  sharpe {}  trades {}",
            st.seed,
            general(m.net_return, 6),
            general(m.sharpe, 6),
            m.trades
        );
        net_low = if s == 0 {
            m.net_return
        } else {
            net_low.min(m.net_return)
        };
        net_high = if s == 0 {
            m.net_return
        } else {
            net_high.max(m.net_return)
        };
        learner_results.push(m);
        for (&cost, log) in alt_costs.iter().zip(&st.alt_logs) {
            artifacts.append_metrics(&config.agent, st.seed, cost, -1, &compute_metrics(log, interval));
        }
        let cm = compute_metrics(&st.control_log, interval);
        artifacts.append_metrics("control:shuffled", st.seed, config.cost_bps, -1, &cm);
        control_results.push(cm);
        if config.valid_bars > 0 {
            let vm = compute_metrics(&st.valid_log, interval);
            artifacts.append_metrics(&format!("{}:valid", config.agent), st.seed, config.cost_bps, -1, &vm);
            valid_results.push(vm);
        }
    }

    let ensemble_metrics = compute_metrics(&ensemble_log, interval);
    let ensemble_control = compute_metrics(&ensemble_control_log, interval);
    artifacts.append_metrics("ensemble", 0, config.cost_bps, -1, &ensemble_metrics);
    artifacts.append_metrics("control:ensemble-shuffled", 0, config.cost_bps, -1, &ensemble_control);
    let ensemble_rotated = rotated_control(&ensemble_log, interval);
    artifacts.append_metrics("control:ensemble-rotated", 0, config.cost_bps, -1, &ensemble_rotated);
    for (&cost, log) in alt_costs.iter().zip(&ensemble_alt_logs) {
        artifacts.append_metrics("ensemble", 0, cost, -1, &compute_metrics(log, interval));
    }

    // The saved policy is the first seed's, fixed before any test window was
    // scored; picking the best seed here would be selection on the test set.
    artifacts.set_policy(seeds[0].learner.policy());

    // Baselines do not learn, so one pass over the test windows is enough.
    let mut baseline_results = Vec::with_capacity(BASELINES.len());
    for name in BASELINES {
        let mut baseline = make_agent(name, &features, &config.dqn, &config.tabular, config.base_seed)?;
        let mut env = make_env(config.cost_bps)?;
        let mut log = TradeLog::default();
        for f in &folds {
            evaluate(baseline.as_mut(), &mut env, series, f.test_start, f.test_end, &mut log)?;
        }
        let m = compute_metrics(&log, interval);
        artifacts.append_metrics(name, 0, config.cost_bps, -1, &m);
        baseline_results.push((name, m));
    }
    artifacts.set_equity(&ensemble_log);

    artifacts.set("agent", config.agent.clone());
    artifacts.set("engine", format!("trade-rl {}", env!("CARGO_PKG_VERSION")));
    artifacts.set("experiment_scalar", "binary64");
    for (key, value) in contract(series, &features) {
        artifacts.set(key, value);
    }
    if config.agent == "dqn" {
        artifacts.set("dqn_numerics", DQN_NUMERICS);
        artifacts.set("dqn_model_format", "TRLMLP01-binary32");
        artifacts.set("dqn_initialization", "He-uniform-hidden-Glorot-uniform-linear");
        artifacts.set("dqn_seed_mapping", "mix64-xor-halves-zero-to-one-v1");
        artifacts.set("dqn_network_seed", network_seed(config.seed(0)).to_string());
        artifacts.set("dqn_optimizer", "Adam-beta1=0.9-beta2=0.999-epsilon=1e-8-clip=10");
    }
    artifacts.set("headline", "ensemble");
    artifacts.set("symbol", series.symbol.clone());
    artifacts.set("interval_ms", series.interval_ms.to_string());
    artifacts.set("bar_count", series.len().to_string());
    artifacts.set("data_hash", hex64(series.hash()));
    artifacts.set("folds", folds.len().to_string());
    config.record(&mut artifacts);
    let directory = artifacts.publish(&config.runs_root, &make_run_id(&config.agent, config.base_seed))?;

    say!(
        out,
        "\nWalk-forward, out of sample, {} bps round trip",
        general(config.cost_bps, 6)
    );
    print_header(out)?;
    let slip = config.slippage_bps;
    for (name, m) in &baseline_results {
        print_row(out, name, m, slip)?;
    }
    print_row(
        out,
        &format!("{} (median)", config.agent),
        &median_by_sharpe(learner_results),
        slip,
    )?;
    print_row(out, &format!("{} (ensemble)", config.agent), &ensemble_metrics, slip)?;
    print_row(out, "control:shuffled", &median_by_sharpe(control_results), slip)?;
    print_row(out, "control:ensemble", &ensemble_control, slip)?;
    print_row(out, "control:rotated", &ensemble_rotated, slip)?;
    print_periods(out, &ensemble_log, interval, config.cost_bps, slip)?;
    if config.valid_bars > 0 {
        // Select on this table. The one above is scored once and reported,
        // never searched over, or the number it prints means nothing.
        say!(out, "\nValidation window, for choosing parameters");
        print_header(out)?;
        print_row(
            out,
            &format!("{} (median)", config.agent),
            &median_by_sharpe(valid_results),
            slip,
        )?;
        print_row(
            out,
            &format!("{} (ensemble)", config.agent),
            &compute_metrics(&ensemble_valid_log, interval),
            slip,
        )?;
    }
    // The seed band is the noise floor: a difference inside it is not a result.
    say!(
        out,
        "\nseed band  net {} .. {}  width {}",
        general(net_low, 6),
        general(net_high, 6),
        general(net_high - net_low, 6)
    );
    say!(out, "run        {}", directory.display());

    let control_failed = ensemble_control.sharpe > ensemble_metrics.sharpe && ensemble_control.sharpe > 0.5;
    if control_failed {
        say!(
            out,
            "\nFAILED CONTROL: the shuffled-return run scores higher than the real one."
        );
    }
    Ok(Outcome {
        directory,
        ensemble: ensemble_metrics,
        control: ensemble_control,
        rotated: ensemble_rotated,
        control_failed,
    })
}

/// The result of `trl verify`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Verdict {
    /// Every recorded artifact came back byte-identical.
    Reproduced { compared: usize, control_failed: bool },
    /// The bars changed since the run.
    DataChanged,
    /// Some artifacts differ.
    Differs { differing: usize, compared: usize },
}

/// Re-run a recorded run under `<root>/verify` and require byte-identical
/// metrics, equity, learning and policy artifacts. `store` overrides the
/// database path the run recorded.
pub fn verify_run(root: &Path, id: &str, store: Option<&Path>, out: &mut dyn Write) -> Result<Verdict> {
    let recorded = root.join(id);
    let manifest = Manifest::read(&recorded.join("manifest.txt"))?;
    let store_path = match store {
        Some(p) => p.to_path_buf(),
        None if !manifest.get("store").is_empty() => PathBuf::from(manifest.get("store")),
        None => return Err(invalid("the manifest records no bar database; pass --db")),
    };
    let db = Store::open_read_only(&store_path)?;
    let series = db.load(manifest.get("series"))?;
    let actual = hex64(series.hash());
    say!(out, "data_hash recorded {}", manifest.get("data_hash"));
    say!(out, "data_hash actual   {actual}");
    if manifest.get("data_hash") != actual {
        say!(out, "\nDATA CHANGED since the run; the result cannot be reproduced.");
        return Ok(Verdict::DataChanged);
    }
    let features = build_features(&series)?;
    for (key, value) in contract(&series, &features) {
        if manifest.get(key) != value {
            return Err(invalid(format!(
                "contract mismatch: {key} recorded '{}', now '{value}'; use the recorded compatible revision",
                manifest.get(key)
            )));
        }
    }
    let mut config = ExperimentConfig::from_manifest(&manifest, root.join("verify"))?;
    if store.is_some() {
        config.store = store_path.display().to_string();
    }
    if config.agent == "dqn" && manifest.get("dqn_numerics") != DQN_NUMERICS {
        return Err(invalid(format!(
            "DQN numerics mismatch: the run used '{}', this build is '{DQN_NUMERICS}'; no implicit migration",
            manifest.get("dqn_numerics")
        )));
    }
    if !recorded.join("metrics.csv").is_file() {
        return Err(invalid(format!(
            "no metrics.csv in {}; there is no recorded result to reproduce",
            recorded.display()
        )));
    }
    say!(out, "\nre-running {} ...", config.agent);
    let outcome = run_experiment(&config, &series, out)?;
    say!(out, "\ncompared with {}", recorded.display());
    let (mut compared, mut differing) = (0, 0);
    for name in COMPARED {
        if !recorded.join(name).exists() {
            continue;
        }
        let same = same_bytes(&recorded.join(name), &outcome.directory.join(name));
        compared += 1;
        if !same {
            differing += 1;
        }
        say!(out, "  {name}  {}", if same { "identical" } else { "DIFFERS" });
    }
    if differing > 0 {
        say!(
            out,
            "\nNOT REPRODUCED: {differing} of {compared} recorded artifacts differ; use the recorded compatible revision"
        );
        return Ok(Verdict::Differs { differing, compared });
    }
    say!(out, "reproduced: {compared} recorded artifacts byte-identical");
    Ok(Verdict::Reproduced {
        compared,
        control_failed: outcome.control_failed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synthetic;

    fn small(agent: &str, root: &Path) -> ExperimentConfig {
        ExperimentConfig {
            store: "unused".into(),
            series: "TEST/15m".into(),
            agent: agent.into(),
            runs_root: root.to_path_buf(),
            train_bars: 1500,
            valid_bars: 200,
            test_bars: 600,
            gap_bars: 300,
            episode_bars: 500,
            epochs: 1,
            seeds: 2,
            base_seed: 7,
            dqn: DqnConfig {
                hidden: vec![8, 8],
                batch_size: 8,
                target_sync: 50,
                replay_warmup: 200,
                replay_capacity: 5000,
                ..DqnConfig::default()
            },
            ..ExperimentConfig::default()
        }
    }

    #[test]
    fn folds_advance_by_the_test_window() {
        let c = ExperimentConfig {
            train_bars: 100,
            valid_bars: 10,
            gap_bars: 5,
            test_bars: 50,
            ..ExperimentConfig::default()
        };
        let folds = make_folds(&c, 20, 400);
        assert_eq!(folds[0].test_start, 20 + 100 + 5 + 10 + 5);
        assert_eq!(folds[1].train_start, 70);
        assert!(folds.iter().all(|f| f.test_end <= 400));
        // Test windows end at 190, 240, 290, 340 and 390.
        assert_eq!(folds.len(), 5);
    }

    #[test]
    fn rotation_keeps_exposure_and_costs() {
        let mut log = TradeLog::default();
        for i in 0..20 {
            log.add(
                i * 1000,
                100.0 + i as f64,
                (i % 3) as i32 - 1,
                0.01,
                0.001,
                900,
                0.002 * i as f64,
                1.0,
            );
        }
        let r = rotate_positions(&log, 7);
        let (a, b) = (compute_metrics(&log, 1000), compute_metrics(&r, 1000));
        assert_eq!(
            (a.bars, a.cost_paid, a.turnover, a.time_long),
            (b.bars, b.cost_paid, b.turnover, b.time_long)
        );
        assert!((a.market_return - b.market_return).abs() < 1e-12);
    }

    #[test]
    fn generic_walks_reproduce_and_controls_match() {
        let dir = tempfile::tempdir().unwrap();
        let series = synthetic::session_series(7000);
        for agent in ["breakout", "qtable", "dqn"] {
            let config = small(agent, &dir.path().join(agent));
            let mut first = Vec::new();
            let a = run_experiment(&config, &series, &mut first).unwrap();
            let b = run_experiment(&config, &series, &mut Vec::new()).unwrap();
            for name in COMPARED {
                let (x, y) = (a.directory.join(name), b.directory.join(name));
                assert_eq!(x.exists(), y.exists(), "{agent} {name}");
                if x.exists() {
                    assert!(same_bytes(&x, &y), "{agent} {name} does not reproduce");
                }
            }
            let text = String::from_utf8(first).unwrap();
            assert!(text.contains("By calendar"), "{text}");
            // The rotation changes timing only.
            assert_eq!(a.ensemble.bars, a.rotated.bars);
            assert_eq!(a.ensemble.cost_paid, a.rotated.cost_paid);
            assert_eq!(a.ensemble.turnover, a.rotated.turnover);
            let manifest = Manifest::read(&a.directory.join("manifest.txt")).unwrap();
            assert_eq!(manifest.get("feature_profile"), "ohlc-v1");
            assert_eq!(manifest.get("calendar"), "fixture-sessions");
            let metrics = std::fs::read_to_string(a.directory.join("metrics.csv")).unwrap();
            let curve: std::collections::BTreeSet<&str> = metrics
                .lines()
                .filter(|l| l.starts_with("ensemble,") && l.split(',').nth(3) == Some("-1"))
                .map(|l| l.split(',').nth(2).unwrap())
                .collect();
            assert_eq!(curve.len(), 4, "{agent} cost curve");
            let reloaded = ExperimentConfig::from_manifest(&manifest, config.runs_root.clone()).unwrap();
            assert_eq!(reloaded.dqn.hidden, config.dqn.hidden);
            assert_eq!(reloaded.valid_bars, config.valid_bars);
        }
        assert!(dir.path().join("dqn").read_dir().unwrap().count() >= 2);
    }

    #[test]
    fn invalid_settings_write_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let series = synthetic::session_series(7000);
        let mut config = small("flat", dir.path());
        config.cost_activity = 1.0;
        let e = run_experiment(&config, &series, &mut Vec::new()).unwrap_err();
        assert!(e.to_string().contains("trade count"), "{e}");
        config.cost_activity = 0.0;
        config.agent = "agnet".into();
        assert!(run_experiment(&config, &series, &mut Vec::new()).is_err());
        assert!(!dir.path().exists() || dir.path().read_dir().unwrap().next().is_none());
        assert!(parse_hidden("64,0").is_err());
        assert!(parse_hidden("64,x").is_err());
        assert_eq!(parse_hidden(" 16, 8 ").unwrap(), vec![16, 8]);
    }
}
