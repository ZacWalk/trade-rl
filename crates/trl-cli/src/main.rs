//! `trl`: the trade-rl command line. Headless and deterministic: bars live
//! in a tradedb database, runs are written under `runs/`.

use std::error::Error;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Args, Parser, Subcommand};
use trl_binance::{DownloadConfig, Fetch, FixtureFetch, HttpFetch, Kind, Limits, Market, Period};
use trl_core::agents::AGENT_NAMES;
use trl_core::experiment::{ExperimentConfig, Verdict, parse_hidden, run_experiment, verify_run};
use trl_core::import::{import_binance, load_ohlcv_csv};
use trl_core::patterns::{PatternsConfig, run_patterns};
use trl_core::run::{write_list, write_report};
use trl_core::text::hex64;
use trl_core::time::{MS_PER_DAY, duration_text, format_time_ms, parse_duration_ms};
use trl_core::{Series, SeriesInfo, SeriesKind, Store, synthetic};

type Fallible<T = u8> = Result<T, Box<dyn Error>>;

const DEFAULT_DB: &str = "data/bars.tradedb";

#[derive(Parser)]
#[command(
    name = "trl",
    version,
    about = "trade-rl: predict short, flat or long positions in offline, reproducible experiments",
    after_help = "Exit codes: 0 success, 1 a control, the audit or a reproduction failed, 2 bad input or an error."
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Fetch checksum-verified Binance archives. The only network command.
    Download(DownloadArgs),
    /// Import bars: Binance trade or kline CSVs, or a generic OHLCV CSV.
    Bars(BarsArgs),
    /// List, show, export or remove stored series.
    #[command(subcommand)]
    Series(SeriesCommand),
    /// Store a deterministic synthetic series, for demos and smoke tests.
    Synth(SynthArgs),
    /// Walk-forward train and evaluate an agent against baselines, costs and controls.
    Walk(Box<WalkArgs>),
    /// Shift audit: prove every feature at t depends only on bars up to t.
    Audit(SeriesArgs),
    /// Measure whether calendar, previous-session levels or flow proxies carry forward return.
    Patterns(PatternsArgs),
    /// Print a run's manifest and pooled results.
    Report(RunArgs),
    /// List recorded runs.
    List(RootArgs),
    /// Re-run a recorded run and require byte-identical artifacts.
    Verify(VerifyArgs),
}

#[derive(Args)]
struct DbArgs {
    /// The bar database.
    #[arg(long, default_value = DEFAULT_DB)]
    db: PathBuf,
}

#[derive(Args)]
struct SeriesArgs {
    #[command(flatten)]
    db: DbArgs,
    /// Series name, e.g. ETHUSDT/1s; optional when the database holds one.
    #[arg(long)]
    series: Option<String>,
}

#[derive(Args)]
struct DownloadArgs {
    #[arg(long, default_value = "ETHUSDT")]
    symbol: String,
    #[arg(long, default_value = "trades", value_parser = ["trades", "aggTrades", "klines", "fundingRate", "metrics"])]
    kind: String,
    /// First day, YYYY-MM-DD.
    #[arg(long)]
    from: String,
    /// Last day, YYYY-MM-DD, inclusive.
    #[arg(long)]
    to: String,
    /// um (USD-M futures), cm (COIN-M futures) or spot.
    #[arg(long, default_value = "um", value_parser = ["um", "cm", "spot"])]
    market: String,
    /// Kline interval: 1s (spot only), 1m, 3m, 5m, 15m, 30m, 1h ... 1d, 3d, 1w or 1mo.
    #[arg(long, default_value = "1m")]
    source: String,
    /// auto is monthly for klines and fundingRate, daily otherwise.
    #[arg(long, default_value = "auto", value_parser = ["auto", "daily", "monthly"])]
    period: String,
    /// Output directory; default data/raw for trades, data/klines/SYMBOL/INTERVAL, or data/KIND/SYMBOL.
    #[arg(long)]
    out: Option<PathBuf>,
    /// Serve archives from this directory instead of the network.
    #[arg(long)]
    fixtures: Option<PathBuf>,
    /// Replace existing files, keeping each old one as .previous.
    #[arg(long)]
    force: bool,
    /// Skip archives Binance does not publish (HTTP 404); a missing checksum still fails.
    #[arg(long)]
    skip_missing: bool,
    #[arg(long, default_value_t = 256 << 20)]
    max_archive_bytes: u64,
    #[arg(long, default_value_t = 512 << 20)]
    max_csv_bytes: u64,
    /// Data rows allowed per archive.
    #[arg(long, default_value_t = 10_000_000)]
    max_rows: u64,
}

#[derive(Args)]
struct BarsArgs {
    #[command(flatten)]
    db: DbArgs,
    /// binance (trade or kline CSVs) or ohlcv (a generic CSV).
    #[arg(long, default_value = "binance", value_parser = ["binance", "ohlcv"])]
    format: String,
    /// A CSV or a directory of CSVs (binance; default data/raw), or a CSV (ohlcv).
    #[arg(long = "in")]
    input: Option<PathBuf>,
    /// Store under this name; default SYMBOL/INTERVAL.
    #[arg(long)]
    series: Option<String>,
    /// Replace a stored series of the same name.
    #[arg(long)]
    replace: bool,
    /// Bar width: a whole number and ms, s, m, h or d. Binance klines default to their own.
    #[arg(long)]
    interval: Option<String>,
    /// Instrument name (binance default ETHUSDT).
    #[arg(long)]
    symbol: Option<String>,
    /// ohlcv: utc, or the name of the producer's session calendar (then a session column is required).
    #[arg(long)]
    calendar: Option<String>,
    /// ohlcv: the price unit, e.g. USD.
    #[arg(long)]
    currency: Option<String>,
    /// ohlcv: where the data came from.
    #[arg(long)]
    source_name: Option<String>,
    /// ohlcv: raw or adjusted prices.
    #[arg(long, value_parser = ["raw", "adjusted"])]
    price_basis: Option<String>,
}

#[derive(Subcommand)]
enum SeriesCommand {
    /// List stored series.
    List(DbArgs),
    /// Show one series' catalog entry.
    Show(NamedArgs),
    /// Write a series as CSV (OHLCV series in the form `trl bars --format ohlcv` reads).
    Export(ExportArgs),
    /// Delete a series.
    Remove(NamedArgs),
}

#[derive(Args)]
struct NamedArgs {
    #[command(flatten)]
    db: DbArgs,
    /// Series name.
    name: String,
}

#[derive(Args)]
struct ExportArgs {
    #[command(flatten)]
    db: DbArgs,
    name: String,
    /// The CSV to write; it must not exist.
    #[arg(long)]
    out: PathBuf,
}

#[derive(Args)]
struct SynthArgs {
    #[command(flatten)]
    db: DbArgs,
    /// trend (15m Binance-style bars with flow), session (15m OHLC sessions) or walk (1m random walk).
    #[arg(long, default_value = "trend", value_parser = ["trend", "session", "walk"])]
    kind: String,
    #[arg(long, default_value_t = 16_000)]
    bars: usize,
    /// Random-walk seed.
    #[arg(long, default_value_t = 1)]
    seed: u64,
    /// Store under this name; default synthetic/KIND.
    #[arg(long)]
    series: Option<String>,
    #[arg(long)]
    replace: bool,
}

#[derive(Args)]
struct WalkArgs {
    #[command(flatten)]
    source: SeriesArgs,
    /// flat, long, short, random, crossover, breakout, qtable or dqn.
    #[arg(long, default_value = "dqn")]
    agent: String,
    /// Independently seeded learners; the headline is their vote.
    #[arg(long, default_value_t = 3)]
    seeds: usize,
    #[arg(long, default_value_t = 40_000)]
    train: usize,
    #[arg(long, default_value_t = 10_000)]
    test: usize,
    /// Validation bars, inside their own gaps; 0 is off. Choose parameters here, never on test.
    #[arg(long, default_value_t = 0)]
    valid: usize,
    /// Purge between windows.
    #[arg(long, default_value_t = 2000)]
    gap: usize,
    #[arg(long, default_value_t = 3600)]
    episode: usize,
    #[arg(long, default_value_t = 2)]
    epochs: usize,
    /// Round-trip cost in basis points.
    #[arg(long, default_value_t = 5.0)]
    cost_bps: f64,
    /// One-way slippage in basis points, on top of cost.
    #[arg(long, default_value_t = 0.0)]
    slippage_bps: f64,
    /// Scale cost by how thin the bar is (Binance only); 1 is inverse-proportional.
    #[arg(long, default_value_t = 0.0)]
    cost_activity: f64,
    #[arg(long, default_value_t = 0.0)]
    hold_penalty: f64,
    #[arg(long, default_value_t = 3e-4)]
    lr: f64,
    /// Learning-rate factor applied after each epoch; 1 is off.
    #[arg(long, default_value_t = 1.0)]
    lr_decay: f64,
    /// Q edge required per unit of position change; 0 is off.
    #[arg(long, default_value_t = 0.0)]
    switch_margin: f64,
    /// Hidden layer widths.
    #[arg(long, default_value = "64,64")]
    hidden: String,
    #[arg(long, default_value_t = 0.99)]
    gamma: f64,
    #[arg(long, default_value_t = 200_000)]
    epsilon_decay: i64,
    #[arg(long, default_value_t = 32)]
    batch: usize,
    #[arg(long, default_value_t = 2000)]
    target_sync: usize,
    /// Q-table learning rate.
    #[arg(long, default_value_t = 0.1)]
    q_lr: f64,
    #[arg(long, default_value_t = 1)]
    seed: u64,
    /// Runs directory.
    #[arg(long, default_value = "runs")]
    out: PathBuf,
}

#[derive(Args)]
struct PatternsArgs {
    #[command(flatten)]
    source: SeriesArgs,
    /// Level bucket width, in previous-session ranges.
    #[arg(long, default_value_t = 0.25)]
    width: f64,
    /// Permutations for the null; at least 20.
    #[arg(long, default_value_t = 200)]
    shuffles: usize,
    /// Permutation block in bars; 0 is one day (Binance only; OHLCV needs an explicit block).
    #[arg(long, default_value_t = 0)]
    block: usize,
    #[arg(long, default_value_t = 5.0)]
    cost_bps: f64,
    #[arg(long, default_value_t = 1)]
    seed: u64,
}

#[derive(Args)]
struct RootArgs {
    #[arg(long, default_value = "runs")]
    root: PathBuf,
}

#[derive(Args)]
struct RunArgs {
    /// Run id (a directory under the root).
    #[arg(long)]
    run: String,
    #[arg(long, default_value = "runs")]
    root: PathBuf,
}

#[derive(Args)]
struct VerifyArgs {
    #[arg(long)]
    run: String,
    #[arg(long, default_value = "runs")]
    root: PathBuf,
    /// Use this bar database instead of the one the run recorded.
    #[arg(long)]
    db: Option<PathBuf>,
}

fn main() -> ExitCode {
    let cli = match Cli::try_parse() {
        Ok(cli) => cli,
        Err(e) => {
            let _ = e.print();
            return ExitCode::from(e.exit_code().clamp(0, 255) as u8);
        }
    };
    match run(cli.command) {
        Ok(code) => ExitCode::from(code),
        Err(e) => {
            let _ = std::io::stdout().flush();
            eprintln!("{e}");
            ExitCode::from(2)
        }
    }
}

fn run(command: Command) -> Fallible {
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    match command {
        Command::Download(a) => download(a, &mut out),
        Command::Bars(a) => bars(a, &mut out),
        Command::Series(c) => series(c, &mut out),
        Command::Synth(a) => synth(a, &mut out),
        Command::Walk(a) => walk(*a, &mut out),
        Command::Audit(a) => audit(a, &mut out),
        Command::Patterns(a) => patterns(a, &mut out),
        Command::Report(a) => {
            write_report(&a.root, &a.run, &mut out)?;
            Ok(0)
        }
        Command::List(a) => {
            write_list(&a.root, &mut out)?;
            Ok(0)
        }
        Command::Verify(a) => {
            let verdict = verify_run(&a.root, &a.run, a.db.as_deref(), &mut out)?;
            Ok(match verdict {
                Verdict::Reproduced { control_failed, .. } => u8::from(control_failed),
                Verdict::DataChanged | Verdict::Differs { .. } => 1,
            })
        }
    }
}

fn download(a: DownloadArgs, out: &mut dyn Write) -> Fallible {
    let config = DownloadConfig {
        symbol: a.symbol,
        market: Market::parse(&a.market).ok_or("--market must be um, cm or spot")?,
        kind: Kind::parse(&a.kind).ok_or("unsupported --kind")?,
        period: Period::parse(&a.period).ok_or("--period must be auto, daily or monthly")?,
        interval: a.source,
        from: a.from,
        to: a.to,
        out: a.out.unwrap_or_default(),
        force: a.force,
        skip_missing: a.skip_missing,
        limits: Limits {
            max_archive_bytes: a.max_archive_bytes,
            max_csv_bytes: a.max_csv_bytes,
            max_rows: a.max_rows,
            ..Limits::default()
        },
    };
    let client: Box<dyn Fetch> = match &a.fixtures {
        Some(dir) => Box::new(FixtureFetch::new(dir)?),
        None => Box::new(HttpFetch::new()),
    };
    let summary = trl_binance::download_window(&config, client.as_ref(), out)?;
    if summary.downloaded + summary.skipped == 0 {
        return Err("no archives were downloaded".into());
    }
    Ok(0)
}

fn parse_interval(text: &str) -> Fallible<i64> {
    parse_duration_ms(text)
        .filter(|&ms| ms <= 365 * MS_PER_DAY)
        .ok_or_else(|| {
            format!(
                "bad --interval '{text}'; expected a whole number and a unit (ms, s, m, h or d) up to 365d, such as 10s, 15m or 1h"
            )
            .into()
        })
}

fn stored(out: &mut dyn Write, info: &SeriesInfo, db: &Path) -> Fallible<()> {
    writeln!(out, "hash {}", hex64(info.hash))?;
    writeln!(out, "stored {} in {}", info.name, db.display())?;
    Ok(())
}

fn bars(a: BarsArgs, out: &mut dyn Write) -> Fallible {
    let interval = a.interval.as_deref().map(parse_interval).transpose()?;
    match a.format.as_str() {
        "ohlcv" => {
            for (flag, given) in [
                ("--in", a.input.is_some()),
                ("--symbol", a.symbol.is_some()),
                ("--interval", interval.is_some()),
                ("--calendar", a.calendar.is_some()),
                ("--currency", a.currency.is_some()),
            ] {
                if !given {
                    return Err(format!("--format ohlcv requires {flag}").into());
                }
            }
            let input = a.input.unwrap_or_default();
            if !input.is_file() {
                return Err(format!(
                    "--format ohlcv requires a single CSV file; cannot open {}",
                    input.display()
                )
                .into());
            }
            let market = trl_core::Market {
                source: a.source_name.unwrap_or_else(|| "local-csv".into()),
                currency: a.currency.unwrap_or_default(),
                calendar: a.calendar.unwrap_or_default(),
                price_basis: a.price_basis.unwrap_or_else(|| "raw".into()),
                volume_available: false,
            };
            let interval = interval.unwrap_or_default();
            let store = Store::open(&a.db.db)?;
            let series = load_ohlcv_csv(&input, &a.symbol.unwrap_or_default(), interval, &market)?;
            let name = a.series.unwrap_or_else(|| series.default_name());
            let info = store.save(&name, &series, a.replace)?;
            let gaps = series
                .bars
                .windows(2)
                .filter(|w| w[1].time - w[0].time != interval)
                .count();
            writeln!(out, "{} bars x {interval}ms, {gaps} gaps preserved", series.len())?;
            writeln!(
                out,
                "format {}, calendar {}, volume {}",
                series.format_name(),
                series.market.calendar,
                if series.market.volume_available {
                    "available"
                } else {
                    "unavailable"
                }
            )?;
            stored(out, &info, &a.db.db)?;
        }
        _ => {
            for (flag, given) in [
                ("--calendar", a.calendar.is_some()),
                ("--currency", a.currency.is_some()),
                ("--source-name", a.source_name.is_some()),
                ("--price-basis", a.price_basis.is_some()),
            ] {
                if given {
                    return Err(format!("{flag} requires --format ohlcv").into());
                }
            }
            let input = a.input.unwrap_or_else(|| PathBuf::from("data/raw"));
            let store = Store::open(&a.db.db)?;
            let symbol = a.symbol.unwrap_or_else(|| "ETHUSDT".into());
            let import = import_binance(&input, interval, &symbol)?;
            for f in &import.files {
                let file = f.path.file_name().unwrap_or_default().to_string_lossy();
                writeln!(out, "  {file}  {} {}", f.rows, f.kind.noun())?;
            }
            let s = &import.series;
            let name = a.series.unwrap_or_else(|| s.default_name());
            let info = store.save(&name, s, a.replace)?;
            let mut counts = Vec::new();
            if import.trade_rows > 0 || import.kline_rows == 0 {
                counts.push(format!("{} trades", import.trade_rows));
            }
            if import.kline_rows > 0 {
                counts.push(format!("{} klines", import.kline_rows));
            }
            writeln!(
                out,
                "\n{} -> {} bars x {}ms ({} empty)",
                counts.join(" + "),
                s.len(),
                s.interval_ms,
                import.quiet
            )?;
            writeln!(
                out,
                "{} .. {}",
                format_time_ms(s.bars[0].time),
                format_time_ms(s.bars[s.len() - 1].time)
            )?;
            stored(out, &info, &a.db.db)?;
        }
    }
    Ok(0)
}

fn describe(info: &SeriesInfo) -> String {
    format!(
        "{}  {}  {} bars x {}  {} .. {}  hash {}",
        info.name,
        info.kind.as_str(),
        info.bars,
        duration_text(info.interval_ms),
        format_time_ms(info.first),
        format_time_ms(info.last),
        hex64(info.hash)
    )
}

fn series(command: SeriesCommand, out: &mut dyn Write) -> Fallible {
    match command {
        SeriesCommand::List(db) => {
            let store = Store::open_read_only(&db.db)?;
            let all = store.list()?;
            if all.is_empty() {
                writeln!(out, "no series in {}", db.db.display())?;
            }
            for info in all {
                writeln!(out, "{}", describe(&info))?;
            }
        }
        SeriesCommand::Show(a) => {
            let store = Store::open_read_only(&a.db.db)?;
            let info = store.resolve(Some(&a.name))?;
            let m = &info.market;
            for (key, value) in [
                ("name", info.name.clone()),
                ("symbol", info.symbol.clone()),
                ("kind", info.kind.as_str().into()),
                ("interval", duration_text(info.interval_ms)),
                ("bars", info.bars.to_string()),
                ("first", format_time_ms(info.first)),
                ("last", format_time_ms(info.last)),
                ("source", m.source.clone()),
                ("currency", m.currency.clone()),
                ("calendar", m.calendar.clone()),
                ("price_basis", m.price_basis.clone()),
                (
                    "volume",
                    if m.volume_available { "available" } else { "unavailable" }.into(),
                ),
                ("hash", hex64(info.hash)),
            ] {
                writeln!(out, "  {key:<12} {value}")?;
            }
        }
        SeriesCommand::Export(a) => {
            let store = Store::open_read_only(&a.db.db)?;
            let series = store.load(&a.name)?;
            let file = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&a.out)
                .map_err(|e| format!("cannot create {}: {e}", a.out.display()))?;
            let mut w = std::io::BufWriter::new(file);
            export_csv(&series, &mut w)?;
            w.flush()?;
            writeln!(out, "wrote {} bars to {}", series.len(), a.out.display())?;
        }
        SeriesCommand::Remove(a) => {
            let store = Store::open(&a.db.db)?;
            if !store.remove(&a.name)? {
                return Err(format!("no series named {}", a.name).into());
            }
            writeln!(out, "removed {}", a.name)?;
        }
    }
    Ok(0)
}

/// Prices print in shortest round-trip form, so an export re-imports exactly.
fn export_csv(series: &Series, w: &mut dyn Write) -> Fallible<()> {
    let x = trl_core::text::exact;
    if series.kind == SeriesKind::Ohlcv {
        let volume = series.market.volume_available;
        let session = series.market.calendar != "utc";
        writeln!(
            w,
            "time,open,high,low,close{}{}",
            if volume { ",volume" } else { "" },
            if session { ",session" } else { "" }
        )?;
        for (i, b) in series.bars.iter().enumerate() {
            write!(w, "{},{},{},{},{}", b.time, x(b.open), x(b.high), x(b.low), x(b.close))?;
            if volume {
                write!(w, ",{}", x(b.volume))?;
            }
            if session {
                write!(w, ",{}", series.sessions[i])?;
            }
            writeln!(w)?;
        }
    } else {
        writeln!(w, "time,open,high,low,close,volume,quote,buy_volume,trades,filled")?;
        for b in &series.bars {
            writeln!(
                w,
                "{},{},{},{},{},{},{},{},{},{}",
                b.time,
                x(b.open),
                x(b.high),
                x(b.low),
                x(b.close),
                x(b.volume),
                x(b.quote),
                x(b.buy_volume),
                b.trades,
                b.filled
            )?;
        }
    }
    Ok(())
}

fn synth(a: SynthArgs, out: &mut dyn Write) -> Fallible {
    let series = match a.kind.as_str() {
        "session" => synthetic::session_series(a.bars),
        "walk" => synthetic::random_walk(a.bars, a.seed),
        _ => synthetic::trend_series(a.bars),
    };
    let store = Store::open(&a.db.db)?;
    let name = a.series.unwrap_or_else(|| format!("synthetic/{}", a.kind));
    let info = store.save(&name, &series, a.replace)?;
    writeln!(out, "{}", describe(&info))?;
    writeln!(out, "stored {} in {}", info.name, a.db.db.display())?;
    Ok(0)
}

fn load(source: &SeriesArgs) -> Fallible<(Store, String, Series)> {
    let store = Store::open_read_only(&source.db.db)?;
    let info = store.resolve(source.series.as_deref())?;
    let series = store.load(&info.name)?;
    Ok((store, info.name, series))
}

fn walk(a: WalkArgs, out: &mut dyn Write) -> Fallible {
    if !AGENT_NAMES.contains(&a.agent.as_str()) {
        return Err(format!("unknown agent '{}'; choose one of {}", a.agent, AGENT_NAMES.join(", ")).into());
    }
    let hidden = parse_hidden(&a.hidden)?;
    let (_store, name, series) = load(&a.source)?;
    let defaults = ExperimentConfig::default();
    let config = ExperimentConfig {
        store: a.source.db.db.display().to_string(),
        series: name,
        agent: a.agent,
        runs_root: a.out,
        train_bars: a.train,
        valid_bars: a.valid,
        test_bars: a.test,
        gap_bars: a.gap,
        episode_bars: a.episode,
        epochs: a.epochs,
        seeds: a.seeds,
        base_seed: a.seed,
        cost_bps: a.cost_bps,
        slippage_bps: a.slippage_bps,
        cost_activity: a.cost_activity,
        hold_penalty: a.hold_penalty,
        dqn: trl_core::agents::DqnConfig {
            hidden,
            learning_rate: a.lr,
            lr_decay: a.lr_decay,
            discount: a.gamma,
            epsilon_decay_steps: a.epsilon_decay,
            batch_size: a.batch,
            target_sync: a.target_sync,
            switch_margin: a.switch_margin,
            ..defaults.dqn.clone()
        },
        tabular: trl_core::agents::TabularConfig {
            learning_rate: a.q_lr,
            discount: a.gamma,
            epsilon_decay_steps: a.epsilon_decay,
            ..defaults.tabular.clone()
        },
        ..defaults
    };
    let outcome = run_experiment(&config, &series, out)?;
    Ok(u8::from(outcome.control_failed))
}

fn audit(a: SeriesArgs, out: &mut dyn Write) -> Fallible {
    let (_store, _, series) = load(&a)?;
    let audit = trl_core::audit::shift_audit(&series)?;
    audit.write(out)?;
    if !audit.passed() {
        out.flush()?;
        eprintln!("\nSHIFT AUDIT FAILED: a feature reads bars later than its own.");
        return Ok(1);
    }
    Ok(0)
}

fn patterns(a: PatternsArgs, out: &mut dyn Write) -> Fallible {
    let (_store, _, series) = load(&a.source)?;
    let config = PatternsConfig {
        bucket_width: a.width,
        shuffles: a.shuffles,
        block: a.block,
        cost_bps: a.cost_bps,
        seed: a.seed,
    };
    run_patterns(&series, &config, out)?;
    Ok(0)
}
