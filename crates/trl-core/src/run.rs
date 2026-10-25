//! Run artifacts: a run directory holds `manifest.txt`, `metrics.csv`,
//! `equity.csv`, `learning.csv` (learners only) and the saved policy
//! (`weights.bin` or `policy.csv`). A run is assembled in a hidden staging
//! directory and renamed into place, so a listed run is always complete.

use std::fmt::Write as _;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use crate::error::{Error, Result, invalid, io};
use crate::metrics::{METRICS_HEADER, Metrics, TradeLog, metrics_row};
use crate::text::{fixed, general, trim};
use crate::time::format_time_ms;

/// The artifacts `trl verify` compares byte for byte.
pub const COMPARED: [&str; 5] = ["metrics.csv", "equity.csv", "learning.csv", "policy.csv", "weights.bin"];

/// Everything a run writes, held until the run completes.
#[derive(Debug, Default)]
pub struct Artifacts {
    manifest: Vec<(String, String)>,
    metrics: String,
    learning: String,
    equity: String,
    policy: Option<(&'static str, Vec<u8>)>,
}

impl Artifacts {
    pub fn new() -> Artifacts {
        Artifacts {
            metrics: format!("variant,seed,cost_bps,fold,{METRICS_HEADER}\n"),
            ..Artifacts::default()
        }
    }

    pub fn set(&mut self, key: &str, value: impl Into<String>) {
        let value = value.into();
        match self.manifest.iter_mut().find(|(k, _)| k == key) {
            Some(entry) => entry.1 = value,
            None => self.manifest.push((key.into(), value)),
        }
    }

    pub fn set_real(&mut self, key: &str, value: f64) {
        self.set(key, general(value, 10));
    }

    pub fn append_metrics(&mut self, variant: &str, seed: u64, cost_bps: f64, fold: i64, m: &Metrics) {
        let _ = writeln!(
            self.metrics,
            "{variant},{seed},{},{fold},{}",
            fixed(cost_bps, 2),
            metrics_row(m)
        );
    }

    pub fn append_learning(&mut self, episode: u64, mean_reward: f64, epsilon: f64, loss: f64) {
        if self.learning.is_empty() {
            self.learning.push_str("episode,mean_reward,epsilon,loss\n");
        }
        let _ = writeln!(
            self.learning,
            "{episode},{},{},{}",
            fixed(mean_reward, 10),
            fixed(epsilon, 6),
            fixed(loss, 10)
        );
    }

    /// The headline's per-bar log. Prices print exactly for OHLCV series.
    pub fn set_equity(&mut self, log: &TradeLog) {
        let mut out = String::with_capacity(64 * log.len() + 64);
        out.push_str("time,price,position,gross,cost,equity\n");
        for i in 0..log.len() {
            let price = if log.elapsed_ms[i] > 0 {
                general(log.price[i], 17)
            } else {
                fixed(log.price[i], 8)
            };
            let _ = writeln!(
                out,
                "{},{price},{},{},{},{}",
                log.time[i],
                log.position[i],
                fixed(log.gross[i], 10),
                fixed(log.cost[i], 10),
                fixed(log.equity[i], 10)
            );
        }
        self.equity = out;
    }

    pub fn set_policy(&mut self, policy: Option<(&'static str, Vec<u8>)>) {
        self.policy = policy;
    }

    /// Publish the run as `<root>/<id>`. Fails rather than replace a run.
    pub fn publish(&self, root: &Path, id: &str) -> Result<PathBuf> {
        ensure_directory(root)?;
        let directory = root.join(id);
        if directory.exists() {
            return Err(invalid(format!(
                "refusing to overwrite run directory {}",
                directory.display()
            )));
        }
        let stage = root.join(format!(".{id}.part"));
        fs::create_dir(&stage).map_err(io(&stage))?;
        let write = |name: &str, bytes: &[u8]| -> Result<()> {
            let path = stage.join(name);
            fs::write(&path, bytes).map_err(io(&path))
        };
        let result = (|| {
            let mut manifest = String::new();
            for (k, v) in &self.manifest {
                let _ = writeln!(manifest, "{k}={v}");
            }
            write("manifest.txt", manifest.as_bytes())?;
            write("metrics.csv", self.metrics.as_bytes())?;
            write("equity.csv", self.equity.as_bytes())?;
            if !self.learning.is_empty() {
                write("learning.csv", self.learning.as_bytes())?;
            }
            if let Some((name, bytes)) = &self.policy {
                write(name, bytes)?;
            }
            fs::rename(&stage, &directory).map_err(io(&directory))
        })();
        if result.is_err() {
            let _ = fs::remove_dir_all(&stage);
        }
        result.map(|_| directory)
    }
}

/// Create `path` as a directory; an existing file there is an error.
pub fn ensure_directory(path: &Path) -> Result<()> {
    if path.as_os_str().is_empty() {
        return Ok(());
    }
    match fs::create_dir_all(path) {
        Ok(()) => Ok(()),
        Err(_) if path.is_dir() => Ok(()),
        Err(e) => Err(Error::Io {
            path: path.into(),
            source: e,
        }),
    }
}

/// `agent-YYYY-MM-DD-HHMMSS-mNNN-sSEED`: sortable by start time.
pub fn make_run_id(agent: &str, seed: u64) -> String {
    let ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as i64);
    let stamp = format_time_ms(ms).replace(' ', "-").replace(':', "");
    format!("{agent}-{stamp}-m{}-s{seed}", ms % 1000)
}

/// A run's manifest, in file order.
#[derive(Clone, Debug, Default)]
pub struct Manifest(pub Vec<(String, String)>);

impl Manifest {
    pub fn read(path: &Path) -> Result<Manifest> {
        let text = fs::read_to_string(path).map_err(|e| match e.kind() {
            io::ErrorKind::NotFound => invalid(format!("cannot open {}", path.display())),
            _ => Error::Io {
                path: path.into(),
                source: e,
            },
        })?;
        Ok(Manifest(
            text.lines()
                .filter_map(|l| l.split_once('='))
                .map(|(k, v)| (trim(k).to_string(), trim(v).to_string()))
                .collect(),
        ))
    }

    /// The value of `key`, or "" when absent.
    pub fn get(&self, key: &str) -> &str {
        self.0.iter().find(|(k, _)| k == key).map_or("", |(_, v)| v.as_str())
    }
}

/// Run directories under `root` that hold a manifest, in name (start time) order.
pub fn list_runs(root: &Path) -> Vec<String> {
    let mut out: Vec<String> = fs::read_dir(root)
        .into_iter()
        .flatten()
        .flatten()
        .filter(|e| e.path().join("manifest.txt").is_file())
        .filter_map(|e| e.file_name().to_str().map(str::to_string))
        .filter(|name| !name.starts_with('.'))
        .collect();
    out.sort();
    out
}

/// `trl list`.
pub fn write_list(root: &Path, out: &mut dyn Write) -> Result<()> {
    let runs = list_runs(root);
    if runs.is_empty() {
        writeln!(out, "no runs under {}", root.display()).map_err(io(root))?;
        return Ok(());
    }
    for id in runs {
        let m = Manifest::read(&root.join(&id).join("manifest.txt")).unwrap_or_default();
        writeln!(
            out,
            "{id}  agent={}  series={}  folds={}  cost={}bps",
            m.get("agent"),
            m.get("series"),
            m.get("folds"),
            m.get("cost_bps")
        )
        .map_err(io(root))?;
    }
    Ok(())
}

/// `trl report`: the manifest, then the pooled (fold -1) results.
pub fn write_report(root: &Path, id: &str, out: &mut dyn Write) -> Result<()> {
    let directory = root.join(id);
    let manifest = Manifest::read(&directory.join("manifest.txt"))?;
    let w = |e| Error::Io {
        path: directory.clone(),
        source: e,
    };
    writeln!(out, "{id}\n").map_err(w)?;
    let width = manifest.0.iter().map(|(k, _)| k.len()).max().unwrap_or(0);
    for (k, v) in &manifest.0 {
        writeln!(out, "  {k:<width$} {v}").map_err(w)?;
    }
    let metrics_path = directory.join("metrics.csv");
    let metrics = fs::read_to_string(&metrics_path).map_err(|_| {
        invalid(format!(
            "no metrics.csv in {}; the run has no results to report",
            directory.display()
        ))
    })?;
    writeln!(out, "\npooled results (fold -1)").map_err(w)?;
    // variant, seed, cost_bps, net_return, sharpe, max_drawdown, turnover
    const PICKED: [usize; 7] = [0, 1, 2, 6, 11, 13, 14];
    let mut table: Vec<Vec<String>> = vec![
        ["variant", "seed", "cost", "net_return", "sharpe", "maxdd", "turnover"]
            .iter()
            .map(|s| s.to_string())
            .collect(),
    ];
    for line in metrics.lines().skip(1) {
        let cells: Vec<&str> = line.split(',').collect();
        if cells.len() < 15 || cells[3] != "-1" {
            continue;
        }
        table.push(PICKED.iter().map(|&c| cells[c].to_string()).collect());
    }
    let mut widths = vec![0; PICKED.len()];
    for row in &table {
        for (c, cell) in row.iter().enumerate() {
            widths[c] = widths[c].max(cell.len());
        }
    }
    for row in &table {
        let mut line = format!("  {:<w$}", row[0], w = widths[0]);
        for (c, cell) in row.iter().enumerate().skip(1) {
            let _ = write!(line, " {cell:>w$}", w = widths[c]);
        }
        writeln!(out, "{line}").map_err(w)?;
    }
    Ok(())
}

/// Whether two files hold the same bytes.
pub fn same_bytes(a: &Path, b: &Path) -> bool {
    match (fs::read(a), fs::read(b)) {
        (Ok(x), Ok(y)) => x == y,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn publish_is_atomic_and_never_replaces() {
        let dir = tempfile::tempdir().unwrap();
        let mut a = Artifacts::new();
        a.set("agent", "flat");
        a.set("agent", "long");
        a.set_real("cost_bps", 5.0);
        a.append_metrics("long", 0, 5.0, -1, &Metrics::default());
        a.append_learning(1, 0.5, 0.25, 0.125);
        let path = a.publish(dir.path(), "run-1").unwrap();
        let m = Manifest::read(&path.join("manifest.txt")).unwrap();
        assert_eq!(m.get("agent"), "long");
        assert_eq!(m.get("cost_bps"), "5");
        assert!(path.join("learning.csv").is_file());
        assert!(a.publish(dir.path(), "run-1").is_err());
        assert_eq!(list_runs(dir.path()), vec!["run-1".to_string()]);
        let mut report = Vec::new();
        write_report(dir.path(), "run-1", &mut report).unwrap();
        let report = String::from_utf8(report).unwrap();
        assert!(report.contains("pooled results"), "{report}");
        assert!(report.contains("long"), "{report}");
    }

    #[test]
    fn run_ids_sort_by_time() {
        let id = make_run_id("dqn", 7);
        assert!(id.starts_with("dqn-20") && id.ends_with("-s7"), "{id}");
        assert!(!id.contains(':') && !id.contains(' '));
    }
}
