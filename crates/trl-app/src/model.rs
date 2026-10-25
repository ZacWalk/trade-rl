//! What the viewer shows: one series from the bar database and, optionally,
//! a run's positions and equity lined up against it by bar time. Read-only:
//! the viewer never trains and never writes.

use std::path::{Path, PathBuf};

use trl_core::run::Manifest;
use trl_core::text::{fixed, general};
use trl_core::time::format_time_ms;
use trl_core::{Series, Store};

/// A bar the run did not score.
pub const NO_POSITION: i8 = -2;
pub const DEFAULT_DB: &str = "data/bars.tradedb";

const CONTROLS: &str = "drag to pan, wheel zoom, shift-wheel scroll, click the map below";

#[derive(Clone, Debug, Default)]
pub struct Options {
    pub db: Option<PathBuf>,
    pub series: Option<String>,
    pub run: Option<PathBuf>,
}

#[derive(Debug, Default)]
pub struct Model {
    pub series: Option<Series>,
    pub name: String,
    pub db: PathBuf,
    /// Per bar; NaN where the run had no position.
    pub equity: Vec<f64>,
    /// Per bar; [`NO_POSITION`] where absent.
    pub position: Vec<i8>,
    pub has_run: bool,
    pub run_id: String,
    /// Why the run could not be shown, drawn in the footer.
    pub run_problem: String,
    pub status: String,
    /// First bar in view.
    pub start: i64,
    /// Bars in view.
    pub span: i64,
}

impl Model {
    /// Load what `options` names. Never fails: problems become the status.
    /// A run's manifest supplies the database and series when not given.
    pub fn load(options: &Options) -> Model {
        let manifest = options
            .run
            .as_ref()
            .and_then(|dir| Manifest::read(&dir.join("manifest.txt")).ok());
        let recorded = |key: &str| {
            manifest
                .as_ref()
                .map(|m| m.get(key).to_string())
                .filter(|v| !v.is_empty())
        };
        let db = options
            .db
            .clone()
            .or_else(|| recorded("store").map(PathBuf::from))
            .unwrap_or_else(|| PathBuf::from(DEFAULT_DB));
        let name = options.series.clone().or_else(|| recorded("series"));
        let mut model = Model {
            db: db.clone(),
            ..Model::default()
        };
        let loaded = Store::open_read_only(&db).and_then(|store| {
            let info = store.resolve(name.as_deref())?;
            let series = store.load(&info.name)?;
            Ok((info.name, series))
        });
        match loaded {
            Ok((name, series)) => {
                model.status = format!("{}  {} bars", series.symbol, series.len());
                model.span = (series.len() as i64).min(20_000);
                model.name = name;
                model.series = Some(series);
            }
            Err(e) => {
                model.status = format!("{e}  -  import bars with: trl bars --db {}", db.display());
                return model;
            }
        }
        if let Some(dir) = &options.run {
            model.load_run(dir);
        }
        model
    }

    pub fn loaded(&self) -> bool {
        self.series.as_ref().is_some_and(|s| !s.is_empty())
    }

    pub fn total(&self) -> i64 {
        self.series.as_ref().map_or(0, |s| s.len() as i64)
    }

    /// Line a run's `equity.csv` up with the bars by open time.
    pub fn load_run(&mut self, dir: &Path) {
        let Some(series) = &self.series else {
            return;
        };
        let Ok(text) = std::fs::read_to_string(dir.join("equity.csv")) else {
            self.status = format!("no equity.csv in {}", dir.display());
            self.run_problem = self.status.clone();
            return;
        };
        let n = series.len();
        self.equity = vec![f64::NAN; n];
        self.position = vec![NO_POSITION; n];
        let mut matched = 0;
        let (mut first, mut last) = (usize::MAX, 0);
        for line in text.lines().skip(1) {
            let cells: Vec<&str> = line.split(',').collect();
            if cells.len() < 6 {
                continue;
            }
            let (Ok(time), Ok(position)) = (cells[0].parse::<i64>(), cells[2].parse::<i8>()) else {
                continue;
            };
            let Some(i) = series.index_at_time(time) else {
                continue;
            };
            let equity = cells[5].parse::<f64>().unwrap_or(f64::NAN);
            self.position[i] = position.clamp(-1, 1);
            self.equity[i] = if equity.is_finite() { equity } else { f64::NAN };
            matched += 1;
            first = first.min(i);
            last = last.max(i);
        }
        self.has_run = matched > 0;
        self.run_id = dir.display().to_string();
        if self.has_run {
            // Open on what the run traded: its out-of-sample bars.
            self.start = first as i64;
            self.span = (last - first + 1) as i64;
        } else {
            self.status = "equity.csv did not line up with these bars".into();
            self.run_problem = format!("equity.csv in {} did not line up with these bars", dir.display());
        }
    }

    /// Keep the view inside the series, at least a pixel's worth of bars wide.
    pub fn clamp_view(&mut self, width: i32) {
        let total = self.total();
        if total == 0 {
            return;
        }
        let minimum = total.min(64.max(i64::from(width) / 8));
        self.span = self.span.clamp(minimum, total);
        self.start = self.start.clamp(0, total - self.span);
    }

    pub fn set_start(&mut self, start: i64, width: i32) {
        self.start = start;
        self.clamp_view(width);
    }

    /// The start that centres the view on minimap column `x`.
    pub fn seek_start(&self, x: f32, width: i32) -> i64 {
        if width <= 0 {
            return self.start;
        }
        let total = self.total();
        let centre = (f64::from(x) / f64::from(width) * total as f64) as i64;
        (centre - self.span / 2).clamp(0, (total - self.span).max(0))
    }

    /// Zoom about `anchor`, a fraction of the width.
    pub fn zoom(&mut self, anchor: f64, zoom_in: bool, width: i32) {
        if !self.loaded() || width <= 0 {
            return;
        }
        let anchor = anchor.clamp(0.0, 1.0);
        let focus = self.start + (anchor * self.span as f64) as i64;
        self.span = (self.span as f64 * if zoom_in { 0.8 } else { 1.25 }) as i64;
        self.clamp_view(width);
        self.start = focus - (anchor * self.span as f64) as i64;
        self.clamp_view(width);
    }

    /// Scroll an eighth of the view.
    pub fn scroll(&mut self, back: bool, width: i32) {
        self.start += if back { -self.span / 8 } else { self.span / 8 };
        self.clamp_view(width);
    }

    pub fn home(&mut self, width: i32) {
        self.start = 0;
        self.span = self.total();
        self.clamp_view(width);
    }

    pub fn end(&mut self, width: i32) {
        self.start = self.total();
        self.clamp_view(width);
    }

    /// The bar under the cursor, or the last one in view.
    pub fn hovered(&self, cursor_x: Option<f32>, width: i32) -> usize {
        let (first, count) = (self.start.max(0) as usize, self.span.max(1) as usize);
        match cursor_x {
            Some(x) if x >= 0.0 && width > 0 => {
                (first + (count as f64 * f64::from(x) / f64::from(width)) as usize).min(first + count - 1)
            }
            _ => first + count - 1,
        }
    }

    pub fn header(&self) -> String {
        let Some(series) = &self.series else {
            return "trade-rl viewer".into();
        };
        format!(
            "{}   bars {}..{} of {}   {}",
            self.name,
            self.start,
            self.start + self.span,
            series.len(),
            self.db.display()
        )
    }

    fn price(&self, value: f64) -> String {
        match &self.series {
            Some(s) if s.generic() => general(value, 8),
            _ => fixed(value, 2),
        }
    }

    /// The hovered bar's time, price, and the run's position and equity.
    pub fn readout(&self, index: usize) -> String {
        let Some(series) = &self.series else {
            return String::new();
        };
        let Some(bar) = series.bars.get(index) else {
            return String::new();
        };
        let mut line = format!("{}   price {}", format_time_ms(bar.time), self.price(bar.close));
        if series.generic() {
            let m = &series.market;
            line += &format!(" {}   {}   {}", m.currency, m.calendar, m.price_basis);
        }
        if self.has_run {
            let p = self.position[index];
            line += "   position ";
            line += match p {
                NO_POSITION => "-",
                1.. => "long",
                0 => "flat",
                _ => "short",
            };
            if !self.equity[index].is_nan() {
                line += &format!("   equity {}%", fixed(self.equity[index] * 100.0, 3));
            }
        }
        line
    }

    /// The footer line, and whether it is a warning.
    pub fn footer(&self) -> (String, bool) {
        if self.has_run {
            (
                format!("run {}   green long, red short, grey flat   {CONTROLS}", self.run_id),
                false,
            )
        } else if !self.run_problem.is_empty() {
            (format!("{}   {CONTROLS}", self.run_problem), true)
        } else {
            (format!("no run loaded - pass --run runs/<id>   {CONTROLS}"), false)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    pub fn model(bars: usize) -> Model {
        let series = trl_core::synthetic::trend_series(bars);
        Model {
            span: (bars as i64).min(20_000),
            name: "ETHUSDT/15m".into(),
            status: "ok".into(),
            series: Some(series),
            ..Model::default()
        }
    }

    #[test]
    fn view_stays_inside_the_series() {
        let mut m = model(50_000);
        m.clamp_view(1400);
        assert_eq!((m.start, m.span), (0, 20_000));
        m.zoom(0.5, true, 1400);
        assert_eq!(m.span, 16_000);
        assert_eq!(m.start, 2000);
        for _ in 0..100 {
            m.zoom(0.0, true, 1400);
        }
        assert_eq!(m.span, 175);
        m.end(1400);
        assert_eq!(m.start + m.span, 50_000);
        m.home(1400);
        assert_eq!((m.start, m.span), (0, 50_000));
        m.scroll(false, 1400);
        assert_eq!(m.start, 0);
        assert_eq!(m.seek_start(700.0, 1400), 0);
        m.span = 10_000;
        assert_eq!(m.seek_start(700.0, 1400), 20_000);
        assert_eq!(m.hovered(None, 1400), 9999);
        assert_eq!(m.hovered(Some(700.0), 1400), 5000);
    }

    #[test]
    fn runs_line_up_by_time() {
        let dir = tempfile::tempdir().unwrap();
        let mut m = model(100);
        let bars = &m.series.as_ref().unwrap().bars;
        let csv = format!(
            "time,price,position,gross,cost,equity\n{},1,1,0,0,0.01\n{},1,-1,0,0,0.02\n12345,1,0,0,0,0.5\n",
            bars[10].time, bars[11].time
        );
        std::fs::write(dir.path().join("equity.csv"), csv).unwrap();
        m.load_run(dir.path());
        assert!(m.has_run);
        assert_eq!((m.position[10], m.position[11], m.position[12]), (1, -1, NO_POSITION));
        // The view opens on the bars the run scored.
        assert_eq!((m.start, m.span), (10, 2));
        assert!(m.readout(11).contains("position short   equity 2.000%"));
        assert!(!m.footer().1);

        let other = tempfile::tempdir().unwrap();
        std::fs::write(
            other.path().join("equity.csv"),
            "time,price,position,gross,cost,equity\n5,1,1,0,0,0\n",
        )
        .unwrap();
        let mut m = model(100);
        m.load_run(other.path());
        assert!(!m.has_run && m.footer().1);
    }

    #[test]
    fn missing_database_is_a_status_not_a_crash() {
        let dir = tempfile::tempdir().unwrap();
        let m = Model::load(&Options {
            db: Some(dir.path().join("absent.tradedb")),
            ..Options::default()
        });
        assert!(!m.loaded());
        assert!(m.status.contains("trl bars"), "{}", m.status);
    }
}
