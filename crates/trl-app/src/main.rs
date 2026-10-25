//! `trade-rl`: the viewer. Shows a series from the bar database and, with
//! `--run`, a run's positions and equity. Read-only: it never trains and
//! never writes the database.

mod app;
mod model;
mod render;

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::atomic::Ordering;

use clap::Parser;

#[derive(Parser)]
#[command(
    name = "trade-rl",
    version,
    about = "trade-rl viewer: bars, positions and equity. Read-only: it never trains and never writes."
)]
struct Args {
    /// The bar database (default data/bars.tradedb, or the one a --run recorded).
    #[arg(long)]
    db: Option<PathBuf>,
    /// Series to show; optional when the database holds one or a --run names it.
    #[arg(long)]
    series: Option<String>,
    /// A run directory whose positions and equity to overlay, e.g. runs/<id>.
    #[arg(long)]
    run: Option<PathBuf>,
    /// Render the chart (without text) to this PNG with no window, then exit.
    #[arg(long, value_name = "PNG")]
    render: Option<PathBuf>,
    /// Size for --render, WIDTHxHEIGHT.
    #[arg(long, default_value = "1400x900")]
    size: String,
    /// Open the window, capture it to this PNG, and exit: 0 when it drew a series.
    #[arg(long, value_name = "PNG", conflicts_with = "render")]
    smoke_test: Option<PathBuf>,
}

/// Write RGBA pixels as a PNG; never replaces an existing file.
pub(crate) fn save_png(path: &Path, width: u32, height: u32, rgba: Vec<u8>) -> Result<(), String> {
    use std::io::Write;
    let size = tiny_skia::IntSize::from_wh(width, height).ok_or("invalid image size")?;
    let pixmap = tiny_skia::Pixmap::from_vec(rgba, size).ok_or("invalid image pixels")?;
    let bytes = pixmap.encode_png().map_err(|e| e.to_string())?;
    std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .and_then(|mut f| f.write_all(&bytes))
        .map_err(|e| format!("{}: {e}", path.display()))
}

fn headless(options: &model::Options, path: &Path, size: &str) -> Result<String, String> {
    let (w, h) = size
        .split_once('x')
        .and_then(|(w, h)| Some((w.parse::<u32>().ok()?, h.parse::<u32>().ok()?)))
        .filter(|&(w, h)| (1..=16_384).contains(&w) && (1..=16_384).contains(&h))
        .ok_or_else(|| format!("bad --size '{size}'; expected WIDTHxHEIGHT such as 1400x900"))?;
    let mut model = model::Model::load(options);
    if !model.loaded() {
        return Err(model.status);
    }
    model.clamp_view(w as i32);
    let pixels = render::render(
        &model,
        &mut render::MinimapCache::default(),
        w as i32,
        h as i32,
        None,
        1.0,
    );
    save_png(path, w, h, pixels)?;
    Ok(format!("{w}x{h}  {}  {}", model.status, model.footer().0))
}

fn main() -> ExitCode {
    let args = Args::parse();
    let options = model::Options {
        db: args.db,
        series: args.series,
        run: args.run,
    };
    if let Some(path) = &args.render {
        return match headless(&options, path, &args.size) {
            Ok(summary) => {
                println!("{summary}");
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("{e}");
                ExitCode::from(2)
            }
        };
    }
    let smoke = args.smoke_test.is_some();
    match app::run(options, args.smoke_test) {
        Ok(()) if smoke && !app::SMOKE_PASSED.load(Ordering::Relaxed) => ExitCode::FAILURE,
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("trade-rl failed: {e}");
            ExitCode::FAILURE
        }
    }
}
