//! Checksum-verified acquisition of Binance public archives from
//! <https://data.binance.vision>.
//!
//! The only network code in trade-rl. Every archive is fetched with its
//! mandatory `.CHECKSUM` sidecar; the SHA-256 of the compressed bytes is
//! verified before the ZIP is opened; the ZIP must hold exactly one regular,
//! unencrypted member with the expected name, and is checked against its
//! declared sizes and CRC; typed CSVs are validated row by row; and a file is
//! published under its final name only once complete, never replacing data
//! unless asked (`--force` keeps the old file as `.previous`). No redirects,
//! cookies, retries or checksum bypass.

mod csv;
mod names;
mod publish;
mod transport;
mod zip;

use std::io::Write;
use std::path::{Path, PathBuf};

pub use csv::count_rows;
pub use names::{ArchiveSpec, Kind, Market, Names, Period};
pub use publish::publish;
pub use transport::{Fetch, FixtureFetch, HttpFetch, Response};
pub use zip::extract_single;

use sha2::{Digest, Sha256};

/// Where an acquisition failed. Every variant is exit code 2 at the CLI.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0}")]
    Invalid(String),
    /// The archive does not exist (HTTP 404). Only this can be skipped.
    #[error("not found: {0}")]
    NotFound(String),
    #[error("checksum: {0}")]
    Checksum(String),
    #[error("HTTP {status} from {url}")]
    Http { status: u16, url: String },
    #[error("transport: {0}")]
    Transport(String),
    #[error("parse: {0}")]
    Parse(String),
    #[error("limit: {0}")]
    Limit(String),
    #[error("{}: {source}", path.display())]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

pub(crate) fn invalid(message: impl Into<String>) -> Error {
    Error::Invalid(message.into())
}

pub(crate) fn io(path: &Path) -> impl FnOnce(std::io::Error) -> Error + '_ {
    move |source| Error::Io {
        path: path.to_path_buf(),
        source,
    }
}

/// Size limits, each checked during transfer or decompression.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Limits {
    pub max_archive_bytes: u64,
    /// Both declared and actual decompressed size.
    pub max_csv_bytes: u64,
    /// Data rows per typed CSV.
    pub max_rows: u64,
    /// Archives per window, checked before any request.
    pub max_archives: usize,
}

impl Default for Limits {
    fn default() -> Limits {
        Limits {
            max_archive_bytes: 256 << 20,
            max_csv_bytes: 512 << 20,
            max_rows: 10_000_000,
            max_archives: 3660,
        }
    }
}

/// Checksum sidecars are a line of text.
const MAX_CHECKSUM_BYTES: u64 = 4096;

/// Verify `zip` against a Binance sidecar: `<64 hex> <space or *><zip name>`.
pub fn verify_checksum(zip: &[u8], sidecar: &[u8], zip_name: &str) -> Result<()> {
    let mut text = sidecar;
    if let Some(rest) = text.strip_suffix(b"\n") {
        text = rest.strip_suffix(b"\r").unwrap_or(rest);
    }
    let name = zip_name.as_bytes();
    if text.len() != 66 + name.len() || text[64] != b' ' || !matches!(text[65], b' ' | b'*') || &text[66..] != name {
        return Err(Error::Checksum("invalid Binance checksum filename or format".into()));
    }
    let mut expected = [0u8; 32];
    for (i, byte) in expected.iter_mut().enumerate() {
        let hex = std::str::from_utf8(&text[2 * i..2 * i + 2]).ok();
        *byte = hex
            .and_then(|h| u8::from_str_radix(h, 16).ok())
            .ok_or_else(|| Error::Checksum("invalid Binance SHA256 digest".into()))?;
    }
    let actual: [u8; 32] = Sha256::digest(zip).into();
    if actual != expected {
        return Err(Error::Checksum("Binance SHA256 mismatch".into()));
    }
    Ok(())
}

fn get(client: &dyn Fetch, url: &str, max_bytes: u64) -> Result<Vec<u8>> {
    let response = client.get(url, max_bytes)?;
    match response.status {
        200 => Ok(response.body),
        404 => Err(Error::NotFound(url.into())),
        status => Err(Error::Http {
            status,
            url: url.into(),
        }),
    }
}

/// Fetch one archive and its sidecar, verify both and return the CSV bytes.
/// A missing ZIP is [`Error::NotFound`]; a missing sidecar is a checksum
/// failure, so skipping missing archives can never bypass verification.
pub fn download(client: &dyn Fetch, spec: &ArchiveSpec, limits: &Limits) -> Result<Vec<u8>> {
    let names = spec.names()?;
    let zip = get(client, &names.zip_url, limits.max_archive_bytes)?;
    let sidecar = match get(client, &names.checksum_url, MAX_CHECKSUM_BYTES) {
        Err(Error::NotFound(_)) => return Err(Error::Checksum("Binance checksum sidecar is missing".into())),
        other => other?,
    };
    verify_checksum(&zip, &sidecar, &names.zip_name)?;
    extract_single(&zip, &names.csv_name, limits.max_csv_bytes)
}

/// What `trl download` should fetch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DownloadConfig {
    pub symbol: String,
    pub market: Market,
    pub kind: Kind,
    /// `None` is auto: monthly for klines and funding rates, else daily.
    pub period: Option<Period>,
    /// Kline interval, e.g. `1m`; ignored for other kinds.
    pub interval: String,
    /// Inclusive `YYYY-MM-DD` bounds.
    pub from: String,
    pub to: String,
    /// Output directory; empty picks the conventional one.
    pub out: PathBuf,
    pub force: bool,
    pub skip_missing: bool,
    pub limits: Limits,
}

impl DownloadConfig {
    pub fn period(&self) -> Period {
        self.period.unwrap_or(match self.kind {
            Kind::Klines | Kind::FundingRate => Period::Monthly,
            _ => Period::Daily,
        })
    }

    /// `data/klines/SYM/INTERVAL`, `data/raw` for trades, else `data/<kind>/SYM`.
    pub fn output(&self) -> PathBuf {
        if !self.out.as_os_str().is_empty() {
            return self.out.clone();
        }
        let data = PathBuf::from("data");
        match self.kind {
            Kind::Klines => data.join("klines").join(&self.symbol).join(&self.interval),
            Kind::Trades => data.join("raw"),
            kind => data.join(kind.as_str()).join(&self.symbol),
        }
    }

    /// The archive dates the window covers: each day, or each whole month
    /// the inclusive window touches.
    pub fn dates(&self) -> Result<Vec<String>> {
        let bad = || invalid("dates must be YYYY-MM-DD in 1970..9999");
        let first = trl_core::time::parse_date(&self.from).ok_or_else(bad)?;
        let last = trl_core::time::parse_date(&self.to).ok_or_else(bad)?;
        if last < first {
            return Err(invalid("--to precedes --from"));
        }
        let too_many = || Error::Limit(format!("archive window exceeds {} files", self.limits.max_archives));
        let mut dates = Vec::new();
        match self.period() {
            Period::Daily => {
                if (last - first + 1) as usize > self.limits.max_archives {
                    return Err(too_many());
                }
                for day in first..=last {
                    let (y, m, d) = trl_core::time::civil_from_days(day);
                    dates.push(format!("{y:04}-{m:02}-{d:02}"));
                }
            }
            Period::Monthly => {
                let (y0, m0, _) = trl_core::time::civil_from_days(first);
                let (y1, m1, _) = trl_core::time::civil_from_days(last);
                let (start, end) = (y0 * 12 + i64::from(m0) - 1, y1 * 12 + i64::from(m1) - 1);
                if (end - start + 1) as usize > self.limits.max_archives {
                    return Err(too_many());
                }
                for month in start..=end {
                    dates.push(format!("{:04}-{:02}", month / 12, month % 12 + 1));
                }
            }
        }
        Ok(dates)
    }
}

/// What a download did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Summary {
    pub downloaded: usize,
    pub skipped: usize,
    pub missing: usize,
}

/// Fetch every archive of the window into the output directory, printing
/// one line per archive. Stops at the first failure; files published before
/// it stay published.
pub fn download_window(config: &DownloadConfig, client: &dyn Fetch, out: &mut dyn Write) -> Result<Summary> {
    if config.limits.max_archive_bytes == 0 || config.limits.max_csv_bytes == 0 || config.limits.max_rows == 0 {
        return Err(invalid("download limits must be positive"));
    }
    let output = config.output();
    let console = |e| Error::Io {
        path: PathBuf::from("<output>"),
        source: e,
    };
    let interval = if config.kind == Kind::Klines {
        config.interval.as_str()
    } else {
        ""
    };
    // Every name is validated before the first request.
    let specs: Vec<ArchiveSpec> = config
        .dates()?
        .into_iter()
        .map(|date| ArchiveSpec {
            market: config.market,
            kind: config.kind,
            period: config.period(),
            symbol: config.symbol.clone(),
            interval: interval.into(),
            date,
        })
        .collect();
    let names: Vec<Names> = specs.iter().map(ArchiveSpec::names).collect::<Result<_>>()?;
    let mut summary = Summary::default();
    for (spec, names) in specs.iter().zip(&names) {
        let destination = output.join(&names.csv_name);
        if destination.exists() && !config.force {
            writeln!(out, "{} already present", names.csv_name).map_err(console)?;
            summary.skipped += 1;
            continue;
        }
        let csv = match download(client, spec, &config.limits) {
            Err(Error::NotFound(_)) if config.skip_missing => {
                writeln!(out, "{} unavailable (404)", names.zip_name).map_err(console)?;
                summary.missing += 1;
                continue;
            }
            other => other?,
        };
        let rows = if spec.kind.typed() {
            Some(count_rows(spec, &csv, &config.limits)?)
        } else {
            None
        };
        std::fs::create_dir_all(&output).map_err(io(&output))?;
        publish(&destination, &csv, config.force)?;
        match rows {
            Some(rows) => writeln!(out, "{} {rows} rows, SHA256 verified", names.csv_name),
            None => writeln!(
                out,
                "{} {} raw CSV bytes (schema unchanged), SHA256 verified",
                names.csv_name,
                csv.len()
            ),
        }
        .map_err(console)?;
        summary.downloaded += 1;
    }
    writeln!(
        out,
        "downloaded {}, already had {}, unavailable {}",
        summary.downloaded, summary.skipped, summary.missing
    )
    .map_err(console)?;
    Ok(summary)
}

#[cfg(test)]
mod tests;
