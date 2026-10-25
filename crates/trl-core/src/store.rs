//! Series stored in a tradedb database.
//!
//! Each series is one fixed-width symbol whose records are keyed by bar
//! open time (epoch ms) and hold the bar's values and session, 72 bytes:
//!
//! ```text
//! 0  open   8  high   16 low   24 close   32 volume   40 quote
//! 48 buy_volume   56 trades:i32   60 filled:i32   64 session:i64
//! ```
//!
//! A variable-width catalog symbol maps each series' symbol id to its name
//! and metadata (`key=value` lines). A save replaces the bars and the catalog
//! entry in one transaction, so readers — another process included — see
//! either the old series or the new one, never a mixture.

use std::path::{Path, PathBuf};

use tradedb::{Batch, Columns, Db, Options, symbol_id};

use crate::bars::{Bar, Market, Series, SeriesKind};
use crate::error::{Result, invalid, io};
use crate::text::{hex64, parse_hex64};

/// Payload bytes of one stored bar.
pub const RECORD_BYTES: u32 = 72;

const CATALOG: u64 = symbol_id("trade-rl/catalog");
const FORMAT_KEY: &str = "trade-rl/format";
const FORMAT: &[u8] = b"1";

/// The tradedb symbol holding a series' bars.
pub fn series_symbol(name: &str) -> u64 {
    symbol_id(&format!("trade-rl/bars/{name}"))
}

/// Series names are short and path-like: `ETHUSDT/15m`, `^GSPC/1d`.
pub fn check_name(name: &str) -> Result<()> {
    let ok = !name.is_empty()
        && name.len() <= 128
        && name
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || b"._/:@^-+".contains(&c));
    if ok {
        Ok(())
    } else {
        Err(invalid(format!(
            "bad series name '{name}': use 1..128 letters, digits and . _ / : @ ^ - +"
        )))
    }
}

/// A series' catalog entry: everything but the bars.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SeriesInfo {
    pub name: String,
    pub symbol: String,
    pub kind: SeriesKind,
    pub interval_ms: i64,
    pub market: Market,
    pub bars: u64,
    /// Open time of the first bar.
    pub first: i64,
    /// Open time of the last bar.
    pub last: i64,
    /// [`Series::hash`] of the stored bars.
    pub hash: u64,
}

impl SeriesInfo {
    fn of(name: &str, s: &Series) -> SeriesInfo {
        SeriesInfo {
            name: name.into(),
            symbol: s.symbol.clone(),
            kind: s.kind,
            interval_ms: s.interval_ms,
            market: s.market.clone(),
            bars: s.len() as u64,
            first: s.bars.first().map_or(0, |b| b.time),
            last: s.bars.last().map_or(0, |b| b.time),
            hash: s.hash(),
        }
    }

    fn to_text(&self) -> String {
        let m = &self.market;
        format!(
            "name={}\nkind={}\nsymbol={}\ninterval_ms={}\nsource={}\ncurrency={}\ncalendar={}\nprice_basis={}\n\
             volume_available={}\nbars={}\nfirst={}\nlast={}\nhash={}\n",
            self.name,
            self.kind.as_str(),
            self.symbol,
            self.interval_ms,
            m.source,
            m.currency,
            m.calendar,
            m.price_basis,
            u8::from(m.volume_available),
            self.bars,
            self.first,
            self.last,
            hex64(self.hash),
        )
    }

    fn from_text(text: &[u8]) -> Result<SeriesInfo> {
        let bad = || invalid("corrupt series catalog entry");
        let text = std::str::from_utf8(text).map_err(|_| bad())?;
        let get = |key: &str| {
            text.lines()
                .find_map(|l| l.split_once('=').filter(|(k, _)| *k == key).map(|(_, v)| v))
                .ok_or_else(bad)
        };
        let number = |key: &str| get(key)?.parse::<i64>().map_err(|_| bad());
        Ok(SeriesInfo {
            name: get("name")?.into(),
            symbol: get("symbol")?.into(),
            kind: SeriesKind::parse(get("kind")?).ok_or_else(bad)?,
            interval_ms: number("interval_ms")?,
            market: Market {
                source: get("source")?.into(),
                currency: get("currency")?.into(),
                calendar: get("calendar")?.into(),
                price_basis: get("price_basis")?.into(),
                volume_available: get("volume_available")? == "1",
            },
            bars: u64::try_from(number("bars")?).map_err(|_| bad())?,
            first: number("first")?,
            last: number("last")?,
            hash: parse_hex64(get("hash")?).ok_or_else(bad)?,
        })
    }
}

/// A database of bar series.
pub struct Store {
    db: Db,
    path: Option<PathBuf>,
}

impl Store {
    /// Open or create the database at `path`, creating its directory.
    pub fn open(path: impl AsRef<Path>) -> Result<Store> {
        let path = path.as_ref();
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent).map_err(io(parent))?;
        }
        let db = Options::new().open(path)?;
        Self::adopt(db, Some(path))
    }

    /// Open an existing database for reading only.
    pub fn open_read_only(path: impl AsRef<Path>) -> Result<Store> {
        let path = path.as_ref();
        if !path.exists() {
            return Err(invalid(format!(
                "no bar database at {}; import bars with `trl bars`",
                path.display()
            )));
        }
        let db = Options::new().read_only(true).open(path)?;
        Self::adopt(db, Some(path))
    }

    /// A private in-memory database, for tests.
    pub fn memory() -> Store {
        Store {
            db: Db::memory(),
            path: None,
        }
    }

    fn adopt(db: Db, path: Option<&Path>) -> Result<Store> {
        match db.get_meta(FORMAT_KEY)? {
            Some(v) if v != FORMAT => {
                return Err(invalid(format!(
                    "{} was written by an incompatible trade-rl (format {})",
                    path.map_or_else(String::new, |p| p.display().to_string()),
                    String::from_utf8_lossy(&v)
                )));
            }
            _ => {}
        }
        Ok(Store {
            db,
            path: path.map(Path::to_path_buf),
        })
    }

    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }

    /// The underlying database.
    pub fn db(&self) -> &Db {
        &self.db
    }

    /// Every series, by name.
    pub fn list(&self) -> Result<Vec<SeriesInfo>> {
        let mut out = Vec::new();
        for entry in self.db.scan(CATALOG, ..)? {
            out.push(SeriesInfo::from_text(entry?.value.as_bytes())?);
        }
        out.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(out)
    }

    pub fn info(&self, name: &str) -> Result<Option<SeriesInfo>> {
        check_name(name)?;
        match self.db.get(CATALOG, series_symbol(name))? {
            Some(v) => Ok(Some(SeriesInfo::from_text(v.as_bytes())?)),
            None => Ok(None),
        }
    }

    /// The series named, or the only one when no name is given.
    pub fn resolve(&self, name: Option<&str>) -> Result<SeriesInfo> {
        if let Some(name) = name {
            return self.info(name)?.ok_or_else(|| missing(name, self));
        }
        let mut all = self.list()?;
        match all.len() {
            1 => Ok(all.remove(0)),
            0 => Err(invalid("the bar database holds no series; import some with `trl bars`")),
            _ => Err(invalid(format!(
                "the bar database holds {} series; choose one with --series ({})",
                all.len(),
                all.iter().map(|s| s.name.as_str()).collect::<Vec<_>>().join(", ")
            ))),
        }
    }

    /// Read a whole series from one consistent snapshot.
    pub fn load(&self, name: &str) -> Result<Series> {
        check_name(name)?;
        let id = series_symbol(name);
        let snap = self.db.snapshot()?;
        let info = match snap.get(CATALOG, id)? {
            Some(v) => SeriesInfo::from_text(v.as_bytes())?,
            None => return Err(missing(name, self)),
        };
        let mut cols = Columns::new();
        snap.read_columns(id, .., &mut cols)?;
        if cols.len() as u64 != info.bars || (!cols.is_empty() && cols.record_size() != RECORD_BYTES as usize) {
            return Err(invalid(format!("series {name} does not match its catalog entry")));
        }
        let generic = info.kind == SeriesKind::Ohlcv;
        let mut bars = Vec::with_capacity(cols.len());
        let mut sessions = Vec::with_capacity(if generic { cols.len() } else { 0 });
        for (time, _, p) in cols.iter() {
            let (bar, session) = decode(time, p);
            bars.push(bar);
            if generic {
                sessions.push(session);
            }
        }
        Ok(Series {
            symbol: info.symbol,
            interval_ms: info.interval_ms,
            kind: info.kind,
            market: info.market,
            bars,
            sessions,
        })
    }

    /// Store `series` under `name` in one transaction. An existing series of
    /// that name is replaced only when `replace` is set.
    pub fn save(&self, name: &str, series: &Series, replace: bool) -> Result<SeriesInfo> {
        check_name(name)?;
        series.validate()?;
        if series.is_empty() {
            return Err(invalid("refusing to store a series with no bars"));
        }
        if !replace && self.info(name)?.is_some() {
            return Err(invalid(format!(
                "refusing to overwrite series {name}; pass --replace to rebuild it"
            )));
        }
        let info = SeriesInfo::of(name, series);
        let id = series_symbol(name);
        let n = series.len();
        let mut batch = Batch::with_capacity(n + 1, n * RECORD_BYTES as usize + 512);
        batch.drop_symbol(id).define_symbol(id, RECORD_BYTES);
        let mut record = [0u8; RECORD_BYTES as usize];
        for (i, b) in series.bars.iter().enumerate() {
            encode(b, series.session(i), &mut record);
            batch.put(id, b.time as u64, &record);
        }
        batch.overwrite(CATALOG, id, 0, info.to_text().as_bytes());
        batch.set_meta(FORMAT_KEY, FORMAT);
        self.db.write(&batch)?;
        Ok(info)
    }

    /// Delete a series. Returns whether it existed.
    pub fn remove(&self, name: &str) -> Result<bool> {
        check_name(name)?;
        let id = series_symbol(name);
        let existed = self.db.get(CATALOG, id)?.is_some();
        let mut batch = Batch::new();
        batch.drop_symbol(id).delete(CATALOG, id, 0);
        self.db.write(&batch)?;
        Ok(existed)
    }
}

fn missing(name: &str, store: &Store) -> crate::Error {
    let known = store
        .list()
        .map(|all| all.into_iter().map(|s| s.name).collect::<Vec<_>>().join(", "))
        .unwrap_or_default();
    if known.is_empty() {
        invalid(format!("no series named {name}; the bar database is empty"))
    } else {
        invalid(format!("no series named {name}; the bar database holds: {known}"))
    }
}

fn encode(b: &Bar, session: i64, out: &mut [u8; RECORD_BYTES as usize]) {
    let values = [b.open, b.high, b.low, b.close, b.volume, b.quote, b.buy_volume];
    for (i, v) in values.iter().enumerate() {
        out[i * 8..i * 8 + 8].copy_from_slice(&v.to_le_bytes());
    }
    out[56..60].copy_from_slice(&b.trades.to_le_bytes());
    out[60..64].copy_from_slice(&b.filled.to_le_bytes());
    out[64..72].copy_from_slice(&session.to_le_bytes());
}

fn decode(time: u64, p: &[u8]) -> (Bar, i64) {
    let f = |i: usize| f64::from_le_bytes(p[i * 8..i * 8 + 8].try_into().unwrap());
    let bar = Bar {
        time: time as i64,
        open: f(0),
        high: f(1),
        low: f(2),
        close: f(3),
        volume: f(4),
        quote: f(5),
        buy_volume: f(6),
        trades: i32::from_le_bytes(p[56..60].try_into().unwrap()),
        filled: i32::from_le_bytes(p[60..64].try_into().unwrap()),
    };
    (bar, i64::from_le_bytes(p[64..72].try_into().unwrap()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synthetic;

    #[test]
    fn save_load_round_trip() {
        let store = Store::memory();
        let s = synthetic::trend_series(500);
        let info = store.save("ETHUSDT/15m", &s, false).unwrap();
        assert_eq!(info.bars, 500);
        assert_eq!(info.hash, s.hash());
        let back = store.load("ETHUSDT/15m").unwrap();
        assert_eq!(back, s);
        assert_eq!(store.list().unwrap(), vec![info.clone()]);
        assert_eq!(store.resolve(None).unwrap(), info);
    }

    #[test]
    fn generic_sessions_round_trip() {
        let store = Store::memory();
        let s = synthetic::session_series(300);
        store.save("SPX/1h", &s, false).unwrap();
        assert_eq!(store.load("SPX/1h").unwrap(), s);
    }

    #[test]
    fn replace_is_explicit_and_atomic() {
        let store = Store::memory();
        let a = synthetic::trend_series(200);
        let b = synthetic::trend_series(150);
        store.save("A/15m", &a, false).unwrap();
        assert!(store.save("A/15m", &b, false).is_err());
        assert_eq!(store.load("A/15m").unwrap(), a);
        store.save("A/15m", &b, true).unwrap();
        assert_eq!(store.load("A/15m").unwrap(), b);
        store.save("B/15m", &a, false).unwrap();
        assert!(store.resolve(None).is_err());
        assert!(store.remove("A/15m").unwrap());
        assert!(!store.remove("A/15m").unwrap());
        assert_eq!(store.list().unwrap().len(), 1);
        assert!(store.load("A/15m").is_err());
    }

    #[test]
    fn names_are_checked() {
        for bad in ["", "a b", "x\ny", "a=b", &"x".repeat(129)] {
            assert!(check_name(bad).is_err(), "{bad:?}");
        }
        check_name("^GSPC/1d").unwrap();
    }

    #[test]
    fn file_store_reopens() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sub").join("bars.tradedb");
        let s = synthetic::trend_series(100);
        {
            let store = Store::open(&path).unwrap();
            store.save("X/15m", &s, false).unwrap();
        }
        let store = Store::open_read_only(&path).unwrap();
        assert_eq!(store.load("X/15m").unwrap(), s);
        assert!(Store::open_read_only(dir.path().join("missing.tradedb")).is_err());
    }
}
