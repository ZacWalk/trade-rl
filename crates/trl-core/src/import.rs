//! Turning files into series: generic OHLCV CSV, and Binance trade and kline
//! CSVs.

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use crate::bars::{Bar, Market, Series, SeriesKind};
use crate::error::{Result, invalid, io};
use crate::math::positive;
use crate::text::{parse_real, trim};
use crate::time::{MS_PER_DAY, duration_text, format_time_ms};

/// Binance publishes at most a few days of continuous silence; a larger gap
/// is a hole in the archive rather than a quiet market, and is not filled.
const MAX_FILL_BARS: i64 = 400_000;

/// Timestamps at or above this are microseconds: Binance spot archives from
/// 2025 on use them, and in milliseconds this is the year 5138.
const MICROSECONDS_FROM: i64 = 100_000_000_000_000;

fn to_ms(t: i64) -> i64 {
    if t >= MICROSECONDS_FROM { t / 1000 } else { t }
}

fn lines(path: &Path) -> Result<impl Iterator<Item = std::io::Result<Vec<u8>>>> {
    let file = File::open(path).map_err(io(path))?;
    let mut reader = BufReader::with_capacity(1 << 20, file);
    Ok(std::iter::from_fn(move || {
        let mut line = Vec::new();
        match reader.read_until(b'\n', &mut line) {
            Ok(0) => None,
            Ok(_) => {
                while matches!(line.last(), Some(b'\n' | b'\r')) {
                    line.pop();
                }
                Some(Ok(line))
            }
            Err(e) => Some(Err(e)),
        }
    }))
}

// ---- generic OHLCV ---------------------------------------------------------

/// Import `time,open,high,low,close[,volume][,session]`, columns in any
/// order. Time is UTC epoch milliseconds at bar open. A `utc` calendar
/// derives sessions from UTC days and takes no session column; any other
/// calendar requires integer session identifiers from the data producer.
/// Gaps stay gaps: no bar, volume or flow is invented.
pub fn load_ohlcv_csv(path: &Path, symbol: &str, interval_ms: i64, market: &Market) -> Result<Series> {
    let mut line_number = 1usize;
    let fail = |line: usize, message: &str| invalid(format!("{}:{line}: {message}", path.display()));
    let mut rows = lines(path)?;
    let header = match rows.next() {
        Some(line) => line.map_err(io(path))?,
        None => return Err(fail(1, "missing OHLCV header")),
    };
    let header = String::from_utf8(header).map_err(|_| fail(1, "the header is not UTF-8"))?;
    let header = header.strip_prefix('\u{feff}').unwrap_or(&header);
    let columns: Vec<&str> = trim(header).split(',').map(trim).collect();
    const ALLOWED: [&str; 7] = ["time", "open", "high", "low", "close", "volume", "session"];
    for c in &columns {
        if !ALLOWED.contains(c) || columns.iter().filter(|x| *x == c).count() != 1 {
            return Err(fail(1, &format!("unknown or duplicate OHLCV column: {c}")));
        }
    }
    let index = |name: &str| columns.iter().position(|c| *c == name);
    for name in ["time", "open", "high", "low", "close"] {
        if index(name).is_none() {
            return Err(fail(1, &format!("missing OHLCV column: {name}")));
        }
    }
    let has_session = index("session").is_some();
    if has_session != (market.calendar != "utc") {
        return Err(fail(
            1,
            "--calendar utc requires no session column; a supplied calendar requires a session column",
        ));
    }
    let mut market = market.clone();
    market.volume_available = index("volume").is_some();
    let col = |name: &str| index(name).unwrap_or(usize::MAX);
    let (t, o, h, l, c, v, s) = (
        col("time"),
        col("open"),
        col("high"),
        col("low"),
        col("close"),
        col("volume"),
        col("session"),
    );

    let mut bars = Vec::new();
    let mut sessions = Vec::new();
    for line in rows {
        let line = line.map_err(io(path))?;
        line_number += 1;
        let line = std::str::from_utf8(&line).map_err(|_| fail(line_number, "the row is not UTF-8"))?;
        if trim(line).is_empty() {
            continue;
        }
        let cells: Vec<&str> = line.split(',').map(trim).collect();
        if cells.len() != columns.len() {
            return Err(fail(line_number, "OHLCV row width differs from the header"));
        }
        let mut b = Bar {
            time: cells[t]
                .parse::<i64>()
                .map_err(|_| fail(line_number, "time must be integer epoch milliseconds"))?,
            ..Bar::default()
        };
        let session = if has_session {
            cells[s]
                .parse::<i64>()
                .map_err(|_| fail(line_number, "session must be an integer identifier"))?
        } else {
            b.time.div_euclid(MS_PER_DAY)
        };
        let fields: [(&str, usize, &mut f64); 4] = [
            ("open", o, &mut b.open),
            ("high", h, &mut b.high),
            ("low", l, &mut b.low),
            ("close", c, &mut b.close),
        ];
        for (name, i, value) in fields {
            *value = parse_real(cells[i]).filter(|x| x.is_finite()).ok_or_else(|| {
                fail(
                    line_number,
                    &format!("{name} must be a finite number, not a missing value"),
                )
            })?;
        }
        if market.volume_available {
            b.volume = parse_real(cells[v])
                .filter(|x| x.is_finite())
                .ok_or_else(|| fail(line_number, "volume must be a finite number, not a missing value"))?;
        }
        bars.push(b);
        sessions.push(session);
    }
    if bars.is_empty() {
        return Err(fail(line_number, "no OHLCV rows"));
    }
    let series = Series {
        symbol: symbol.into(),
        interval_ms,
        kind: SeriesKind::Ohlcv,
        market,
        bars,
        sessions,
    };
    series.validate()?;
    Ok(series)
}

// ---- Binance trades and klines ---------------------------------------------

/// Aggregates trades or finer klines into fixed-interval bars. An interval
/// with no trades between two that had some becomes a filled bar carrying
/// the last close, so the series is continuous.
#[derive(Debug)]
pub struct BarBuilder {
    interval_ms: i64,
    slot: i64,
    last_time: i64,
    current: Bar,
    open: bool,
    bars: Vec<Bar>,
}

impl BarBuilder {
    pub fn new(interval_ms: i64) -> BarBuilder {
        BarBuilder {
            interval_ms: if interval_ms > 0 { interval_ms } else { 1000 },
            slot: -1,
            last_time: i64::MIN,
            current: Bar::default(),
            open: false,
            bars: Vec::new(),
        }
    }

    pub fn interval(&self) -> i64 {
        self.interval_ms
    }

    fn open_bar(&mut self, slot: i64, price: f64) {
        self.current = Bar {
            time: slot * self.interval_ms,
            open: price,
            high: price,
            low: price,
            close: price,
            ..Bar::default()
        };
        self.slot = slot;
        self.open = true;
    }

    fn close_bar(&mut self) {
        if self.open {
            self.bars.push(self.current);
            self.open = false;
        }
    }

    fn advance(&mut self, slot: i64, price: f64) {
        if !self.open {
            self.open_bar(slot, price);
            return;
        }
        if slot <= self.slot {
            return;
        }
        let carry = self.current.close;
        self.close_bar();
        let gap = slot - self.slot - 1;
        if gap > 0 && gap < MAX_FILL_BARS {
            for i in 1..=gap {
                self.bars.push(Bar {
                    time: (self.slot + i) * self.interval_ms,
                    open: carry,
                    high: carry,
                    low: carry,
                    close: carry,
                    filled: 1,
                    ..Bar::default()
                });
            }
        }
        self.open_bar(slot, price);
    }

    /// Add one trade. Trades within a file are ordered, so one earlier than
    /// the open bar is a timestamp artifact and folds into the bar in progress.
    pub fn push(&mut self, time_ms: i64, price: f64, qty: f64, quote: f64, buyer_is_maker: bool) {
        self.advance(time_ms.div_euclid(self.interval_ms), price);
        self.last_time = self.last_time.max(time_ms);
        let c = &mut self.current;
        c.high = c.high.max(price);
        c.low = c.low.min(price);
        c.close = price;
        c.volume += qty;
        c.quote += quote;
        if !buyer_is_maker {
            c.buy_volume += qty;
        }
        c.trades = c.trades.saturating_add(1);
    }

    /// Merge a complete source bar (a kline). One that does not follow the
    /// previous one is refused, never merged into the open bar.
    pub fn push_bar(&mut self, source: &Bar) -> bool {
        if source.time <= self.last_time {
            return false;
        }
        self.last_time = source.time;
        self.advance(source.time.div_euclid(self.interval_ms), source.open);
        let c = &mut self.current;
        c.high = c.high.max(source.high);
        c.low = c.low.min(source.low);
        c.close = source.close;
        c.volume += source.volume;
        c.quote += source.quote;
        c.buy_volume += source.buy_volume;
        c.trades = c.trades.saturating_add(source.trades);
        true
    }

    /// Whether a source file starting at `time_ms` follows what was already read.
    pub fn continues(&self, time_ms: i64) -> bool {
        time_ms >= self.last_time
    }

    pub fn finish(mut self) -> Vec<Bar> {
        self.close_bar();
        self.bars
    }
}

/// The layout of a Binance CSV.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CsvKind {
    Unknown,
    Trades,
    Klines,
}

impl CsvKind {
    pub fn noun(self) -> &'static str {
        match self {
            CsvKind::Trades => "trades",
            CsvKind::Klines => "klines",
            CsvKind::Unknown => "rows",
        }
    }
}

/// Classify a CSV by its header, or by its width when it has none.
pub fn detect_csv_kind(path: &Path) -> CsvKind {
    let Ok(mut rows) = lines(path) else {
        return CsvKind::Unknown;
    };
    let Some(Ok(line)) = rows.next() else {
        return CsvKind::Unknown;
    };
    if line.starts_with(b"open_time") {
        return CsvKind::Klines;
    }
    if line.starts_with(b"id,") {
        return CsvKind::Trades;
    }
    // Aggregate trades are wide enough for the trades layout, which would
    // read their ids as timestamps and publish bars dated 1970.
    if line.starts_with(b"agg_trade_id") {
        return CsvKind::Unknown;
    }
    match line.iter().filter(|&&c| c == b',').count() {
        10.. => CsvKind::Klines,
        4.. => CsvKind::Trades,
        _ => CsvKind::Unknown,
    }
}

/// The leading numeric fields of a row; `None` when one does not parse.
fn numbers<const N: usize>(line: &[u8]) -> Option<([f64; N], &[u8])> {
    let mut out = [0.0; N];
    let mut rest = line;
    for slot in out.iter_mut() {
        let end = rest.iter().position(|&c| c == b',').unwrap_or(rest.len());
        *slot = std::str::from_utf8(&rest[..end]).ok()?.parse().ok()?;
        rest = rest.get(end + 1..).unwrap_or(&[]);
    }
    Some((out, rest))
}

fn is_header(line: &[u8]) -> bool {
    !line.first().is_some_and(u8::is_ascii_digit)
}

/// Stream one trades CSV (`id,price,qty,quote_qty,time,is_buyer_maker`)
/// into the builder. Returns the number of trades read; malformed rows are
/// skipped.
pub fn load_trades_csv(path: &Path, builder: &mut BarBuilder) -> Result<usize> {
    let mut rows = 0usize;
    let mut first = true;
    for line in lines(path)? {
        let line = line.map_err(io(path))?;
        if line.is_empty() {
            continue;
        }
        if std::mem::take(&mut first) && is_header(&line) {
            continue;
        }
        let Some((f, rest)) = numbers::<5>(&line) else {
            continue;
        };
        let flag = rest.iter().find(|&&c| c != b' ').copied();
        let buyer_is_maker = matches!(flag, Some(b't' | b'T' | b'1'));
        let (price, qty, quote, time) = (f[1], f[2], f[3], f[4]);
        // A non-finite value is not a trade, and the gap fill would carry it
        // forward into every quiet bar after it.
        if !price.is_finite() || !qty.is_finite() || !quote.is_finite() || !positive(time) || time >= 9.2e18 {
            continue;
        }
        let time_ms = to_ms(time as i64);
        if price <= 0.0 || qty < 0.0 || time_ms <= 0 {
            continue;
        }
        // A file starting inside data already read overlaps it, typically a
        // daily archive beside the monthly one covering the same dates.
        if rows == 0 && !builder.continues(time_ms) {
            return Err(invalid(format!(
                "{} starts at {}, inside trades already read; remove overlapping daily/monthly archives",
                path.display(),
                format_time_ms(time_ms)
            )));
        }
        builder.push(
            time_ms,
            price,
            qty,
            if quote > 0.0 { quote } else { price * qty },
            buyer_is_maker,
        );
        rows += 1;
    }
    Ok(rows)
}

/// Stream one klines CSV into the builder. Kline columns carry taker buy
/// volume and the trade count, so a kline file produces the same bar as the
/// trades it was aggregated from.
pub fn load_klines_csv(path: &Path, builder: &mut BarBuilder) -> Result<usize> {
    let mut rows = 0usize;
    let mut first = true;
    for line in lines(path)? {
        let line = line.map_err(io(path))?;
        if line.is_empty() {
            continue;
        }
        if std::mem::take(&mut first) && is_header(&line) {
            continue;
        }
        // open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,...
        let Some((f, _)) = numbers::<10>(&line) else {
            continue;
        };
        if !f.iter().all(|v| v.is_finite()) || f[0] <= 0.0 || f[0] >= 9.2e18 {
            continue;
        }
        let b = Bar {
            time: to_ms(f[0] as i64),
            open: f[1],
            high: f[2],
            low: f[3],
            close: f[4],
            volume: f[5],
            quote: f[7],
            trades: f[8] as i32,
            buy_volume: f[9],
            filled: 0,
        };
        if b.time <= 0 || b.open <= 0.0 || b.close <= 0.0 {
            continue;
        }
        if !builder.push_bar(&b) {
            return Err(invalid(format!(
                "{}: kline at {} does not follow the previous one; remove overlapping daily/monthly archives or duplicate rows",
                path.display(),
                format_time_ms(b.time)
            )));
        }
        rows += 1;
    }
    Ok(rows)
}

/// The interval a kline file was published at, from its first row with
/// both open and close times; 0 when there is none. Binance closes a kline
/// one time unit before the next opens.
pub fn kline_interval_ms(path: &Path) -> i64 {
    let Ok(rows) = lines(path) else {
        return 0;
    };
    for line in rows.map_while(std::result::Result::ok) {
        if line.is_empty() || is_header(&line) {
            continue;
        }
        let Some((f, _)) = numbers::<7>(&line) else {
            continue;
        };
        let span = f[6] - f[0] + 1.0;
        if (1.0..9.2e18).contains(&span) {
            let span = span as i64;
            return if f[0] >= MICROSECONDS_FROM as f64 {
                span / 1000
            } else {
                span
            };
        }
    }
    0
}

/// The CSV files of a directory in name order, or the one file named.
pub fn gather_inputs(input: &Path) -> Vec<PathBuf> {
    if input.is_dir() {
        let mut files: Vec<PathBuf> = std::fs::read_dir(input)
            .into_iter()
            .flatten()
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.is_file() && p.extension().is_some_and(|x| x == "csv"))
            .collect();
        files.sort();
        files
    } else {
        vec![input.to_path_buf()]
    }
}

/// One file read by [`import_binance`].
#[derive(Clone, Debug)]
pub struct SourceFile {
    pub path: PathBuf,
    pub kind: CsvKind,
    pub rows: usize,
}

/// The result of [`import_binance`].
#[derive(Clone, Debug)]
pub struct BinanceImport {
    pub series: Series,
    pub files: Vec<SourceFile>,
    pub trade_rows: usize,
    pub kline_rows: usize,
    /// Bars filled because their interval had no trades.
    pub quiet: usize,
}

/// Aggregate Binance trade and kline CSVs under `input` into bars of
/// `interval_ms`. Klines default to their own interval and accept only whole
/// multiples of it: a kline cannot be split between bars.
pub fn import_binance(input: &Path, interval_ms: Option<i64>, symbol: &str) -> Result<BinanceImport> {
    if !input.exists() {
        return Err(invalid(format!("cannot open {}", input.display())));
    }
    let paths = gather_inputs(input);
    if paths.is_empty() {
        return Err(invalid(format!("no csv files under {}", input.display())));
    }
    let mut files = Vec::with_capacity(paths.len());
    let mut sources = Vec::new();
    for path in paths {
        let kind = detect_csv_kind(&path);
        if kind == CsvKind::Unknown {
            return Err(invalid(format!("unrecognised csv layout in {}", path.display())));
        }
        if kind == CsvKind::Klines {
            sources.push(kline_interval_ms(&path));
        }
        files.push(SourceFile { path, kind, rows: 0 });
    }
    let native = sources.iter().copied().max().unwrap_or(0);
    let interval = match interval_ms {
        Some(ms) => ms,
        None if native > 0 => native,
        None => 1000,
    };
    if interval <= 0 || interval > 365 * MS_PER_DAY {
        return Err(invalid("--interval must be between 1ms and 365d"));
    }
    for &source in &sources {
        if source > 0 && interval % source != 0 {
            let below = interval / source * source;
            let options = if below > 0 {
                format!("{} or {}", duration_text(below), duration_text(below + source))
            } else {
                duration_text(below + source)
            };
            return Err(invalid(format!(
                "--interval {} is not a whole multiple of the {} klines in {}, and a kline cannot be split between bars.\n\
                 Use --interval {options}, or any other whole multiple of {}.",
                duration_text(interval),
                duration_text(source),
                input.display(),
                duration_text(source)
            )));
        }
    }
    let mut builder = BarBuilder::new(interval);
    let (mut trade_rows, mut kline_rows) = (0, 0);
    for file in &mut files {
        file.rows = match file.kind {
            CsvKind::Klines => load_klines_csv(&file.path, &mut builder)?,
            _ => load_trades_csv(&file.path, &mut builder)?,
        };
        match file.kind {
            CsvKind::Klines => kline_rows += file.rows,
            _ => trade_rows += file.rows,
        }
    }
    let bars = builder.finish();
    if bars.is_empty() {
        return Err(invalid("no bars produced"));
    }
    let quiet = bars.iter().filter(|b| b.filled != 0).count();
    let series = Series::binance(symbol, interval, bars);
    series.validate()?;
    Ok(BinanceImport {
        series,
        files,
        trade_rows,
        kline_rows,
        quiet,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synthetic;
    use std::io::Write;

    fn write(dir: &Path, name: &str, text: &str) -> PathBuf {
        let path = dir.join(name);
        std::fs::File::create(&path)
            .unwrap()
            .write_all(text.as_bytes())
            .unwrap();
        path
    }

    #[test]
    fn trades_fill_quiet_intervals() {
        let dir = tempfile::tempdir().unwrap();
        write(
            dir.path(),
            "a.csv",
            "id,price,qty,quote_qty,time,is_buyer_maker\n\
             1,100,2,200,1000,true\n2,101,1,101,1500,false\n3,99,1,99,4200,false\n",
        );
        let out = import_binance(dir.path(), Some(1000), "X").unwrap();
        let bars = &out.series.bars;
        assert_eq!(out.trade_rows, 3);
        assert_eq!(bars.len(), 4);
        assert_eq!((bars[0].open, bars[0].high, bars[0].close), (100.0, 101.0, 101.0));
        assert_eq!((bars[0].volume, bars[0].buy_volume, bars[0].trades), (3.0, 1.0, 2));
        assert_eq!(bars[1].filled, 1);
        assert_eq!(bars[2].close, 101.0);
        assert_eq!((bars[3].time, bars[3].close, out.quiet), (4000, 99.0, 2));
    }

    #[test]
    fn microsecond_trades_are_read_as_milliseconds() {
        let dir = tempfile::tempdir().unwrap();
        write(
            dir.path(),
            "a.csv",
            "1,100,2,200,1735689600000000,true,true\n2,101,1,101,1735689601500000,false,true\n",
        );
        let out = import_binance(dir.path(), Some(1000), "X").unwrap();
        assert_eq!(out.series.bars[0].time, 1_735_689_600_000);
        assert_eq!(out.series.len(), 2);
    }

    #[test]
    fn overlapping_files_are_refused() {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "a.csv", "1,100,2,200,5000,true\n");
        write(dir.path(), "b.csv", "2,100,2,200,1000,true\n");
        let e = import_binance(dir.path(), Some(1000), "X").unwrap_err();
        assert!(e.to_string().contains("inside trades already read"), "{e}");
    }

    #[test]
    fn klines_aggregate_to_whole_multiples_only() {
        let dir = tempfile::tempdir().unwrap();
        let mut text = String::from(
            "open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,taker_buy_quote_volume,ignore\n",
        );
        for i in 0..6i64 {
            let t = 1_700_000_040_000 + i * 60_000;
            text += &format!(
                "{t},{o},{h},{l},{c},10,{},1000,5,4,400,0\n",
                t + 59_999,
                o = 100 + i,
                h = 102 + i,
                l = 99 + i,
                c = 101 + i
            );
        }
        write(dir.path(), "k.csv", &text);
        let native = import_binance(dir.path(), None, "X").unwrap();
        assert_eq!((native.series.interval_ms, native.series.len()), (60_000, 6));
        let wide = import_binance(dir.path(), Some(180_000), "X").unwrap();
        let b = wide.series.bars[0];
        assert_eq!((b.volume, b.trades, b.buy_volume), (10.0, 5, 4.0));
        assert_eq!(wide.series.len(), 3);
        // 1_700_000_040_000 is two minutes into a three-minute slot: k0 | k1 k2 k3 | k4 k5.
        assert_eq!(
            (
                wide.series.bars[1].open,
                wide.series.bars[1].high,
                wide.series.bars[1].close
            ),
            (101.0, 105.0, 104.0)
        );
        let e = import_binance(dir.path(), Some(90_000), "X").unwrap_err().to_string();
        assert!(e.contains("Use --interval 1m or 2m"), "{e}");
    }

    #[test]
    fn csv_kinds() {
        let dir = tempfile::tempdir().unwrap();
        let k = write(dir.path(), "k.csv", "1,2,3,4,5,6,7,8,9,10,11,12\n");
        let t = write(dir.path(), "t.csv", "1,2,3,4,5,true\n");
        let a = write(
            dir.path(),
            "a.csv",
            "agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker\n",
        );
        let n = write(dir.path(), "n.csv", "1,2\n");
        assert_eq!(detect_csv_kind(&k), CsvKind::Klines);
        assert_eq!(detect_csv_kind(&t), CsvKind::Trades);
        assert_eq!(detect_csv_kind(&a), CsvKind::Unknown);
        assert_eq!(detect_csv_kind(&n), CsvKind::Unknown);
    }

    #[test]
    fn ohlcv_import_and_rejections() {
        let dir = tempfile::tempdir().unwrap();
        let market = Market {
            source: "local-csv".into(),
            currency: "USD".into(),
            calendar: "fixture-sessions".into(),
            price_basis: "raw".into(),
            volume_available: false,
        };
        let path = write(dir.path(), "p.csv", &synthetic::session_csv(300));
        let s = load_ohlcv_csv(&path, "TEST", 900_000, &market).unwrap();
        assert_eq!(s, {
            let mut expect = synthetic::session_series(300);
            expect.market.source = "local-csv".into();
            expect
        });

        let utc = Market {
            calendar: "utc".into(),
            currency: "EUR".into(),
            ..market.clone()
        };
        let path = write(
            dir.path(),
            "u.csv",
            "\u{feff}close,low,high,open,time,volume\n100,99,101,100,1577975400000,0\n101,100,102,100,1577976300000,12.5\n",
        );
        let s = load_ohlcv_csv(&path, "OTHER", 900_000, &utc).unwrap();
        assert!(s.market.volume_available);
        assert_eq!(s.sessions, vec![1577975400000 / MS_PER_DAY; 2]);

        for rows in [
            "time,open,high,low,close,session\n1577975400000,100,101,99,NaN,1\n",
            "time,open,high,low,close,session\n1577975400000,100,101,99,0,1\n",
            "time,open,high,low,close,session\n1577975400000,100,99,101,100,1\n",
            "time,open,high,low,close,session\n1577975400000x,100,101,99,100,1\n",
            "time,open,high,low,close,session\n1577975400000,100,101,99,100,1\n1577975400000,100,101,99,100,1\n",
            "time,open,high,low,close,session\n1577975400000,100,101,99,100,2\n1577976300000,100,101,99,100,1\n",
            "time,open,high,low,close,volume,session\n1577975400000,100,101,99,100,,1\n",
            "time,open,high,low,close\n1577975400000,100,101,99,100\n",
            "time,open,high,low,close,session\n1577975400000,100,101,99,100,1\n1577975400900,100,101,99,100,1\n",
            "time,open,high,low,close,colour,session\n1,1,1,1,1,red,1\n",
        ] {
            let path = write(dir.path(), "bad.csv", rows);
            assert!(load_ohlcv_csv(&path, "TEST", 900_000, &market).is_err(), "{rows}");
        }
    }
}
