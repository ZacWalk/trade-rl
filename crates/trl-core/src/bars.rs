//! Bars and series: the data every command reads.

use crate::error::{Result, invalid};
use crate::math::{finite, positive};
use crate::text::{Fnv, valid_label};
use crate::time::{MS_PER_DAY, duration_text};

/// One bar, stamped with its open time in epoch milliseconds (UTC).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Bar {
    pub time: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    /// Base asset traded.
    pub volume: f64,
    /// Quote asset traded.
    pub quote: f64,
    /// Base asset bought by the aggressor.
    pub buy_volume: f64,
    pub trades: i32,
    /// 1 when the interval had no trades and the bar carries the last close.
    pub filled: i32,
}

/// Bytes of [`Bar::to_bytes`].
pub const BAR_BYTES: usize = 72;

impl Bar {
    /// The 72-byte little-endian layout: time, seven doubles, trades, filled.
    pub fn to_bytes(&self) -> [u8; BAR_BYTES] {
        let mut out = [0u8; BAR_BYTES];
        out[0..8].copy_from_slice(&self.time.to_le_bytes());
        for (i, v) in self.prices().iter().enumerate() {
            out[8 + i * 8..16 + i * 8].copy_from_slice(&v.to_le_bytes());
        }
        out[64..68].copy_from_slice(&self.trades.to_le_bytes());
        out[68..72].copy_from_slice(&self.filled.to_le_bytes());
        out
    }

    pub fn from_bytes(b: &[u8; BAR_BYTES]) -> Bar {
        let f = |i: usize| f64::from_le_bytes(b[8 + i * 8..16 + i * 8].try_into().unwrap());
        Bar {
            time: i64::from_le_bytes(b[0..8].try_into().unwrap()),
            open: f(0),
            high: f(1),
            low: f(2),
            close: f(3),
            volume: f(4),
            quote: f(5),
            buy_volume: f(6),
            trades: i32::from_le_bytes(b[64..68].try_into().unwrap()),
            filled: i32::from_le_bytes(b[68..72].try_into().unwrap()),
        }
    }

    fn prices(&self) -> [f64; 7] {
        [
            self.open,
            self.high,
            self.low,
            self.close,
            self.volume,
            self.quote,
            self.buy_volume,
        ]
    }
}

/// Where the bars came from and how they may be read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Market {
    pub source: String,
    pub currency: String,
    /// `utc` (sessions are UTC days) or a producer-supplied calendar name.
    pub calendar: String,
    /// `raw` or `adjusted`.
    pub price_basis: String,
    pub volume_available: bool,
}

impl Market {
    /// The metadata of bars aggregated from Binance trades or klines.
    pub fn binance() -> Market {
        Market {
            source: "binance".into(),
            currency: "unspecified".into(),
            calendar: "utc".into(),
            price_basis: "raw".into(),
            volume_available: true,
        }
    }
}

/// The two kinds of series, which the research contract treats differently.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SeriesKind {
    /// Continuous 24/7 bars built from Binance trades or klines: quiet
    /// intervals are filled, and taker flow and trade counts are known.
    Binance,
    /// Generic OHLC(V) imported from CSV: gaps stay gaps, sessions come from
    /// the producer, and no flow is invented.
    Ohlcv,
}

impl SeriesKind {
    pub fn as_str(self) -> &'static str {
        match self {
            SeriesKind::Binance => "binance",
            SeriesKind::Ohlcv => "ohlcv",
        }
    }

    pub fn parse(s: &str) -> Option<SeriesKind> {
        match s {
            "binance" => Some(SeriesKind::Binance),
            "ohlcv" => Some(SeriesKind::Ohlcv),
            _ => None,
        }
    }
}

/// A run of bars of one instrument at one interval.
#[derive(Clone, Debug, PartialEq)]
pub struct Series {
    /// Instrument, e.g. `ETHUSDT`.
    pub symbol: String,
    pub interval_ms: i64,
    pub kind: SeriesKind,
    pub market: Market,
    pub bars: Vec<Bar>,
    /// Session identifier of each bar; only [`SeriesKind::Ohlcv`] stores them.
    pub sessions: Vec<i64>,
}

impl Series {
    pub fn binance(symbol: impl Into<String>, interval_ms: i64, bars: Vec<Bar>) -> Series {
        Series {
            symbol: symbol.into(),
            interval_ms,
            kind: SeriesKind::Binance,
            market: Market::binance(),
            bars,
            sessions: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.bars.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bars.is_empty()
    }

    pub fn generic(&self) -> bool {
        self.kind == SeriesKind::Ohlcv
    }

    /// The session bar `i` belongs to: the UTC day unless the producer said otherwise.
    pub fn session(&self, i: usize) -> i64 {
        if self.generic() {
            self.sessions[i]
        } else {
            self.bars[i].time.div_euclid(MS_PER_DAY)
        }
    }

    /// The index of the bar opening exactly at `time`.
    pub fn index_at_time(&self, time: i64) -> Option<usize> {
        let i = self.bars.partition_point(|b| b.time < time);
        (i < self.bars.len() && self.bars[i].time == time).then_some(i)
    }

    /// The name of the bar format, recorded in run manifests.
    pub fn format_name(&self) -> &'static str {
        match self.kind {
            SeriesKind::Binance => "binance-bars-v1",
            SeriesKind::Ohlcv => "ohlcv-bars-v1",
        }
    }

    /// The conventional store name, e.g. `ETHUSDT/15m`.
    pub fn default_name(&self) -> String {
        format!("{}/{}", self.symbol, duration_text(self.interval_ms))
    }

    /// The first `n` bars (and sessions) as their own series.
    pub fn prefix(&self, n: usize) -> Series {
        Series {
            symbol: self.symbol.clone(),
            interval_ms: self.interval_ms,
            kind: self.kind,
            market: self.market.clone(),
            bars: self.bars[..n].to_vec(),
            sessions: if self.generic() {
                self.sessions[..n].to_vec()
            } else {
                Vec::new()
            },
        }
    }

    /// FNV-1a over the symbol, interval and bars (plus, for OHLCV, the market
    /// metadata and sessions).
    pub fn hash(&self) -> u64 {
        let mut h = Fnv::new();
        h.feed(self.symbol.as_bytes());
        h.feed(&self.interval_ms.to_le_bytes());
        for b in &self.bars {
            h.feed(&b.to_bytes());
        }
        if self.generic() {
            h.feed(b"ohlcv-bars-v1");
            h.feed(&u32::from(self.market.volume_available).to_le_bytes());
            for text in [
                &self.market.source,
                &self.market.currency,
                &self.market.calendar,
                &self.market.price_basis,
            ] {
                h.feed(&(text.len() as u32).to_le_bytes());
                h.feed(text.as_bytes());
            }
            for s in &self.sessions {
                h.feed(&s.to_le_bytes());
            }
        }
        h.digest()
    }

    /// Check the invariants every reader relies on.
    pub fn validate(&self) -> Result<()> {
        if self.interval_ms <= 0 || self.interval_ms > 365 * MS_PER_DAY {
            return Err(invalid(format!("interval {}ms is outside 1ms..365d", self.interval_ms)));
        }
        if !valid_label(&self.symbol, 64) {
            return Err(invalid("symbol must be 1..64 printable characters"));
        }
        if self.bars.len() > i32::MAX as usize {
            return Err(invalid("too many bars"));
        }
        match self.kind {
            SeriesKind::Ohlcv => self.validate_generic(),
            SeriesKind::Binance => self.validate_binance(),
        }
    }

    fn validate_binance(&self) -> Result<()> {
        if !self.sessions.is_empty() {
            return Err(invalid("Binance series derive their sessions from UTC days"));
        }
        for (i, b) in self.bars.iter().enumerate() {
            if b.time < 0 || (i > 0 && b.time - self.bars[i - 1].time < self.interval_ms) {
                return Err(invalid(format!(
                    "bar {}: times must be nonnegative and at least one interval apart",
                    i + 1
                )));
            }
            if ![b.open, b.high, b.low, b.close, b.volume, b.quote, b.buy_volume]
                .iter()
                .all(|&v| finite(v))
            {
                return Err(invalid(format!("bar {}: values must be finite", i + 1)));
            }
        }
        Ok(())
    }

    fn validate_generic(&self) -> Result<()> {
        let m = &self.market;
        if !valid_label(&m.source, 256)
            || !valid_label(&m.currency, 256)
            || !valid_label(&m.calendar, 256)
            || (m.price_basis != "raw" && m.price_basis != "adjusted")
        {
            return Err(invalid("invalid OHLCV interval, symbol or market metadata"));
        }
        if self.sessions.len() != self.bars.len() {
            return Err(invalid("OHLCV sessions must match the bar count"));
        }
        for (i, b) in self.bars.iter().enumerate() {
            let row = format!("OHLCV record {}: ", i + 1);
            if b.time < 0
                || b.time > i64::MAX - self.interval_ms
                || (i > 0 && b.time - self.bars[i - 1].time < self.interval_ms)
            {
                return Err(invalid(
                    row + "timestamps must be increasing, non-overlapping epoch milliseconds",
                ));
            }
            if !positive(b.low)
                || !finite(b.open)
                || !finite(b.high)
                || !finite(b.low)
                || !finite(b.close)
                || b.low > b.open.min(b.close)
                || b.high < b.open.max(b.close)
                || b.high < b.low
            {
                return Err(invalid(
                    row + "expected finite positive OHLC prices with low <= open/close <= high",
                ));
            }
            if !b.volume.is_finite()
                || b.volume < 0.0
                || (!m.volume_available && b.volume != 0.0)
                || b.quote != 0.0
                || b.buy_volume != 0.0
                || b.trades != 0
                || b.filled != 0
            {
                return Err(invalid(row + "invalid volume or unsupported flow/filled fields"));
            }
            let s = self.sessions[i];
            if s < 0 || (i > 0 && s < self.sessions[i - 1]) || (m.calendar == "utc" && s != b.time / MS_PER_DAY) {
                return Err(invalid(
                    row + "session identifiers must be nonnegative and nondecreasing and match the declared calendar",
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bar(time: i64, close: f64) -> Bar {
        Bar {
            time,
            open: close,
            high: close + 1.0,
            low: close - 1.0,
            close,
            ..Bar::default()
        }
    }

    #[test]
    fn bytes_round_trip() {
        let b = Bar {
            time: 123,
            open: 1.0,
            high: 2.0,
            low: 0.5,
            close: 1.5,
            volume: 10.0,
            quote: 15.0,
            buy_volume: 4.0,
            trades: 7,
            filled: 1,
        };
        assert_eq!(Bar::from_bytes(&b.to_bytes()), b);
    }

    #[test]
    fn index_at_time_is_exact() {
        let s = Series::binance("X", 1000, vec![bar(0, 10.0), bar(1000, 11.0), bar(3000, 12.0)]);
        assert_eq!(s.index_at_time(1000), Some(1));
        assert_eq!(s.index_at_time(2000), None);
        assert_eq!(s.index_at_time(5000), None);
    }

    #[test]
    fn hash_depends_on_everything() {
        let a = Series::binance("X", 1000, vec![bar(0, 10.0), bar(1000, 11.0)]);
        let mut b = a.clone();
        b.bars[1].close = 11.5;
        let mut c = a.clone();
        c.symbol = "Y".into();
        assert_ne!(a.hash(), b.hash());
        assert_ne!(a.hash(), c.hash());
        assert_eq!(a.hash(), a.clone().hash());
    }

    #[test]
    fn generic_validation() {
        let mut s = Series {
            symbol: "SPX".into(),
            interval_ms: MS_PER_DAY,
            kind: SeriesKind::Ohlcv,
            market: Market {
                source: "local-csv".into(),
                currency: "USD".into(),
                calendar: "utc".into(),
                price_basis: "raw".into(),
                volume_available: false,
            },
            bars: vec![bar(0, 10.0), bar(MS_PER_DAY, 11.0)],
            sessions: vec![0, 1],
        };
        s.validate().unwrap();
        s.sessions[1] = 5;
        assert!(s.validate().is_err());
        s.sessions[1] = 1;
        s.bars[1].low = 20.0;
        assert!(s.validate().is_err());
    }
}
