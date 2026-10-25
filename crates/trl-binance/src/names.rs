//! Archive names and URLs.

use crate::{Result, invalid};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Market {
    Spot,
    /// USD-M futures.
    Um,
    /// COIN-M futures.
    Cm,
}

impl Market {
    pub fn parse(s: &str) -> Option<Market> {
        match s {
            "spot" => Some(Market::Spot),
            "um" => Some(Market::Um),
            "cm" => Some(Market::Cm),
            _ => None,
        }
    }

    fn path(self) -> &'static str {
        match self {
            Market::Spot => "spot",
            Market::Um => "futures/um",
            Market::Cm => "futures/cm",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Klines,
    Trades,
    AggTrades,
    FundingRate,
    Metrics,
}

impl Kind {
    pub fn parse(s: &str) -> Option<Kind> {
        match s {
            "klines" => Some(Kind::Klines),
            "trades" => Some(Kind::Trades),
            "aggTrades" => Some(Kind::AggTrades),
            "fundingRate" => Some(Kind::FundingRate),
            "metrics" => Some(Kind::Metrics),
            _ => None,
        }
    }

    /// Binance's exact spelling, used in paths and file names.
    pub fn as_str(self) -> &'static str {
        match self {
            Kind::Klines => "klines",
            Kind::Trades => "trades",
            Kind::AggTrades => "aggTrades",
            Kind::FundingRate => "fundingRate",
            Kind::Metrics => "metrics",
        }
    }

    /// Klines and trades are validated row by row; the other kinds are
    /// verified and stored as raw CSV.
    pub fn typed(self) -> bool {
        matches!(self, Kind::Klines | Kind::Trades)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Period {
    Daily,
    Monthly,
}

impl Period {
    pub fn parse(s: &str) -> Option<Option<Period>> {
        match s {
            "auto" => Some(None),
            "daily" => Some(Some(Period::Daily)),
            "monthly" => Some(Some(Period::Monthly)),
            _ => None,
        }
    }
}

/// One archive: market, kind, period, symbol, kline interval and date
/// (`YYYY-MM-DD` daily, `YYYY-MM` monthly).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArchiveSpec {
    pub market: Market,
    pub kind: Kind,
    pub period: Period,
    pub symbol: String,
    /// Required for klines; empty for every other kind.
    pub interval: String,
    pub date: String,
}

/// The URLs and names of one archive.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Names {
    pub zip_url: String,
    /// The ZIP URL followed by `.CHECKSUM`: a checksum of the ZIP, not the CSV.
    pub checksum_url: String,
    pub zip_name: String,
    pub csv_name: String,
}

/// The archive's time span in microseconds, and its timestamp unit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Span {
    pub begin_us: i64,
    pub end_us: i64,
    /// Spot archives from 2025 on stamp microseconds; all others milliseconds.
    pub microseconds: bool,
}

/// `[A-Z0-9][A-Z0-9_]{0,63}`; no case conversion.
fn valid_symbol(s: &str) -> bool {
    !s.is_empty()
        && s.len() <= 64
        && s.bytes()
            .enumerate()
            .all(|(i, c)| c.is_ascii_uppercase() || c.is_ascii_digit() || (c == b'_' && i > 0))
}

const INTERVALS: [&str; 16] = [
    "1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1mo", "1M",
];

fn valid_interval(s: &str, market: Market) -> bool {
    if s == "1s" {
        return market == Market::Spot;
    }
    INTERVALS.contains(&s)
}

impl ArchiveSpec {
    pub(crate) fn span(&self) -> Result<Span> {
        let interval_ok = if self.kind == Kind::Klines {
            valid_interval(&self.interval, self.market)
        } else {
            self.interval.is_empty()
        };
        let date_len = match self.period {
            Period::Daily => 10,
            Period::Monthly => 7,
        };
        if !valid_symbol(&self.symbol) || !interval_ok || self.date.len() != date_len {
            return Err(invalid(format!(
                "invalid Binance market, kind, symbol '{}', interval '{}' or date '{}'",
                self.symbol, self.interval, self.date
            )));
        }
        let day = if self.period == Period::Monthly {
            format!("{}-01", self.date)
        } else {
            self.date.clone()
        };
        let begin = trl_core::time::parse_date(&day)
            .ok_or_else(|| invalid("Binance date must be a real calendar date in 1970..9999"))?;
        let (year, month, _) = trl_core::time::civil_from_days(begin);
        let days = match self.period {
            Period::Daily => 1,
            Period::Monthly => i64::from(trl_core::time::days_in_month(year, month)),
        };
        Ok(Span {
            begin_us: begin * 86_400_000_000,
            end_us: (begin + days) * 86_400_000_000,
            microseconds: self.market == Market::Spot && year >= 2025,
        })
    }

    /// Validate the spec and build its names, e.g. `ETHUSDT-trades-2024-07-22.zip`.
    pub fn names(&self) -> Result<Names> {
        self.span()?;
        let period = match self.period {
            Period::Daily => "daily",
            Period::Monthly => "monthly",
        };
        let base = format!(
            "https://data.binance.vision/data/{}/{period}/{}/{}",
            self.market.path(),
            self.kind.as_str(),
            self.symbol
        );
        let (stem, prefix) = if self.kind == Kind::Klines {
            (
                format!("{}-{}-{}", self.symbol, self.interval, self.date),
                format!("{base}/{}", self.interval),
            )
        } else {
            (format!("{}-{}-{}", self.symbol, self.kind.as_str(), self.date), base)
        };
        let zip_name = format!("{stem}.zip");
        let zip_url = format!("{prefix}/{zip_name}");
        Ok(Names {
            checksum_url: format!("{zip_url}.CHECKSUM"),
            zip_url,
            zip_name,
            csv_name: format!("{stem}.csv"),
        })
    }
}
