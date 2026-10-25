//! Typed validation of Binance kline and trade CSVs.
//!
//! All klines have 12 columns; spot trades have 7 (with `isBestMatch`), and
//! futures trades 6. An optional header must match the known schema, in
//! snake_case or Binance's camelCase. Numbers must be nonnegative JSON
//! number lexemes, ids and counts exact u64, booleans `true`/`false`. The
//! timestamp unit is chosen by market and archive date, never guessed: spot
//! archives from 2025 on stamp microseconds, all others milliseconds. Every
//! row's time must fall inside the archive's own day or month.

use crate::names::{ArchiveSpec, Kind, Market, Span};
use crate::{Error, Limits, Result};

fn parse(row: u64, message: &str) -> Error {
    Error::Parse(format!("Binance CSV data row {}: {message}", row + 1))
}

/// A nonnegative JSON number: `(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?`.
fn json_number(s: &[u8]) -> bool {
    let digits = |s: &[u8], at: usize| s[at..].iter().take_while(|c| c.is_ascii_digit()).count();
    let mut at = match s.first() {
        Some(b'0') => 1,
        Some(b'1'..=b'9') => digits(s, 0),
        _ => return false,
    };
    if s.get(at) == Some(&b'.') {
        let n = digits(s, at + 1);
        if n == 0 {
            return false;
        }
        at += 1 + n;
    }
    if matches!(s.get(at), Some(b'e' | b'E')) {
        at += 1;
        if matches!(s.get(at), Some(b'+' | b'-')) {
            at += 1;
        }
        let n = digits(s, at);
        if n == 0 {
            return false;
        }
        at += n;
    }
    at == s.len()
}

fn number(text: &str, row: u64) -> Result<()> {
    if text.len() > 256 {
        return Err(Error::Limit(format!(
            "Binance CSV data row {}: numeric field over 256 bytes",
            row + 1
        )));
    }
    if json_number(text.as_bytes()) {
        Ok(())
    } else {
        Err(parse(row, "invalid numeric field"))
    }
}

fn integer(text: &str, row: u64) -> Result<u64> {
    number(text, row)?;
    if !text.bytes().all(|c| c.is_ascii_digit()) {
        return Err(parse(row, "integers must contain decimal digits only"));
    }
    text.parse().map_err(|_| parse(row, "integer exceeds u64"))
}

fn timestamp(text: &str, span: &Span, row: u64) -> Result<i64> {
    let value = integer(text, row)?;
    let factor = if span.microseconds { 1 } else { 1000 };
    value
        .checked_mul(factor)
        .and_then(|us| i64::try_from(us).ok())
        .ok_or_else(|| {
            Error::Limit(format!(
                "Binance CSV data row {}: timestamp exceeds int64 microseconds",
                row + 1
            ))
        })
}

fn boolean(text: &str, row: u64) -> Result<bool> {
    if text.eq_ignore_ascii_case("true") {
        Ok(true)
    } else if text.eq_ignore_ascii_case("false") {
        Ok(false)
    } else {
        Err(parse(row, "invalid boolean"))
    }
}

fn is_header(spec: &ArchiveSpec, f: &[&str]) -> bool {
    let cm = spec.market == Market::Cm;
    let any = |s: &str, names: &[&str]| names.contains(&s);
    if spec.kind == Kind::Trades {
        return any(f[0], &["id", "trade_id"])
            && f[1] == "price"
            && f[2] == "qty"
            && if cm {
                any(f[3], &["base_qty", "baseQty"])
            } else {
                any(f[3], &["quote_qty", "quoteQty"])
            }
            && f[4] == "time"
            && any(f[5], &["is_buyer_maker", "isBuyerMaker"])
            && (f.len() == 6 || any(f[6], &["is_best_match", "isBestMatch"]));
    }
    // Binance's published COIN-M klines label their base-asset columns
    // quote_volume and taker_buy_quote_volume, exactly as USD-M does.
    any(f[0], &["open_time", "openTime"])
        && f[1] == "open"
        && f[2] == "high"
        && f[3] == "low"
        && f[4] == "close"
        && f[5] == "volume"
        && any(f[6], &["close_time", "closeTime"])
        && if cm {
            any(f[7], &["base_volume", "base_asset_volume", "quote_volume"])
        } else {
            any(f[7], &["quote_volume", "quote_asset_volume"])
        }
        && any(f[8], &["count", "number_of_trades"])
        && if cm {
            f[9] == "taker_buy_volume"
        } else {
            any(f[9], &["taker_buy_volume", "taker_buy_base_asset_volume"])
        }
        && if cm {
            any(
                f[10],
                &[
                    "taker_buy_base_volume",
                    "taker_buy_base_asset_volume",
                    "taker_buy_quote_volume",
                ],
            )
        } else {
            any(f[10], &["taker_buy_quote_volume", "taker_buy_quote_asset_volume"])
        }
        && f[11] == "ignore"
}

/// Validate a kline or trade CSV against its archive and count its data rows.
pub fn count_rows(spec: &ArchiveSpec, csv: &[u8], limits: &Limits) -> Result<u64> {
    if !spec.kind.typed() {
        return Err(crate::invalid("typed validation covers klines and trades only"));
    }
    let span = spec.span()?;
    if csv.len() as u64 > limits.max_csv_bytes {
        return Err(Error::Limit("Binance CSV exceeds the byte limit".into()));
    }
    let expected = match (spec.kind, spec.market) {
        (Kind::Klines, _) => 12,
        (_, Market::Spot) => 7,
        _ => 6,
    };
    let text = std::str::from_utf8(csv).map_err(|_| Error::Parse("Binance CSV is not UTF-8".into()))?;
    let body = text.strip_suffix('\n').unwrap_or(text);
    let mut rows = 0u64;
    for (line_index, line) in body.split('\n').enumerate() {
        let line = line.strip_suffix('\r').unwrap_or(line);
        let f: Vec<&str> = line.split(',').collect();
        if f.len() != expected {
            return Err(parse(rows, &format!("expected {expected} columns, found {}", f.len())));
        }
        if line_index == 0 && is_header(spec, &f) {
            continue;
        }
        if rows >= limits.max_rows {
            return Err(Error::Limit(format!(
                "Binance CSV row limit of {} exceeded; raise --max-rows",
                limits.max_rows
            )));
        }
        let time = if spec.kind == Kind::Klines {
            let open = timestamp(f[0], &span, rows)?;
            let close = timestamp(f[6], &span, rows)?;
            integer(f[8], rows)?;
            for (i, field) in f.iter().enumerate() {
                if !matches!(i, 0 | 6 | 8) {
                    number(field, rows)?;
                }
            }
            if close < open {
                return Err(parse(rows, "kline closes before it opens"));
            }
            open
        } else {
            integer(f[0], rows)?;
            for field in &f[1..4] {
                number(field, rows)?;
            }
            let time = timestamp(f[4], &span, rows)?;
            boolean(f[5], rows)?;
            if f.len() == 7 {
                boolean(f[6], rows)?;
            }
            time
        };
        if time < span.begin_us || time >= span.end_us {
            return Err(parse(rows, "timestamp is outside its archive period"));
        }
        rows += 1;
    }
    if rows == 0 {
        return Err(Error::Parse("Binance CSV contains no data rows".into()));
    }
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::names::Period;

    #[test]
    fn json_numbers() {
        for ok in ["0", "12", "0.5", "1e5", "1.25E-3", "4.15070000000000000000000001"] {
            assert!(json_number(ok.as_bytes()), "{ok}");
        }
        for bad in ["", "-1", "01", "1.", ".5", "1e", "nan", "1,0", " 1"] {
            assert!(!json_number(bad.as_bytes()), "{bad}");
        }
    }

    fn spec(market: Market, kind: Kind, date: &str) -> ArchiveSpec {
        ArchiveSpec {
            market,
            kind,
            period: if date.len() == 7 {
                Period::Monthly
            } else {
                Period::Daily
            },
            symbol: "ETHUSDT".into(),
            interval: if kind == Kind::Klines {
                "1m".into()
            } else {
                String::new()
            },
            date: date.into(),
        }
    }

    #[test]
    fn archive_samples_validate() {
        let cases: [(Market, Kind, &str, &str); 8] = [
            (
                Market::Cm,
                Kind::Klines,
                "2025-01-01",
                "open_time,open,high,low,close,volume,close_time,base_volume,count,taker_buy_volume,taker_buy_base_volume,ignore\n\
                 1735689600000,9635.0,9640.0,9630.0,9639.0,206,1735689659999,2.13660389,48,119,1.23424865,0\n",
            ),
            (
                Market::Cm,
                Kind::Klines,
                "2025-01",
                "open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,taker_buy_quote_volume,ignore\n\
                 1735689600000,9635.0,9640.0,9630.0,9639.0,206,1735689659999,2.13660389,48,119,1.23424865,0\n",
            ),
            (
                Market::Cm,
                Kind::Trades,
                "2025-01-01",
                "id,price,qty,base_qty,time,is_buyer_maker\n9007199254740993,9635.0,1,0.01037883,1735689600123,false\n",
            ),
            (
                Market::Spot,
                Kind::Klines,
                "2025-01-01",
                "1735689600000123,4.15070000000000000000000001,4.1587,4.1506,4.1554,539.23,1735689659999999,2240.398609,18446744073709551615,401.82,1669.981213,0\n",
            ),
            (
                Market::Spot,
                Kind::Trades,
                "2024-07-22",
                "18446744073709551615,0.123456789012345678901234567890,2.00000000,0.246913578024691357802469135780,1721606400123,True,False\n",
            ),
            (
                Market::Spot,
                Kind::Trades,
                "2025-01-01",
                "id,price,qty,quoteQty,time,isBuyerMaker,isBestMatch\n\
                 9007199254740993,0.123456789012345678901234567890,2,0.246913578024691357802469135780,1735689600123456,false,true\n",
            ),
            (
                Market::Um,
                Kind::Klines,
                "2025-01-01",
                "open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,taker_buy_quote_volume,ignore\r\n\
                 1735689600000,4.15070000000000000000000001,4.1587,4.1506,4.1554,539.23,1735689659999,2240.398609,18446744073709551615,401.82,1669.981213,0\r\n",
            ),
            (
                Market::Um,
                Kind::Trades,
                "2025-01-01",
                "id,price,qty,quote_qty,time,is_buyer_maker\n9007199254740993,0.123456789012345678901234567890,2,0.246913578024691357802469135780,1735689600123,true\n",
            ),
        ];
        for (market, kind, date, csv) in cases {
            let s = spec(market, kind, date);
            assert_eq!(count_rows(&s, csv.as_bytes(), &Limits::default()).unwrap(), 1, "{csv}");
        }
    }

    #[test]
    fn rejections() {
        let s = spec(Market::Um, Kind::Trades, "2025-01-01");
        let limits = Limits::default();
        for bad in [
            "",
            "id,price,qty,quote_qty,time,is_buyer_maker\n",
            "1,2,3,4,1735689600123\n",
            "1,-2,3,4,1735689600123,true\n",
            "1,2,3,4,1735689600123,yes\n",
            "1,2,3,4,1735776000000,true\n",
            "1.5,2,3,4,1735689600123,true\n",
            "18446744073709551616,2,3,4,1735689600123,true\n",
            "1,2,3,4,1735689600123,true\n\n2,2,3,4,1735689600124,true\n",
        ] {
            assert!(count_rows(&s, bad.as_bytes(), &limits).is_err(), "{bad:?}");
        }
        let two = "1,2,3,4,1735689600123,true\n2,2,3,4,1735689600124,true\n";
        let tight = Limits {
            max_rows: 1,
            ..Limits::default()
        };
        assert!(matches!(count_rows(&s, two.as_bytes(), &tight), Err(Error::Limit(_))));
        assert_eq!(count_rows(&s, two.as_bytes(), &limits).unwrap(), 2);
    }
}
