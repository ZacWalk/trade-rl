//! UTC calendar arithmetic on epoch milliseconds, without a time zone
//! database: bars are stamped in UTC and calendars beyond UTC arrive as
//! explicit session identifiers.

pub const MS_PER_DAY: i64 = 86_400_000;

/// `"1s"`, `"500ms"`, `"5m"`, `"1h"`, `"1d"` or a bare number of
/// milliseconds. `None` when unparseable, zero, or too large to represent.
pub fn parse_duration_ms(s: &str) -> Option<i64> {
    let t = crate::text::trim(s);
    let digits = t.bytes().take_while(u8::is_ascii_digit).count();
    if digits == 0 {
        return None;
    }
    let value: i64 = t[..digits].parse().ok()?;
    let scale = match &t[digits..] {
        "" | "ms" => 1,
        "s" => 1000,
        "m" => 60_000,
        "h" => 3_600_000,
        "d" => MS_PER_DAY,
        _ => return None,
    };
    value.checked_mul(scale).filter(|&ms| ms > 0)
}

/// The inverse of [`parse_duration_ms`], in the largest unit that divides exactly.
pub fn duration_text(ms: i64) -> String {
    for (size, unit) in [(MS_PER_DAY, "d"), (3_600_000, "h"), (60_000, "m"), (1000, "s")] {
        if ms != 0 && ms % size == 0 {
            return format!("{}{unit}", ms / size);
        }
    }
    format!("{ms}ms")
}

/// Days since 1970-01-01 to a proleptic Gregorian `(year, month, day)`.
pub fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let month = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let year = yoe + era * 400 + i64::from(month <= 2);
    (year, month, day)
}

/// A proleptic Gregorian date to days since 1970-01-01.
pub fn days_from_civil(year: i64, month: u32, day: u32) -> i64 {
    let y = if month <= 2 { year - 1 } else { year };
    let era = y.div_euclid(400);
    let yoe = y.rem_euclid(400);
    let m = i64::from(month);
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + i64::from(day) - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

pub fn days_in_month(year: i64, month: u32) -> u32 {
    match month {
        2 if year % 4 == 0 && (year % 100 != 0 || year % 400 == 0) => 29,
        2 => 28,
        4 | 6 | 9 | 11 => 30,
        _ => 31,
    }
}

/// Day of the week with Sunday = 0, as C's `tm_wday`.
pub fn weekday(ms: i64) -> u32 {
    (ms.div_euclid(MS_PER_DAY) + 4).rem_euclid(7) as u32
}

/// Hour of the UTC day, 0..24.
pub fn hour(ms: i64) -> u32 {
    (ms.rem_euclid(MS_PER_DAY) / 3_600_000) as u32
}

/// `YYYY-MM-DD HH:MM:SS`, UTC.
pub fn format_time_ms(ms: i64) -> String {
    let (y, mo, d) = civil_from_days(ms.div_euclid(MS_PER_DAY));
    let s = ms.rem_euclid(MS_PER_DAY) / 1000;
    format!("{y:04}-{mo:02}-{d:02} {:02}:{:02}:{:02}", s / 3600, s / 60 % 60, s % 60)
}

/// `YYYY-MM-DD`, UTC.
pub fn format_date_ms(ms: i64) -> String {
    let (y, mo, d) = civil_from_days(ms.div_euclid(MS_PER_DAY));
    format!("{y:04}-{mo:02}-{d:02}")
}

/// Parse a strict `YYYY-MM-DD` into days since 1970-01-01; years 1970..=9999.
pub fn parse_date(s: &str) -> Option<i64> {
    let b = s.as_bytes();
    if b.len() != 10 || b[4] != b'-' || b[7] != b'-' {
        return None;
    }
    if !b
        .iter()
        .enumerate()
        .all(|(i, c)| i == 4 || i == 7 || c.is_ascii_digit())
    {
        return None;
    }
    let year: i64 = s[0..4].parse().ok()?;
    let month: u32 = s[5..7].parse().ok()?;
    let day: u32 = s[8..10].parse().ok()?;
    if year < 1970 || !(1..=12).contains(&month) || day == 0 || day > days_in_month(year, month) {
        return None;
    }
    Some(days_from_civil(year, month, day))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn durations() {
        assert_eq!(parse_duration_ms("1s"), Some(1000));
        assert_eq!(parse_duration_ms("500ms"), Some(500));
        assert_eq!(parse_duration_ms("15m"), Some(900_000));
        assert_eq!(parse_duration_ms("2h"), Some(7_200_000));
        assert_eq!(parse_duration_ms("1d"), Some(MS_PER_DAY));
        assert_eq!(parse_duration_ms("250"), Some(250));
        for bad in ["", "s", "1x", "0s", "106751991168d", "-1s", "1.5m"] {
            assert_eq!(parse_duration_ms(bad), None, "{bad}");
        }
        for ms in [1, 999, 1000, 60_000, 900_000, 3_600_000, MS_PER_DAY, 90_000] {
            assert_eq!(parse_duration_ms(&duration_text(ms)), Some(ms));
        }
        assert_eq!(duration_text(900_000), "15m");
    }

    #[test]
    fn calendar_round_trip() {
        for days in -800_000..800_000i64 {
            if days % 997 != 0 {
                continue;
            }
            let (y, m, d) = civil_from_days(days);
            assert_eq!(days_from_civil(y, m, d), days);
        }
        assert_eq!(format_time_ms(0), "1970-01-01 00:00:00");
        assert_eq!(format_time_ms(1_577_836_800_000), "2020-01-01 00:00:00");
        assert_eq!(format_time_ms(1_709_210_096_000), "2024-02-29 12:34:56");
        assert_eq!(weekday(0), 4);
        assert_eq!(weekday(1_577_836_800_000), 3);
        assert_eq!(parse_date("2024-02-29"), Some(19_782));
        assert_eq!(parse_date("2023-02-29"), None);
        assert_eq!(parse_date("1969-12-31"), None);
    }
}
