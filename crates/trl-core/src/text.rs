//! Small text helpers shared by importers, manifests and reports.

/// Trim ASCII whitespace: space, tab, CR and LF.
pub fn trim(s: &str) -> &str {
    s.trim_matches([' ', '\t', '\r', '\n'])
}

/// Parse a whole token as a number: `"0.5x"` is a typo, not 0.5.
pub fn parse_real(s: &str) -> Option<f64> {
    let t = trim(s);
    if t.is_empty() {
        return None;
    }
    t.parse::<f64>().ok()
}

/// Parse a whole token as a finite number.
pub fn parse_finite(s: &str) -> Option<f64> {
    parse_real(s).filter(|v| v.is_finite())
}

pub fn hex64(v: u64) -> String {
    format!("{v:016x}")
}

pub fn parse_hex64(s: &str) -> Option<u64> {
    let t = trim(s);
    if t.len() != 16 {
        return None;
    }
    u64::from_str_radix(t, 16).ok()
}

/// FNV-1a, 64-bit: the fingerprint recorded with every run.
#[derive(Clone, Copy, Debug)]
pub struct Fnv(u64);

impl Default for Fnv {
    fn default() -> Fnv {
        Fnv(0xcbf2_9ce4_8422_2325)
    }
}

impl Fnv {
    pub fn new() -> Fnv {
        Fnv::default()
    }

    pub fn feed(&mut self, bytes: &[u8]) {
        let mut h = self.0;
        for &b in bytes {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        self.0 = h;
    }

    pub fn digest(&self) -> u64 {
        self.0
    }
}

/// Shortest text that reads back as the same `f64`, the form every number
/// that must round-trip through an artifact is written in.
pub fn exact(v: f64) -> String {
    if v == 0.0 {
        // -0 and 0 print alike; the sign of a zero carries no result.
        return "0".into();
    }
    format!("{v:?}")
}

/// Fixed decimals, like printf's `%.Nf`.
pub fn fixed(v: f64, decimals: usize) -> String {
    format!("{v:.decimals$}")
}

/// Format `v` like C's `%.Ng`: `precision` significant digits, trailing
/// zeros removed, scientific notation outside 1e-4..1e`precision`.
pub fn general(v: f64, precision: usize) -> String {
    if !v.is_finite() {
        return if v.is_nan() {
            "nan".into()
        } else if v > 0.0 {
            "inf".into()
        } else {
            "-inf".into()
        };
    }
    if v == 0.0 {
        return if v.is_sign_negative() { "-0".into() } else { "0".into() };
    }
    let p = precision.max(1);
    let sci = format!("{:.*e}", p - 1, v);
    let (mantissa, exponent) = sci.split_once('e').unwrap_or((&sci, "0"));
    let x: i32 = exponent.parse().unwrap_or(0);
    if x < -4 || x >= p as i32 {
        let m = strip_zeros(mantissa);
        let sign = if x < 0 { '-' } else { '+' };
        format!("{m}e{sign}{:02}", x.unsigned_abs())
    } else {
        let decimals = (p as i32 - 1 - x).max(0) as usize;
        strip_zeros(&format!("{v:.decimals$}")).to_string()
    }
}

fn strip_zeros(s: &str) -> &str {
    if s.contains('.') {
        s.trim_end_matches('0').trim_end_matches('.')
    } else {
        s
    }
}

/// Text safe to store in a manifest or catalog: printable, single line, bounded.
pub fn valid_label(text: &str, limit: usize) -> bool {
    !text.is_empty() && text.len() <= limit && text.bytes().all(|c| (32..127).contains(&c) || c >= 128)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn general_matches_printf() {
        assert_eq!(general(0.0001, 6), "0.0001");
        assert_eq!(general(0.00001, 6), "1e-05");
        assert_eq!(general(123456.0, 6), "123456");
        assert_eq!(general(1234567.0, 6), "1.23457e+06");
        assert_eq!(general(1.5, 17), "1.5");
        assert_eq!(general(0.1, 17), "0.10000000000000001");
        assert_eq!(general(-2.5e-7, 3), "-2.5e-07");
        assert_eq!(general(100.0, 3), "100");
        assert_eq!(general(999.5, 3), "1e+03");
    }

    #[test]
    fn exact_round_trips() {
        for v in [0.1, 1.0 / 3.0, -1e-300, 12345.678, 1e22] {
            assert_eq!(exact(v).parse::<f64>().unwrap(), v);
        }
        assert_eq!(exact(-0.0), "0");
    }

    #[test]
    fn fnv_reference() {
        let mut h = Fnv::new();
        h.feed(b"a");
        assert_eq!(h.digest(), 0xaf63_dc4c_8601_ec8c);
    }

    #[test]
    fn parse_real_whole_token() {
        assert_eq!(parse_real(" 2.5 "), Some(2.5));
        assert_eq!(parse_real("0.5x"), None);
        assert_eq!(parse_real(""), None);
        assert_eq!(parse_finite("inf"), None);
    }
}
