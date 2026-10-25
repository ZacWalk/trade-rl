//! Transcendental functions from `libm`, a pure-Rust port of musl's libm.
//!
//! The platform C libraries disagree in the last bit on some inputs, which
//! would make a run recorded on Windows fail `verify` on Linux. These give
//! the same bits everywhere. `sqrt` is exact in IEEE 754, so std's is fine.

#[inline]
pub fn ln(x: f64) -> f64 {
    libm::log(x)
}

#[inline]
pub fn ln_1p(x: f64) -> f64 {
    libm::log1p(x)
}

#[inline]
pub fn exp(x: f64) -> f64 {
    libm::exp(x)
}

#[inline]
pub fn sin(x: f64) -> f64 {
    libm::sin(x)
}

#[inline]
pub fn cos(x: f64) -> f64 {
    libm::cos(x)
}

#[inline]
pub fn pow(x: f64, y: f64) -> f64 {
    libm::pow(x, y)
}

#[inline]
pub fn clamp(v: f64, lo: f64, hi: f64) -> f64 {
    if v < lo {
        lo
    } else if v > hi {
        hi
    } else {
        v
    }
}

/// Finite and not absurdly large: the bound used for prices.
#[inline]
pub fn finite(v: f64) -> bool {
    v > -1e300 && v < 1e300
}

/// Strictly positive; false for NaN.
#[inline]
pub fn positive(v: f64) -> bool {
    v > 0.0
}

/// Finite and at least 0: the domain of every cost and rate setting.
#[inline]
pub fn nonnegative(v: f64) -> bool {
    v.is_finite() && v >= 0.0
}
