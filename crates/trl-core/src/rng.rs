//! Deterministic randomness. Results must reproduce bit-exactly, so the
//! generator is written out here rather than taken from a crate whose
//! sequence could change between versions.

use crate::math;

const GAMMA: u64 = 0x9e37_79b9_7f4a_7c15;

/// SplitMix64, with the polar Box–Muller method for normals.
#[derive(Clone, Debug)]
pub struct Rng {
    state: u64,
    spare: f64,
    has_spare: bool,
}

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng {
            state: seed.wrapping_add(GAMMA),
            spare: 0.0,
            has_spare: false,
        }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(GAMMA);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)` with 53 bits of resolution.
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
    }

    pub fn uniform_in(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.uniform()
    }

    pub fn normal(&mut self) -> f64 {
        if self.has_spare {
            self.has_spare = false;
            return self.spare;
        }
        let (u, v, s) = loop {
            let u = self.uniform_in(-1.0, 1.0);
            let v = self.uniform_in(-1.0, 1.0);
            let s = u * u + v * v;
            if s < 1.0 && s != 0.0 {
                break (u, v, s);
            }
        };
        let f = (-2.0 * math::ln(s) / s).sqrt();
        self.spare = v * f;
        self.has_spare = true;
        u * f
    }

    /// Uniform index in `0..n`; 0 when `n` is 0.
    pub fn index(&mut self, n: usize) -> usize {
        if n == 0 {
            0
        } else {
            (self.next_u64() % n as u64) as usize
        }
    }

    /// Fisher–Yates shuffle, drawing from the back.
    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            let j = self.index(i + 1);
            items.swap(i, j);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splitmix_reference_values() {
        // SplitMix64 seeded with 0 after the constructor's extra gamma step:
        // the reference sequence for state 0x9e3779b97f4a7c15.
        let mut r = Rng::new(0);
        let first = r.next_u64();
        let mut reference = Rng {
            state: GAMMA,
            spare: 0.0,
            has_spare: false,
        };
        assert_eq!(first, reference.next_u64());
        // The canonical SplitMix64 output for an initial state of 0 is
        // 0xe220a8397b1dcdaf; ours starts one gamma later.
        let mut canonical = Rng {
            state: 0,
            spare: 0.0,
            has_spare: false,
        };
        assert_eq!(canonical.next_u64(), 0xe220_a839_7b1d_cdaf);
    }

    #[test]
    fn uniform_and_normal_are_sane() {
        let mut r = Rng::new(42);
        let n = 100_000;
        let (mut sum, mut sq) = (0.0, 0.0);
        for _ in 0..n {
            let u = r.uniform();
            assert!((0.0..1.0).contains(&u));
            let z = r.normal();
            sum += z;
            sq += z * z;
        }
        let mean = sum / n as f64;
        let var = sq / n as f64 - mean * mean;
        assert!(mean.abs() < 0.02, "{mean}");
        assert!((var - 1.0).abs() < 0.03, "{var}");
    }

    #[test]
    fn index_bounds() {
        let mut r = Rng::new(7);
        assert_eq!(r.index(0), 0);
        for _ in 0..1000 {
            assert!(r.index(13) < 13);
        }
    }
}
