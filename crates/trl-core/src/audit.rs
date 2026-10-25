//! The shift audit: a causal feature recomputed over a series truncated just
//! after bar `t` must give exactly the same value at `t`. Any look-ahead
//! makes the two disagree.

use std::io::{self, Write};

use crate::bars::Series;
use crate::error::{Result, invalid};
use crate::features::build_features;
use crate::text::general;

/// One truncation point.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Probe {
    pub t: usize,
    /// Columns that disagreed or were not finite.
    pub bad: usize,
}

#[derive(Clone, Debug)]
pub struct Audit {
    pub probes: Vec<Probe>,
    pub columns: usize,
    pub worst: f64,
    pub worst_name: Option<&'static str>,
}

impl Audit {
    pub fn passed(&self) -> bool {
        self.probes.iter().all(|p| p.bad == 0)
    }

    /// The report `trl audit` prints.
    pub fn write(&self, out: &mut dyn Write) -> io::Result<()> {
        for p in &self.probes {
            writeln!(out, "  t={:<8} {}", p.t, if p.bad == 0 { "ok" } else { "LOOK-AHEAD" })?;
        }
        writeln!(out, "\n{} probes, {} features", self.probes.len(), self.columns)?;
        write!(out, "largest disagreement {}", general(self.worst, 6))?;
        if let Some(name) = self.worst_name {
            write!(out, " in {name}")?;
        }
        writeln!(out)?;
        if self.passed() {
            writeln!(
                out,
                "shift audit passed: every feature at t depends only on bars up to t"
            )?;
        }
        Ok(())
    }
}

/// Probe eight evenly spaced bars after the warmup.
pub fn shift_audit(series: &Series) -> Result<Audit> {
    let full = build_features(series)?;
    let total = series.len() as i64;
    let warmup = full.warmup as i64;
    let probes: Vec<usize> = (1..=8)
        .map(|fraction| warmup + (total - warmup - 1) * fraction / 9)
        .filter(|&t| t > warmup && t < total)
        .map(|t| t as usize)
        .collect();
    if probes.is_empty() {
        return Err(invalid("not enough bars for a causal shift audit"));
    }
    let mut audit = Audit {
        probes: Vec::new(),
        columns: full.columns,
        worst: 0.0,
        worst_name: None,
    };
    for t in probes {
        let partial = build_features(&series.prefix(t + 1))?;
        let (a, b) = (full.row(t), partial.row(t));
        let mut bad = 0;
        for c in 0..full.columns {
            let difference = (a[c] - b[c]).abs();
            if difference > audit.worst {
                audit.worst = difference;
                audit.worst_name = Some(full.names[c]);
            }
            if !a[c].is_finite() || !b[c].is_finite() || difference > 1e-12 {
                bad += 1;
            }
        }
        audit.probes.push(Probe { t, bad });
    }
    Ok(audit)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synthetic;

    #[test]
    fn features_are_causal() {
        for s in [synthetic::trend_series(4000), synthetic::session_series(4000)] {
            let audit = shift_audit(&s).unwrap();
            assert_eq!(audit.probes.len(), 8);
            assert!(audit.passed(), "{audit:?}");
            assert_eq!(audit.worst, 0.0);
        }
    }

    #[test]
    fn short_series_cannot_be_audited() {
        assert!(shift_audit(&synthetic::trend_series(1000)).is_err());
    }
}
