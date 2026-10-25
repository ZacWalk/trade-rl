//! Per-bar trade logs and the metrics computed from them.

use crate::math;

/// What the agent held at each scored bar and what it earned.
#[derive(Clone, Debug, Default)]
pub struct TradeLog {
    pub time: Vec<i64>,
    pub price: Vec<f64>,
    pub position: Vec<i32>,
    /// Log return earned this bar, before costs.
    pub gross: Vec<f64>,
    pub cost: Vec<f64>,
    /// Cumulative net log return.
    pub equity: Vec<f64>,
    pub elapsed_ms: Vec<i64>,
    pub market_return: Vec<f64>,
    /// Position change charged, in cost-scaled units.
    pub units: Vec<f64>,
}

impl TradeLog {
    pub fn clear(&mut self) {
        *self = TradeLog::default();
    }

    #[allow(clippy::too_many_arguments)]
    pub fn add(
        &mut self,
        time: i64,
        price: f64,
        position: i32,
        gross: f64,
        cost: f64,
        elapsed_ms: i64,
        market_return: f64,
        units: f64,
    ) {
        let previous = self.equity.last().copied().unwrap_or(0.0);
        self.time.push(time);
        self.price.push(price);
        self.position.push(position);
        self.gross.push(gross);
        self.cost.push(cost);
        self.equity.push(previous + gross - cost);
        self.elapsed_ms.push(elapsed_ms);
        self.market_return.push(market_return);
        self.units.push(units);
    }

    pub fn len(&self) -> usize {
        self.time.len()
    }

    pub fn is_empty(&self) -> bool {
        self.time.is_empty()
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Metrics {
    pub bars: i64,
    pub trades: i32,
    /// Cumulative net log return.
    pub net_return: f64,
    pub gross_return: f64,
    pub cost_paid: f64,
    /// Costs as a share of gross, when gross is positive.
    pub cost_fraction: f64,
    pub annual_return: f64,
    pub sharpe: f64,
    pub sortino: f64,
    pub max_drawdown: f64,
    /// Mean |position change| per bar.
    pub turnover: f64,
    pub hit_rate: f64,
    pub average_win: f64,
    pub average_loss: f64,
    pub time_long: f64,
    pub time_short: f64,
    pub time_flat: f64,
    /// Console only; metrics.csv keeps its columns.
    pub cost_units: f64,
    /// Summed underlying log return over the scored bars.
    pub market_return: f64,
}

const YEAR_MS: f64 = 365.25 * 24.0 * 60.0 * 60.0 * 1000.0;

/// Metrics of a log. Returns are annualised by the elapsed time the scored
/// bars span when it is recorded, else by the bar interval. A trade is a run
/// of one non-zero position.
pub fn compute_metrics(log: &TradeLog, interval_ms: i64) -> Metrics {
    let mut m = Metrics {
        bars: log.len() as i64,
        ..Metrics::default()
    };
    if log.is_empty() {
        return m;
    }
    let (mut sum, mut sum_sq, mut downside_sq) = (0.0, 0.0, 0.0);
    let mut downside_count = 0i64;
    let (mut long_bars, mut short_bars) = (0i64, 0i64);
    let mut turnover_total = 0.0;
    let mut previous_position = 0i32;
    let mut trade_pnl = 0.0;
    let (mut wins, mut losses) = (0.0, 0.0);
    let (mut win_count, mut loss_count) = (0i32, 0i32);

    let mut close_trade = |previous_position: i32, trade_pnl: &mut f64, trades: &mut i32| {
        if previous_position == 0 {
            return;
        }
        *trades += 1;
        if *trade_pnl >= 0.0 {
            wins += *trade_pnl;
            win_count += 1;
        } else {
            losses += -*trade_pnl;
            loss_count += 1;
        }
        *trade_pnl = 0.0;
    };

    for i in 0..log.len() {
        let net = log.gross[i] - log.cost[i];
        sum += net;
        sum_sq += net * net;
        if net < 0.0 {
            downside_sq += net * net;
            downside_count += 1;
        }
        m.gross_return += log.gross[i];
        m.cost_paid += log.cost[i];
        m.cost_units += log.units[i];
        m.market_return += log.market_return[i];

        let position = log.position[i];
        if position > 0 {
            long_bars += 1;
        } else if position < 0 {
            short_bars += 1;
        }
        turnover_total += f64::from((position - previous_position).abs());
        if position != previous_position {
            close_trade(previous_position, &mut trade_pnl, &mut m.trades);
        }
        previous_position = position;
        trade_pnl += net;
    }
    close_trade(previous_position, &mut trade_pnl, &mut m.trades);

    let bars = m.bars as f64;
    m.net_return = *log.equity.last().unwrap();
    m.cost_fraction = if m.gross_return > 0.0 {
        m.cost_paid / m.gross_return
    } else {
        0.0
    };
    m.turnover = turnover_total / bars;
    m.time_long = long_bars as f64 / bars;
    m.time_short = short_bars as f64 / bars;
    m.time_flat = 1.0 - m.time_long - m.time_short;

    let mean = sum / bars;
    let variance = if m.bars > 1 {
        (sum_sq - bars * mean * mean) / (bars - 1.0)
    } else {
        0.0
    };
    let sd = if variance > 0.0 { variance.sqrt() } else { 0.0 };

    let elapsed: f64 = log.elapsed_ms.iter().map(|&d| d as f64).sum();
    let bars_per_year = if elapsed > 0.0 {
        YEAR_MS * bars / elapsed
    } else if interval_ms > 0 {
        YEAR_MS / interval_ms as f64
    } else {
        0.0
    };
    m.annual_return = mean * bars_per_year;
    m.sharpe = if sd > 1e-15 {
        mean / sd * bars_per_year.sqrt()
    } else {
        0.0
    };
    let downside_sd = if downside_count > 0 {
        (downside_sq / bars).sqrt()
    } else {
        0.0
    };
    m.sortino = if downside_sd > 1e-15 {
        mean / downside_sd * bars_per_year.sqrt()
    } else {
        0.0
    };

    let mut peak = log.equity[0];
    for &e in &log.equity {
        peak = peak.max(e);
        m.max_drawdown = m.max_drawdown.max(peak - e);
    }

    let closed = win_count + loss_count;
    m.hit_rate = if closed > 0 {
        f64::from(win_count) / f64::from(closed)
    } else {
        0.0
    };
    m.average_win = if win_count > 0 {
        wins / f64::from(win_count)
    } else {
        0.0
    };
    m.average_loss = if loss_count > 0 {
        losses / f64::from(loss_count)
    } else {
        0.0
    };
    m
}

/// The round-trip cost in bps at which net would be zero, for the same
/// one-way slippage. `None` when no nonnegative cost pays: gross not
/// positive, or nothing traded.
pub fn breakeven_cost_bps(m: &Metrics, slippage_bps: f64) -> Option<f64> {
    if !crate::math::positive(m.gross_return) || !crate::math::positive(m.cost_units) {
        return None;
    }
    // Costs are linear in the units charged, so gross / units is the one-way
    // cost that consumes the whole gross; slippage takes its share first.
    let round_trip = 2.0 * 10_000.0 * m.gross_return / m.cost_units - 2.0 * slippage_bps;
    (round_trip > 0.0).then_some(round_trip)
}

pub const METRICS_HEADER: &str = "bars,trades,net_return,gross_return,cost_paid,cost_fraction,\
annual_return,sharpe,sortino,max_drawdown,turnover,hit_rate,\
average_win,average_loss,time_long,time_short,time_flat";

pub fn metrics_row(m: &Metrics) -> String {
    format!(
        "{},{},{:.8},{:.8},{:.8},{:.4},{:.6},{:.4},{:.4},{:.8},{:.6},{:.4},{:.8},{:.8},{:.4},{:.4},{:.4}",
        m.bars,
        m.trades,
        m.net_return,
        m.gross_return,
        m.cost_paid,
        m.cost_fraction,
        m.annual_return,
        m.sharpe,
        m.sortino,
        m.max_drawdown,
        m.turnover,
        m.hit_rate,
        m.average_win,
        m.average_loss,
        m.time_long,
        m.time_short,
        m.time_flat
    )
}

/// `exp(x) - 1` as a percentage: a log return read as simple return.
pub fn simple_percent(log_return: f64) -> f64 {
    (math::exp(log_return) - 1.0) * 100.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trades_runs_and_drawdown() {
        let mut log = TradeLog::default();
        // long 2 bars (+1, -0.5), flat, short 1 bar (+0.25), then long losing.
        let rows = [
            (1, 1.0, 0.1),
            (1, -0.5, 0.0),
            (0, 0.0, 0.1),
            (-1, 0.25, 0.1),
            (1, -1.0, 0.2),
        ];
        for (i, (p, g, c)) in rows.into_iter().enumerate() {
            log.add(i as i64 * 1000, 100.0, p, g, c, 1000, 0.0, c / 0.1);
        }
        let m = compute_metrics(&log, 1000);
        assert_eq!(m.bars, 5);
        assert_eq!(m.trades, 3);
        assert!((m.net_return - (1.0 - 0.5 + 0.25 - 1.0 - 0.5)).abs() < 1e-12);
        // Equity 0.9, 0.4, 0.3, 0.45, -0.75: the peak-to-trough fall is 1.65.
        assert!((m.max_drawdown - 1.65).abs() < 1e-12, "{}", m.max_drawdown);
        assert_eq!(m.time_long, 0.6);
        assert_eq!(m.time_short, 0.2);
        assert!((m.turnover - 5.0 / 5.0).abs() < 1e-12);
        assert!((m.hit_rate - 2.0 / 3.0).abs() < 1e-12);
        assert!(breakeven_cost_bps(&m, 0.0).is_none());
        assert_eq!(metrics_row(&m).split(',').count(), METRICS_HEADER.split(',').count());
    }

    #[test]
    fn breakeven_is_linear_in_units() {
        let m = Metrics {
            gross_return: 0.01,
            cost_units: 10.0,
            ..Metrics::default()
        };
        assert!((breakeven_cost_bps(&m, 0.0).unwrap() - 20.0).abs() < 1e-9);
        assert!((breakeven_cost_bps(&m, 1.0).unwrap() - 18.0).abs() < 1e-9);
        assert!(breakeven_cost_bps(&m, 10.0).is_none());
    }
}
