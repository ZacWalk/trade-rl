//! The chart as pixels. Every screen column aggregates the bars that fall in
//! it — range, close, majority position and last equity — so a million bars
//! draw as fast as a thousand. Pure Rust and deterministic, so it renders
//! headless (`trade-rl --render`) and in tests.

use crate::model::{Model, NO_POSITION};

/// `0xRRGGBB` as little-endian RGBA, the byte order images are uploaded in.
const fn rgb(c: u32) -> u32 {
    ((c >> 16) & 0xff) | (c & 0xff00) | ((c & 0xff) << 16) | 0xff00_0000
}

pub const BACKGROUND: u32 = rgb(0x101418);
pub const GRID: u32 = rgb(0x202830);
pub const PRICE: u32 = rgb(0xd8dee4);
pub const LONG: u32 = rgb(0x38c172);
pub const SHORT: u32 = rgb(0xe05252);
pub const FLAT: u32 = rgb(0x404850);
pub const EQUITY: u32 = rgb(0x46b8dd);
pub const DRAWDOWN: u32 = rgb(0x303a44);
pub const CURSOR: u32 = rgb(0x606c78);
const MAP_BACKGROUND: u32 = rgb(0x161b20);
const MAP_PRICE: u32 = rgb(0x4e5860);
const MAP_SHADE: u32 = rgb(0x0c0f12);
const MAP_THUMB: u32 = rgb(0x1f2933);
const MAP_EDGE: u32 = rgb(0x7d8b99);

/// Logical sizes; the chart's text lives outside the raster, so only a
/// hairline of padding above and below.
const HEADER: i32 = 2;
const FOOTER: i32 = 2;
const BAND: i32 = 12;
const MAP: i32 = 52;
const GAP: i32 = 6;
const INSET: i32 = 3;

/// A logical size in physical pixels.
fn scaled(value: i32, scale: f32) -> i32 {
    (value as f32 * scale + 0.5) as i32
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Pane {
    pub top: i32,
    pub height: i32,
}

impl Pane {
    pub fn bottom(&self) -> i32 {
        self.top + self.height
    }

    pub fn contains(&self, y: i32) -> bool {
        y >= self.top && y < self.bottom()
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Panes {
    pub price: Pane,
    pub band: Pane,
    pub equity: Pane,
    pub map: Pane,
}

/// A short window gives up the minimap before the chart it summarises, and
/// a pane with no room is hidden rather than drawn over its neighbours.
pub fn layout(model: &Model, height: i32, scale: f32) -> Panes {
    let [header, footer, band_size, map_size, gap] = [HEADER, FOOTER, BAND, MAP, GAP].map(|v| scaled(v, scale));
    let chart = height - header - footer - band_size - gap * 2;
    let map_height = if model.loaded() && chart - map_size >= map_size {
        map_size
    } else {
        0
    };
    let map = Pane {
        top: height - footer - map_height,
        height: map_height,
    };
    let usable = (map.top - header - band_size - gap * 2).max(0);
    let split = model.has_run && usable > 0;
    let price = Pane {
        top: header,
        height: if split { usable * 62 / 100 } else { usable },
    };
    let band = Pane {
        top: price.bottom() + scaled(2, scale),
        height: if split { band_size } else { 0 },
    };
    let equity = Pane {
        top: band.bottom() + gap,
        height: if split { usable - price.height } else { 0 },
    };
    Panes {
        price,
        band,
        equity,
        map,
    }
}

/// The whole series reduced to one column per pixel: rebuilt only when the
/// width changes, so dragging the minimap never walks every bar.
#[derive(Clone, Debug, Default)]
pub struct MinimapCache {
    width: i32,
    has_run: bool,
    low: Vec<f64>,
    high: Vec<f64>,
    side: Vec<i8>,
}

impl MinimapCache {
    fn build(&mut self, model: &Model, width: i32) {
        if (self.width == width && self.has_run == model.has_run && width > 0) || !model.loaded() {
            return;
        }
        let Some(series) = &model.series else {
            return;
        };
        let w = width.max(0) as usize;
        let total = series.len();
        self.width = width;
        self.has_run = model.has_run;
        self.low = vec![f64::NAN; w];
        self.high = vec![f64::NAN; w];
        self.side = vec![NO_POSITION; w];
        for x in 0..w {
            let from = total * x / w;
            let to = (from + 1).max(total * (x + 1) / w).min(total);
            let (mut low, mut high, mut priced) = (f64::INFINITY, f64::NEG_INFINITY, false);
            let mut votes = [0usize; 3];
            for i in from..to {
                let b = &series.bars[i];
                if b.low.is_finite() && b.high.is_finite() {
                    low = low.min(b.low);
                    high = high.max(b.high);
                    priced = true;
                }
                if model.has_run && model.position[i] != NO_POSITION {
                    votes[(model.position[i] + 1) as usize] += 1;
                }
            }
            if priced {
                self.low[x] = low;
                self.high[x] = high;
            }
            self.side[x] = majority(votes);
        }
    }
}

/// Long beats ties with flat and short, short beats ties with flat.
fn majority([shorts, flats, longs]: [usize; 3]) -> i8 {
    if shorts + flats + longs == 0 {
        NO_POSITION
    } else if longs >= shorts && longs >= flats {
        1
    } else if shorts >= flats {
        -1
    } else {
        0
    }
}

struct Surface {
    px: Vec<u32>,
    w: i32,
    h: i32,
}

impl Surface {
    fn dot(&mut self, x: i32, y: i32, c: u32) {
        if x >= 0 && y >= 0 && x < self.w && y < self.h {
            self.px[(y * self.w + x) as usize] = c;
        }
    }

    /// Clipped before the loop, so an off-surface span costs nothing.
    fn column(&mut self, x: i32, y0: i32, y1: i32, c: u32) {
        let (y0, y1) = if y0 > y1 { (y1, y0) } else { (y0, y1) };
        if x < 0 || x >= self.w {
            return;
        }
        for y in y0.max(0)..=y1.min(self.h - 1) {
            self.px[(y * self.w + x) as usize] = c;
        }
    }

    fn fill(&mut self, x0: i32, y0: i32, x1: i32, y1: i32, c: u32) {
        let (x0, x1) = (x0.max(0), x1.min(self.w - 1));
        if x1 < x0 {
            return;
        }
        for y in y0.max(0)..=y1.min(self.h - 1) {
            let row = (y * self.w) as usize;
            self.px[row + x0 as usize..=row + x1 as usize].fill(c);
        }
    }
}

/// A value's row in a pane, clamped into it; NaN lands on the bottom edge.
fn to_y(value: f64, low: f64, high: f64, p: Pane) -> i32 {
    let t = (value - low) / (high - low);
    let t = if t.is_nan() { 0.0 } else { t.clamp(0.0, 1.0) };
    p.bottom() - 1 - (t * f64::from(p.height - 1)) as i32
}

/// Render the view into `width` x `height` physical RGBA pixels at `scale`
/// physical pixels per logical one. `cursor` draws the dotted hover line.
pub fn render(
    model: &Model,
    cache: &mut MinimapCache,
    width: i32,
    height: i32,
    cursor: Option<(i32, i32)>,
    scale: f32,
) -> Vec<u8> {
    let (width, height) = (width.max(1), height.max(1));
    let mut s = Surface {
        px: vec![BACKGROUND; (width * height) as usize],
        w: width,
        h: height,
    };
    if let Some(series) = model.series.as_ref().filter(|_| model.loaded()) {
        let panes = layout(model, height, scale);
        cache.build(model, width);
        draw_chart(&mut s, model, series, panes, cursor);
        draw_minimap(&mut s, model, cache, panes.map, scaled(INSET, scale));
    }
    s.px.iter().flat_map(|c| c.to_le_bytes()).collect()
}

fn draw_chart(s: &mut Surface, model: &Model, series: &trl_core::Series, panes: Panes, cursor: Option<(i32, i32)>) {
    let Panes {
        price, band, equity, ..
    } = panes;
    let width = s.w;
    let first = model.start.max(0) as usize;
    if first >= series.len() {
        return;
    }
    let count = (model.span.max(1) as usize).min(series.len() - first);

    let (mut price_low, mut price_high, mut priced) = (f64::INFINITY, f64::NEG_INFINITY, false);
    let (mut equity_low, mut equity_high) = (f64::INFINITY, f64::NEG_INFINITY);
    for i in first..first + count {
        let b = &series.bars[i];
        if b.low.is_finite() && b.high.is_finite() {
            price_low = price_low.min(b.low);
            price_high = price_high.max(b.high);
            priced = true;
        }
        if model.has_run && !model.equity[i].is_nan() {
            equity_low = equity_low.min(model.equity[i]);
            equity_high = equity_high.max(model.equity[i]);
        }
    }
    if !priced {
        (price_low, price_high) = (0.0, 1.0);
    } else if price_high <= price_low {
        price_high = price_low + 1.0;
    }
    if equity_high < equity_low {
        (equity_low, equity_high) = (-0.001, 0.001);
    } else if equity_high <= equity_low {
        equity_low -= 0.001;
        equity_high += 0.001;
    }

    if price.height > 0 {
        for g in 1..4 {
            let y = price.top + price.height * g / 4;
            for x in (0..width).step_by(3) {
                s.dot(x, y, GRID);
            }
        }
    }

    let (mut previous_price_y, mut previous_equity_y) = (None, None);
    let w = width as usize;
    for x in 0..width {
        // Zoomed in past one bar per pixel, a column still shows its bar;
        // skipping it would stripe the band and the equity fill.
        let xu = x as usize;
        let from = first + count * xu / w;
        let to = (first + count).min((from + 1).max(first + count * (xu + 1) / w));
        let (mut low, mut high, mut column_priced) = (f64::INFINITY, f64::NEG_INFINITY, false);
        let mut close = f64::NAN;
        let mut votes = [0usize; 3];
        let mut last_equity = f64::NAN;
        for i in from..to {
            let b = &series.bars[i];
            if b.low.is_finite() && b.high.is_finite() {
                low = low.min(b.low);
                high = high.max(b.high);
                column_priced = true;
            }
            if b.close.is_finite() {
                close = b.close;
            }
            if !model.has_run || model.position[i] == NO_POSITION {
                continue;
            }
            votes[(model.position[i] + 1) as usize] += 1;
            if !model.equity[i].is_nan() {
                last_equity = model.equity[i];
            }
        }

        if price.height > 0 {
            if column_priced {
                let y_low = to_y(low, price_low, price_high, price);
                let y_high = to_y(high, price_low, price_high, price);
                s.column(x, y_high, y_low, PRICE);
            }
            if !close.is_nan() {
                let y_close = to_y(close, price_low, price_high, price);
                if let Some(previous) = previous_price_y {
                    s.column(x, previous, y_close, PRICE);
                }
                previous_price_y = Some(y_close);
            }
        }
        if !model.has_run || band.height <= 0 {
            continue;
        }
        let colour = match majority(votes) {
            1 => LONG,
            -1 => SHORT,
            0 => FLAT,
            _ => BACKGROUND,
        };
        s.column(x, band.top, band.bottom() - 1, colour);
        if last_equity.is_nan() || equity.height <= 0 {
            continue;
        }
        let y_equity = to_y(last_equity, equity_low, equity_high, equity);
        let y_zero = to_y(0.0, equity_low, equity_high, equity);
        s.column(x, y_equity.min(y_zero), y_equity.max(y_zero), DRAWDOWN);
        if let Some(previous) = previous_equity_y {
            s.column(x, previous, y_equity, EQUITY);
        }
        previous_equity_y = Some(y_equity);
        s.dot(x, y_equity, EQUITY);
    }

    if let Some((cx, cy)) = cursor
        && cx >= 0
        && cx < width
        && !panes.map.contains(cy)
    {
        for y in (price.top..panes.map.top).step_by(2) {
            s.dot(cx, y, CURSOR);
        }
    }
}

fn draw_minimap(s: &mut Surface, model: &Model, cache: &MinimapCache, map: Pane, inset: i32) {
    if map.height <= 0 || cache.low.len() != s.w as usize {
        return;
    }
    let width = s.w;
    let (top, bottom) = (map.top, map.bottom() - 1);
    s.fill(0, top, width - 1, bottom, MAP_BACKGROUND);
    let (mut low, mut high) = (f64::INFINITY, f64::NEG_INFINITY);
    for x in 0..width as usize {
        if !cache.low[x].is_nan() {
            low = low.min(cache.low[x]);
            high = high.max(cache.high[x]);
        }
    }
    if high < low {
        (low, high) = (0.0, 1.0);
    } else if high <= low {
        high = low + 1.0;
    }
    let total = model.total() as f64;
    let thumb_left = (model.start as f64 / total * f64::from(width)) as i32;
    let thumb_right = ((model.start + model.span) as f64 / total * f64::from(width)) as i32;
    // The view reads as a lit window over a dimmed whole.
    s.fill(thumb_left.max(0), top, thumb_right.min(width - 1), bottom, MAP_THUMB);
    let (plot_top, plot_bottom) = (top + inset, bottom - inset);
    let plot_height = f64::from((plot_bottom - plot_top).max(1));
    for x in 0..width {
        let inside = x >= thumb_left && x <= thumb_right;
        if !inside {
            s.dot(x, top, MAP_SHADE);
        }
        let xu = x as usize;
        if cache.low[xu].is_nan() {
            continue;
        }
        let y_high = plot_bottom - ((cache.high[xu] - low) / (high - low) * plot_height) as i32;
        let y_low = plot_bottom - ((cache.low[xu] - low) / (high - low) * plot_height) as i32;
        let colour = match cache.side[xu] {
            1 => LONG,
            -1 => SHORT,
            _ if inside => PRICE,
            _ => MAP_PRICE,
        };
        s.column(x, y_high, y_low, colour);
    }
    s.column(thumb_left.max(0), top, bottom, MAP_EDGE);
    s.column(thumb_right.min(width - 1), top, bottom, MAP_EDGE);
    for x in 0..width {
        s.dot(x, top, GRID);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pixel(raster: &[u8], width: i32, x: i32, y: i32) -> u32 {
        let at = ((y * width + x) * 4) as usize;
        u32::from_le_bytes(raster[at..at + 4].try_into().unwrap())
    }

    fn model(bars: usize, run: bool) -> Model {
        let series = trl_core::synthetic::trend_series(bars);
        let n = series.len();
        Model {
            span: n as i64,
            has_run: run,
            position: if run {
                (0..n).map(|i| if i < n / 2 { 1 } else { -1 }).collect()
            } else {
                Vec::new()
            },
            equity: if run {
                (0..n).map(|i| i as f64 * 1e-4).collect()
            } else {
                Vec::new()
            },
            series: Some(series),
            ..Model::default()
        }
    }

    #[test]
    fn layout_splits_only_with_a_run() {
        let panes = layout(&model(100, false), 600, 1.0);
        assert_eq!(panes.band.height, 0);
        assert_eq!(panes.map.height, MAP);
        let panes = layout(&model(100, true), 600, 1.0);
        assert!(panes.band.height > 0 && panes.equity.height > 0);
        assert!(panes.price.bottom() < panes.band.top && panes.band.bottom() < panes.equity.top);
        assert!(panes.equity.bottom() <= panes.map.top);
        // Too short for a minimap: the chart keeps the room.
        assert_eq!(layout(&model(100, false), 120, 1.0).map.height, 0);
    }

    #[test]
    fn band_shows_the_majority_position() {
        let m = model(2000, true);
        let (w, h) = (400, 500);
        let mut cache = MinimapCache::default();
        let raster = render(&m, &mut cache, w, h, None, 1.0);
        assert_eq!(raster.len(), (w * h * 4) as usize);
        let band = layout(&m, h, 1.0).band;
        assert_eq!(pixel(&raster, w, 10, band.top + 2), LONG);
        assert_eq!(pixel(&raster, w, w - 10, band.top + 2), SHORT);
        // Some price was drawn, and the cursor line appears where asked.
        let price = layout(&m, h, 1.0).price;
        assert!((price.top..price.bottom()).any(|y| pixel(&raster, w, 200, y) == PRICE));
        let with_cursor = render(&m, &mut cache, w, h, Some((123, 50)), 1.0);
        assert_eq!(pixel(&with_cursor, w, 123, price.top), CURSOR);
    }

    #[test]
    fn rendering_is_deterministic_and_handles_tiny_windows() {
        let m = model(5000, true);
        let a = render(&m, &mut MinimapCache::default(), 640, 480, None, 1.0);
        let b = render(&m, &mut MinimapCache::default(), 640, 480, None, 1.0);
        assert_eq!(a, b);
        for (w, h) in [(1, 1), (3, 40), (900, 30)] {
            assert_eq!(
                render(&m, &mut MinimapCache::default(), w, h, Some((0, 0)), 1.5).len(),
                (w * h * 4) as usize
            );
        }
        let empty = render(&Model::default(), &mut MinimapCache::default(), 10, 10, None, 1.0);
        assert!(
            empty
                .as_chunks::<4>()
                .0
                .iter()
                .all(|p| u32::from_le_bytes(*p) == BACKGROUND)
        );
    }
}
