//! The iced shell: two text lines above the chart, a footer below, and the
//! raster chart in a canvas that turns mouse and keys into view changes.

use std::cell::RefCell;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};

use iced::widget::image::{FilterMethod, Handle};
use iced::widget::{canvas, column, container, text};
use iced::{Color, Element, Font, Length, Point, Rectangle, Renderer, Size, Subscription, Task, Theme};
use iced::{keyboard, mouse, window};

use crate::model::{Model, Options};
use crate::render::{MinimapCache, layout, render};

const HEAD: Color = Color::from_rgb8(0xd8, 0xde, 0xe4);
const BODY: Color = Color::from_rgb8(0x9a, 0xa4, 0xae);
const FOOT: Color = Color::from_rgb8(0x6a, 0x74, 0x7e);
const WARN: Color = Color::from_rgb8(0xe0, 0x82, 0x52);
const BACKGROUND: Color = Color::from_rgb8(0x10, 0x14, 0x18);

/// Set when a smoke test captured its screenshot.
pub static SMOKE_PASSED: AtomicBool = AtomicBool::new(false);

/// iced uploads RGBA images below 2 MiB synchronously and larger ones on a
/// worker thread, drawing nothing until they arrive; tiles keep the chart
/// on the synchronous path so it never blinks while panning.
const MAX_TILE_BYTES: usize = 2 * 1024 * 1024 - 1;

/// What a rendered chart depends on; the tiles are reused until it changes.
#[derive(Clone, Copy, PartialEq)]
struct TileKey {
    width: u32,
    height: u32,
    start: i64,
    span: i64,
    hover: Option<(i32, i32)>,
    has_run: bool,
    scale: u32,
}

/// A horizontal strip of the chart: its image, first row and row count.
type Tile = (Handle, u32, u32);

pub struct App {
    model: Model,
    cache: RefCell<MinimapCache>,
    tiles: RefCell<Option<(TileKey, Vec<Tile>)>>,
    /// Hovered point in physical pixels.
    hover: Option<Point>,
    /// Chart width in physical pixels at the last interaction.
    width: f32,
    /// Physical pixels per logical pixel.
    scale: f32,
    smoke: Option<PathBuf>,
    frames: u32,
    capturing: bool,
}

#[derive(Debug, Clone)]
pub enum Message {
    Hover(Option<Point>, f32),
    SetStart(i64, f32),
    Zoom { anchor: f64, zoom_in: bool, width: f32 },
    Scroll { back: bool, width: f32 },
    Key(keyboard::key::Named, f32),
    Opened(window::Id),
    Scale(f32),
    Frame,
    Capture(window::Screenshot),
}

pub fn run(options: Options, smoke: Option<PathBuf>) -> iced::Result {
    iced::application(
        move || App {
            model: Model::load(&options),
            cache: RefCell::new(MinimapCache::default()),
            tiles: RefCell::new(None),
            hover: None,
            width: 0.0,
            scale: 1.0,
            smoke: smoke.clone(),
            frames: 0,
            capturing: false,
        },
        App::update,
        App::view,
    )
    .title(|_: &App| "trade-rl viewer".to_string())
    .theme(|_: &App| Theme::Dark)
    .subscription(App::subscription)
    .window(window::Settings {
        size: Size::new(1400.0, 900.0),
        min_size: Some(Size::new(640.0, 480.0)),
        ..window::Settings::default()
    })
    .run()
}

impl App {
    fn update(&mut self, message: Message) -> Task<Message> {
        match message {
            Message::Hover(at, width) => {
                self.hover = at;
                self.width = width;
            }
            Message::SetStart(start, width) => self.model.set_start(start, width as i32),
            Message::Zoom { anchor, zoom_in, width } => self.model.zoom(anchor, zoom_in, width as i32),
            Message::Scroll { back, width } => self.model.scroll(back, width as i32),
            Message::Key(key, width) => {
                let width = width as i32;
                match key {
                    keyboard::key::Named::Home => self.model.home(width),
                    keyboard::key::Named::End => self.model.end(width),
                    keyboard::key::Named::ArrowLeft => self.model.scroll(true, width),
                    keyboard::key::Named::ArrowRight => self.model.scroll(false, width),
                    _ => {}
                }
            }
            Message::Opened(id) => return window::scale_factor(id).map(Message::Scale),
            Message::Scale(scale) => {
                if scale.is_finite() && scale > 0.0 {
                    self.scale = scale;
                }
            }
            Message::Frame => {
                self.frames += 1;
                if self.smoke.is_some() && !self.capturing && self.frames >= 5 {
                    self.capturing = true;
                    return window::oldest().and_then(window::screenshot).map(Message::Capture);
                }
                if self.frames > 600 {
                    eprintln!("smoke test: no screenshot after 600 frames");
                    return iced::exit();
                }
            }
            Message::Capture(shot) => {
                if let Some(path) = &self.smoke {
                    match crate::save_png(path, shot.size.width, shot.size.height, shot.rgba.to_vec()) {
                        Ok(()) if self.model.loaded() => {
                            println!(
                                "UI smoke passed: {}x{}  {}  screenshot {}",
                                shot.size.width,
                                shot.size.height,
                                self.model.status,
                                path.display()
                            );
                            SMOKE_PASSED.store(true, Ordering::Relaxed);
                        }
                        Ok(()) => eprintln!("UI smoke failed: {}", self.model.status),
                        Err(e) => eprintln!("UI smoke failed: {e}"),
                    }
                }
                return iced::exit();
            }
        }
        Task::none()
    }

    fn subscription(&self) -> Subscription<Message> {
        let scale = window::events().filter_map(|(_, event)| match event {
            window::Event::Rescaled(scale) => Some(Message::Scale(scale)),
            _ => None,
        });
        let mut subscriptions = vec![window::open_events().map(Message::Opened), scale];
        if self.smoke.is_some() {
            subscriptions.push(window::frames().map(|_| Message::Frame));
        }
        Subscription::batch(subscriptions)
    }

    fn view(&self) -> Element<'_, Message> {
        let mono = Font::MONOSPACE;
        let body: Element<'_, Message> = if self.model.loaded() {
            let hovered = self.model.hovered(self.hover.map(|p| p.x), self.width as i32);
            let (footer, warn) = self.model.footer();
            column![
                column![
                    text(self.model.header()).font(mono).size(14).color(HEAD),
                    text(self.model.readout(hovered)).font(mono).size(14).color(BODY),
                ]
                .spacing(2)
                .padding([6, 10]),
                canvas(Chart { app: self }).width(Length::Fill).height(Length::Fill),
                container(text(footer).font(mono).size(12).color(if warn { WARN } else { FOOT })).padding([2, 10]),
            ]
            .into()
        } else {
            column![
                text("trade-rl viewer").font(mono).size(16).color(HEAD),
                text(&self.model.status).font(mono).size(14).color(WARN),
            ]
            .spacing(8)
            .padding(16)
            .into()
        };
        container(body)
            .width(Length::Fill)
            .height(Length::Fill)
            .style(|_| container::Style {
                background: Some(BACKGROUND.into()),
                ..container::Style::default()
            })
            .into()
    }
}

struct Chart<'a> {
    app: &'a App,
}

#[derive(Default)]
struct ChartState {
    drag: Option<Drag>,
    shift: bool,
}

#[derive(Clone, Copy)]
struct Drag {
    minimap: bool,
    x: f32,
    start: i64,
}

impl canvas::Program<Message> for Chart<'_> {
    type State = ChartState;

    fn update(
        &self,
        state: &mut ChartState,
        event: &canvas::Event,
        bounds: Rectangle,
        cursor: mouse::Cursor,
    ) -> Option<canvas::Action<Message>> {
        let model = &self.app.model;
        // Everything the model sees is in physical pixels, like the raster.
        let scale = self.app.scale;
        let width = bounds.width * scale;
        let at = cursor.position_in(bounds).map(|p| Point::new(p.x * scale, p.y * scale));
        match event {
            canvas::Event::Keyboard(keyboard::Event::ModifiersChanged(m)) => {
                state.shift = m.shift();
                None
            }
            canvas::Event::Keyboard(keyboard::Event::KeyPressed {
                key: keyboard::Key::Named(named),
                ..
            }) => {
                use keyboard::key::Named::*;
                matches!(named, Home | End | ArrowLeft | ArrowRight)
                    .then(|| canvas::Action::publish(Message::Key(*named, width)).and_capture())
            }
            canvas::Event::Mouse(mouse::Event::CursorMoved { .. }) => {
                if let Some(drag) = state.drag {
                    // Dragging continues outside the chart, so use the raw position.
                    let x = (cursor.position()?.x - bounds.x) * scale;
                    let moved = if drag.minimap {
                        f64::from(x - drag.x) * model.total() as f64 / f64::from(width)
                    } else {
                        f64::from(drag.x - x) * model.span as f64 / f64::from(width)
                    };
                    return Some(canvas::Action::publish(Message::SetStart(
                        drag.start + moved as i64,
                        width,
                    )));
                }
                Some(canvas::Action::publish(Message::Hover(at, width)))
            }
            canvas::Event::Mouse(mouse::Event::CursorLeft) => state
                .drag
                .is_none()
                .then(|| canvas::Action::publish(Message::Hover(None, width))),
            canvas::Event::Mouse(mouse::Event::ButtonPressed(mouse::Button::Left)) => {
                let p = at?;
                let height = (bounds.height * scale) as i32;
                let minimap = layout(model, height, scale).map.contains(p.y as i32);
                let start = if minimap {
                    model.seek_start(p.x, width as i32)
                } else {
                    model.start
                };
                state.drag = Some(Drag { minimap, x: p.x, start });
                Some(canvas::Action::publish(Message::SetStart(start, width)).and_capture())
            }
            canvas::Event::Mouse(mouse::Event::ButtonReleased(mouse::Button::Left)) => {
                state.drag.take()?;
                // Released outside: no CursorLeft follows a captured drag.
                at.is_none()
                    .then(|| canvas::Action::publish(Message::Hover(None, width)))
            }
            canvas::Event::Mouse(mouse::Event::WheelScrolled { delta }) => {
                let p = at?;
                let y = match delta {
                    mouse::ScrollDelta::Lines { y, .. } | mouse::ScrollDelta::Pixels { y, .. } => *y,
                };
                if y == 0.0 {
                    return None;
                }
                let message = if state.shift {
                    Message::Scroll { back: y > 0.0, width }
                } else {
                    Message::Zoom {
                        anchor: f64::from(p.x / width),
                        zoom_in: y > 0.0,
                        width,
                    }
                };
                Some(canvas::Action::publish(message).and_capture())
            }
            _ => None,
        }
    }

    fn draw(
        &self,
        _state: &ChartState,
        renderer: &Renderer,
        _theme: &Theme,
        bounds: Rectangle,
        _cursor: mouse::Cursor,
    ) -> Vec<canvas::Geometry> {
        let app = self.app;
        let scale = app.scale;
        let w = (bounds.width * scale).round().max(1.0) as u32;
        let h = (bounds.height * scale).round().max(1.0) as u32;
        let hover = app.hover.map(|p| (p.x as i32, p.y as i32));
        let key = TileKey {
            width: w,
            height: h,
            start: app.model.start,
            span: app.model.span,
            hover,
            has_run: app.model.has_run,
            scale: scale.to_bits(),
        };
        let mut tiles = app.tiles.borrow_mut();
        if tiles.as_ref().is_none_or(|(k, _)| *k != key) {
            let pixels = render(
                &app.model,
                &mut app.cache.borrow_mut(),
                w as i32,
                h as i32,
                hover,
                scale,
            );
            let stride = w as usize * 4;
            let rows = (MAX_TILE_BYTES / stride).max(1);
            let strips = pixels
                .chunks(rows * stride)
                .enumerate()
                .map(|(i, chunk)| {
                    let n = (chunk.len() / stride) as u32;
                    (Handle::from_rgba(w, n, chunk.to_vec()), (i * rows) as u32, n)
                })
                .collect();
            *tiles = Some((key, strips));
        }
        let mut frame = canvas::Frame::new(renderer, bounds.size());
        for (handle, y, n) in tiles.as_ref().map(|(_, t)| t.as_slice()).unwrap_or_default() {
            let mut image = canvas::Image::new(handle.clone()).filter_method(FilterMethod::Nearest);
            image.snap = true;
            let at = Point::new(0.0, *y as f32 / scale);
            frame.draw_image(
                Rectangle::new(at, Size::new(w as f32 / scale, *n as f32 / scale)),
                image,
            );
        }
        vec![frame.into_geometry()]
    }

    fn mouse_interaction(&self, state: &ChartState, bounds: Rectangle, cursor: mouse::Cursor) -> mouse::Interaction {
        if state.drag.is_some() {
            mouse::Interaction::ResizingHorizontally
        } else if cursor.is_over(bounds) {
            mouse::Interaction::Crosshair
        } else {
            mouse::Interaction::default()
        }
    }
}
