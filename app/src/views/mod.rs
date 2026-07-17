//! gpui UI. One `Root` entity owns view state; the daemon client (api.rs) runs on
//! a background tokio thread. Repaints are wake-driven and fingerprint-gated:
//! api.rs wake() -> channel -> maybe_notify(), which only calls cx.notify() when
//! the on-screen snapshot actually changed. Idle => zero repaints.

pub mod chat;
pub mod library;
pub mod stats;
pub mod system;
pub mod theme;

use crate::api::{self, Client, Cmd};
use gpui::prelude::*;
use gpui::{
    App, Bounds, Context, FocusHandle, ListAlignment, ListState, SharedString, TitlebarOptions,
    Window, WindowBounds, WindowControlArea, WindowOptions, div, point, px, size,
};
use std::hash::{Hash, Hasher};
use std::time::Duration;
use theme::theme;

#[derive(Clone, Copy, PartialEq)]
pub enum Tab {
    Chat,
    Library,
    Stats,
    System,
}

/// Drives the animated ellipsis while the engine prefills (no tokens yet).
pub fn thinking_phase() -> u64 {
    (std::time::UNIX_EPOCH.elapsed().map_or(0, |d| d.as_millis() as u64) / 400) % 4
}

pub struct Root {
    pub client: Client,
    pub tab: Tab,
    pub input: String,
    pub input_focus: FocusHandle,
    pub chat_list: ListState,
    last_fp: u64,
}

impl Root {
    fn new(cx: &mut Context<Self>) -> Self {
        let (tx, mut rx) = futures_channel::mpsc::unbounded::<()>();
        let client = api::start(move || {
            let _ = tx.unbounded_send(());
        });
        // Debug/verification hooks (used by headless checks; harmless otherwise).
        if let Ok(msg) = std::env::var("OVERHANG_AUTOSEND") {
            client.send(Cmd::SendChat(msg));
        }
        if std::env::var("OVERHANG_AUTOEJECT").is_ok() {
            client.send(Cmd::Eject);
        }
        if let Ok(name) = std::env::var("OVERHANG_AUTOLOAD") {
            client.send(Cmd::Load(name));
        }
        let tab = match std::env::var("OVERHANG_TAB").as_deref() {
            Ok("library") => Tab::Library,
            Ok("stats") => Tab::Stats,
            Ok("system") => Tab::System,
            _ => Tab::Chat,
        };

        cx.spawn(async move |this, cx| {
            use futures_util::{StreamExt, future, future::Either};
            loop {
                // Wait for a daemon wake; tick at 250ms only to animate "thinking…".
                let timer = cx.background_executor().timer(Duration::from_millis(250));
                let next = rx.next();
                futures_util::pin_mut!(next);
                match future::select(next, timer).await {
                    Either::Left((None, _)) => break,
                    Either::Left((Some(()), _)) => {
                        while rx.try_recv().is_ok() {} // coalesce bursts
                    }
                    Either::Right(_) => {}
                }
                if this.update(cx, |this, cx| this.maybe_notify(cx)).is_err() {
                    break;
                }
                // Cap repaint rate (~20 fps) while tokens stream.
                cx.background_executor().timer(Duration::from_millis(50)).await;
            }
        })
        .detach();

        Self {
            client,
            tab,
            input: String::new(),
            input_focus: cx.focus_handle(),
            chat_list: ListState::new(0, ListAlignment::Bottom, px(512.)),
            last_fp: 0,
        }
    }

    /// Hash of everything the current tab renders from shared state.
    fn fingerprint(&self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        let s = self.client.shared.lock().unwrap();
        matches!(self.tab, Tab::Chat).hash(&mut h);
        matches!(self.tab, Tab::Library).hash(&mut h);
        s.daemon_up.hash(&mut h);
        s.events_connected.hash(&mut h);
        // footer state (visible on every tab)
        s.generating.hash(&mut h);
        ((s.gen_tok_s * 10.0) as u64).hash(&mut h);
        s.eject_unsupported.hash(&mut h);
        s.load_unsupported.hash(&mut h);
        s.loading_model.hash(&mut h);
        s.load_error.hash(&mut h);
        if s.loading_model.is_some() {
            thinking_phase().hash(&mut h); // indeterminate dots
            ((s.stats.resident_gb * 100.0) as u64).hash(&mut h); // progress bar
        }
        if let Some(c) = &s.capacity {
            c.engine_model.hash(&mut h);
            c.engine_up.hash(&mut h);
        }
        match self.tab {
            Tab::Chat => {
                s.messages.len().hash(&mut h);
                if let Some(m) = s.messages.last() {
                    m.content.len().hash(&mut h);
                }
                s.generating.hash(&mut h);
                ((s.gen_tok_s * 10.0) as u64).hash(&mut h);
                if s.generating {
                    thinking_phase().hash(&mut h);
                    ((s.stats.tok_s * 10.0) as u64).hash(&mut h); // prefill readout
                }
            }
            Tab::Library => {
                if let Some(c) = &s.capacity {
                    c.models.len().hash(&mut h);
                    c.engine_model.hash(&mut h);
                    c.engine_up.hash(&mut h);
                    (c.machine_ram_gb as u64).hash(&mut h);
                    (c.disk_free_gb as u64).hash(&mut h);
                    for m in &c.models {
                        m.name.hash(&mut h);
                        m.fits.hash(&mut h);
                        m.active.hash(&mut h);
                    }
                }
            }
            Tab::Stats => {
                s.tok_s_history.len().hash(&mut h);
                ((s.stats.tok_s * 10.0) as u64).hash(&mut h);
                ((s.stats.hit_rate * 1000.0) as u64).hash(&mut h);
                ((s.stats.resident_gb * 10.0) as u64).hash(&mut h);
                ((s.stats.streamed_mb_s * 10.0) as u64).hash(&mut h);
            }
            Tab::System => {
                s.system.is_some().hash(&mut h);
                s.system_unsupported.hash(&mut h);
                if let Some(sys) = &s.system {
                    sys.chip.hash(&mut h);
                    (sys.model_volume_free_gb as u64).hash(&mut h);
                }
                if let Some(c) = &s.capacity {
                    c.engine_model.hash(&mut h);
                    c.engine_up.hash(&mut h);
                    c.models.iter().filter(|m| m.fits).count().hash(&mut h);
                }
            }
        }
        h.finish()
    }

    fn maybe_notify(&mut self, cx: &mut Context<Self>) {
        let fp = self.fingerprint();
        if fp != self.last_fp {
            self.last_fp = fp;
            cx.notify();
        }
    }

    /// CD-changer footer: model "disc" in the slot, live readout, eject control.
    fn footer(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let t = theme();
        let (daemon_up, cap, generating, gen_tok_s, eject_unsupported, loading_model, stats, events) = {
            let s = self.client.shared.lock().unwrap();
            (
                s.daemon_up,
                s.capacity.clone(),
                s.generating,
                s.gen_tok_s,
                s.eject_unsupported,
                s.loading_model.clone(),
                s.stats,
                s.events_connected,
            )
        };
        let engine_up = cap.as_ref().is_some_and(|c| c.engine_up);
        let loading = daemon_up && loading_model.is_some();
        // progress = resident RAM climbing toward the loading model's RAM budget
        let progress = loading_model.as_ref().and_then(|name| {
            let ram = cap
                .as_ref()?
                .models
                .iter()
                .find(|m| &m.name == name)
                .map(|m| m.ram_gb)?;
            (events && ram > 0.0).then(|| ((stats.resident_gb / ram) as f32).clamp(0.0, 1.0))
        });
        let engine = loading_model.or_else(|| cap.as_ref().and_then(|c| c.engine_model.clone()));
        let loaded = daemon_up && engine_up && engine.is_some();

        // left: disc + name + state
        let slot: gpui::AnyElement = if !daemon_up {
            div()
                .flex()
                .items_center()
                .gap_2()
                .child(div().size_2p5().rounded_full().border_1().border_color(t.warn))
                .child(div().text_color(t.text_muted).child("daemon down — start overhangd"))
                .into_any_element()
        } else if loaded || loading {
            let name = engine.unwrap_or_default();
            div()
                .flex()
                .items_center()
                .gap_2()
                .child(div().size_2p5().rounded_full().bg(t.accent))
                .child(div().font_family(theme::MONO).child(name))
                .child(if loading {
                    library::load_progress(progress)
                } else {
                    div().text_color(t.text_muted).child("loaded").into_any_element()
                })
                .into_any_element()
        } else {
            div()
                .flex()
                .items_center()
                .gap_2()
                .child(div().size_2p5().rounded_full().border_1().border_color(t.border))
                .child(
                    div()
                        .text_color(t.text_muted)
                        .child("no model loaded — pick one in Library"),
                )
                .into_any_element()
        };

        // right: eject (atomic: never mid-load or mid-generation)
        let can_eject = loaded && !eject_unsupported && !loading && !generating;
        let eject: gpui::AnyElement = if eject_unsupported {
            div()
                .text_xs()
                .text_color(t.text_muted)
                .child("⏏ eject unavailable (update overhangd)")
                .into_any_element()
        } else {
            div()
                .id("eject")
                .px_3()
                .py_0p5()
                .rounded_md()
                .border_1()
                .border_color(t.border)
                .text_xs()
                .text_color(if can_eject { t.text } else { t.text_muted })
                .when(can_eject, |d| {
                    d.cursor_pointer()
                        .hover(|d| d.bg(t.warn_soft).border_color(t.warn).text_color(t.warn))
                        .on_click(cx.listener(|this, _, _, cx| {
                            // optimistic: slot empties now, /status reconciles after
                            if let Some(c) =
                                this.client.shared.lock().unwrap().capacity.as_mut()
                            {
                                c.engine_up = false;
                            }
                            this.client.send(Cmd::Eject);
                            cx.notify();
                        }))
                })
                .child("⏏ Eject")
                .into_any_element()
        };

        div()
            .h(px(40.))
            .flex_none()
            .flex()
            .items_center()
            .gap_3()
            .px_4()
            .bg(t.inset)
            .border_t_1()
            .border_color(t.border)
            .text_xs()
            .child(slot)
            .child(div().flex_grow(1.))
            .when(generating, |d| {
                d.child(
                    div()
                        .font_family(theme::MONO)
                        .text_color(t.text_muted)
                        .child(if gen_tok_s > 0.0 {
                            format!("· {gen_tok_s:.1} tok/s")
                        } else {
                            "· --.- tok/s".to_string() // no rate yet (pending)
                        }),
                )
            })
            .child(eject)
    }

    fn tab_button(
        &self,
        id: &'static str,
        tab: Tab,
        label: &'static str,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        let t = theme();
        let active = self.tab == tab;
        div()
            .id(id)
            .px_3()
            .py_1()
            .rounded_md()
            .cursor_pointer()
            .when(active, |d| d.bg(t.surface).text_color(t.accent))
            .when(!active, |d| d.text_color(t.text_muted))
            .hover(|d| d.bg(t.surface_hover))
            .on_click(cx.listener(move |this, _, _, cx| {
                this.tab = tab;
                this.last_fp = this.fingerprint();
                cx.notify();
            }))
            .child(label)
    }
}

impl Render for Root {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let t = theme();
        let daemon_up = self.client.shared.lock().unwrap().daemon_up;
        let body = match self.tab {
            Tab::Chat => chat::render(self, cx).into_any_element(),
            Tab::Library if daemon_up => library::render(self, cx).into_any_element(),
            Tab::Stats if daemon_up => stats::render(self, cx).into_any_element(),
            Tab::System if daemon_up => system::render(self, cx).into_any_element(),
            _ => daemon_down(self, cx).into_any_element(),
        };
        div()
            .size_full()
            .flex()
            .flex_col()
            .bg(t.bg)
            .text_color(t.text)
            .text_sm()
            .child(
                // custom titlebar: draggable, clears the inset traffic lights
                div()
                    .window_control_area(WindowControlArea::Drag)
                    .flex()
                    .items_center()
                    .gap_2()
                    .pl(px(80.))
                    .pr_4()
                    .py_2()
                    .border_b_1()
                    .border_color(t.border)
                    .child(div().text_lg().text_color(t.accent).child("overhang"))
                    .child(div().w_2())
                    .child(self.tab_button("tab-chat", Tab::Chat, "Chat", cx))
                    .child(self.tab_button("tab-library", Tab::Library, "Model library", cx))
                    .child(self.tab_button("tab-stats", Tab::Stats, "Live stats", cx))
                    .child(self.tab_button("tab-system", Tab::System, "System", cx))
                    .child(div().flex_grow(1.)),
            )
            .child(div().flex_grow(1.).min_h_0().child(body))
            .child(self.footer(cx))
    }
}

/// Full-pane "start overhangd" message with a retry button.
pub fn daemon_down(root: &Root, cx: &mut Context<Root>) -> impl IntoElement {
    let t = theme();
    let err: Option<SharedString> = root
        .client
        .shared
        .lock()
        .unwrap()
        .last_error
        .clone()
        .map(SharedString::from);
    div()
        .size_full()
        .flex()
        .flex_col()
        .items_center()
        .justify_center()
        .gap_2()
        .child(div().text_xl().child("overhangd is not running"))
        .child(div().text_color(t.text_muted).child("Start the daemon, then retry:"))
        .child(
            div()
                .font_family(theme::MONO)
                .text_color(t.accent)
                .child("overhangd   # listens on 127.0.0.1:11544"),
        )
        .children(err.map(|e| div().text_xs().text_color(t.text_muted).child(e)))
        .child(
            div()
                .id("retry")
                .mt_2()
                .px_4()
                .py_1()
                .rounded_md()
                .bg(t.surface)
                .border_1()
                .border_color(t.border)
                .cursor_pointer()
                .hover(|d| d.bg(t.surface_hover).border_color(t.accent))
                .on_click(cx.listener(|this, _, _, _| {
                    this.client.send(Cmd::RefreshStatus);
                }))
                .child("Retry connection"),
        )
}

pub fn run() {
    gpui_platform::application().run(|cx: &mut App| {
        let bounds = Bounds::centered(None, size(px(960.), px(680.)), cx);
        cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(bounds)),
                window_min_size: Some(size(px(720.), px(520.))),
                titlebar: Some(TitlebarOptions {
                    title: Some("overhang".into()),
                    appears_transparent: true,
                    traffic_light_position: Some(point(px(12.), px(12.))),
                }),
                ..Default::default()
            },
            |window, cx| {
                let root = cx.new(Root::new);
                window.focus(&root.read(cx).input_focus.clone(), cx);
                root
            },
        )
        .unwrap();
        cx.activate(true);
    });
}
