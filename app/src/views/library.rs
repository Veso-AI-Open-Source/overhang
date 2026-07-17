//! Model library as a CD changer: each model is a disc in a tray; the active
//! one is "in the slot". Load spins the engine up on that container.

use super::{Root, theme::MONO, theme::theme};
use crate::api::{Cmd, ModelRow};
use gpui::prelude::*;
use gpui::{AnyElement, Context, ElementId, IntoElement, div, px};

/// Downloadable catalog: models the shipped engines can actually run
/// (qwen3_5_moe family -> qwen, olmoe -> olmoe). Sizes are int8-container
/// estimates used only for fit hints; installed rows always show measured
/// numbers from the daemon instead. `tool` is the colibri command that
/// produces the container.
struct CatalogEntry {
    display: &'static str,
    container: &'static str, // expected container dir name (ladder `name`)
    disk_gb: f64,
    ram_gb: f64,
    tool: &'static str,
}

const CATALOG: &[CatalogEntry] = &[
    CatalogEntry {
        display: "Qwen3.6-35B-A3B", // HF: Qwen/Qwen3.6-35B-A3B (colibri int8 conversion)
        container: "qwen36_i8",
        disk_gb: 39.0,
        ram_gb: 16.7,
        tool: "colibri convert_qwen.py",
    },
    CatalogEntry {
        display: "Qwen3.5-122B-A10B",
        container: "qwen35_122b_i8",
        disk_gb: 61.0,
        ram_gb: 21.0,
        tool: "colibri convert_qwen.py",
    },
    CatalogEntry {
        display: "Gemma 4 26B-A4B (instruct)",
        container: "gemma4_26b_i8",
        disk_gb: 26.0, // measured container (2026-07-17)
        ram_gb: 10.0,  // measured at the engine's 24-slot expert cache
        tool: "overhang convert_gemma.py",
    },
    CatalogEntry {
        display: "OLMoE-1B-7B-Instruct",
        container: "olmoe_i8",
        disk_gb: 13.0,
        ram_gb: 11.0,
        tool: "colibri convert_olmoe.py",
    },
    CatalogEntry {
        display: "GLM-5.2",
        container: "glm52_i8",
        disk_gb: 700.0,
        ram_gb: 15.0,
        tool: "colibri download_glm52.py",
    },
];

/// Full display name for an installed container, when the catalog knows it.
fn display_name(container: &str) -> Option<&'static str> {
    CATALOG.iter().find(|c| c.container == container).map(|c| c.display)
}

/// Mirrors the daemon's RAM_FIT_FRACTION: a model needs headroom next to the
/// OS and other apps.
const RAM_FIT_FRACTION: f64 = 0.9;

/// A compact disc: ring, data surface, center hole.
fn disc(active: bool) -> impl IntoElement {
    let t = theme();
    div()
        .size_8()
        .rounded_full()
        .border_2()
        .border_color(if active { t.accent } else { t.border })
        .bg(if active { t.accent_soft } else { t.surface_hover })
        .flex()
        .items_center()
        .justify_center()
        .child(div().size_2().rounded_full().bg(t.inset).border_1().border_color(t.border))
}

/// "loading… 62%" with a thin accent bar; indeterminate dots when no signal yet.
pub fn load_progress(progress: Option<f32>) -> AnyElement {
    let t = theme();
    let row = div().flex().items_center().gap_2().text_color(t.accent);
    match progress {
        Some(f) => row
            .child(format!("loading… {:.0}%", f * 100.0))
            .child(
                div().w(px(96.)).h(px(3.)).rounded_full().bg(t.surface_hover).child(
                    div().h_full().w(gpui::relative(f)).rounded_full().bg(t.accent),
                ),
            )
            .into_any_element(),
        None => row
            .child(format!(
                "loading{}",
                ".".repeat(super::thinking_phase() as usize + 1)
            ))
            .into_any_element(),
    }
}

/// Shared list-row chrome: disc + name/meta on the left, badge + state on the
/// right. Full-width rows read as a clean list; nothing can clip.
fn row_shell(
    active: bool,
    fits: bool,
    display: String,
    meta: String,
    badge_state: AnyElement,
) -> gpui::Div {
    let t = theme();
    div()
        .w_full()
        .rounded_lg()
        .bg(t.surface)
        .border_1()
        .border_color(if active {
            t.accent
        } else if !fits {
            t.warn
        } else {
            t.border
        })
        .px_3()
        .py_2()
        .flex()
        .items_center()
        .gap_3()
        .child(disc(active))
        .child(
            div()
                .flex()
                .flex_col()
                .min_w_0() // long names/specs wrap instead of pushing the right side out
                .child(div().child(display))
                .child(div().text_xs().text_color(t.text_muted).child(meta)),
        )
        .child(div().flex_grow(1.))
        .child(badge_state)
}

/// "✓ fits" / "✗ too big" pill.
fn fits_badge(fits: bool) -> AnyElement {
    let t = theme();
    div()
        .flex_none()
        .px_2()
        .py_0p5()
        .rounded_full()
        .text_xs()
        .when(fits, |d| d.bg(t.ok_soft).text_color(t.ok))
        .when(!fits, |d| d.bg(t.warn_soft).text_color(t.warn))
        .child(if fits { "✓ fits" } else { "✗ too big" })
        .into_any_element()
}

fn tray(
    m: &ModelRow,
    loading: bool,
    progress: Option<f32>,
    busy: bool, // a load or generation is in flight: keep ops atomic, no new Load
    error: Option<String>, // this model's last load failure, until refresh/retry
    load_unsupported: bool,
    cx: &mut Context<Root>,
) -> AnyElement {
    let t = theme();

    // right-side state: text or the Load control
    let state: AnyElement = if m.active {
        div().text_color(t.accent).child("● in the slot").into_any_element()
    } else if loading {
        load_progress(progress)
    } else if let Some(err) = error {
        div().text_color(t.warn).child(format!("✗ load failed: {err}")).into_any_element()
    } else if !m.fits {
        div().text_color(t.warn).child("won't fit on this machine").into_any_element()
    } else if load_unsupported {
        div().text_color(t.text_muted).child("load unavailable (update overhangd)").into_any_element()
    } else if busy {
        div().text_color(t.text_muted).child("⏵ Load  (slot busy)").into_any_element()
    } else {
        let name = m.name.clone();
        div()
            .id(ElementId::Name(format!("load-{}", m.name).into()))
            .flex_none()
            .px_3()
            .py_0p5()
            .rounded_md()
            .border_1()
            .border_color(t.border)
            .cursor_pointer()
            .hover(|d| d.bg(t.accent_soft).border_color(t.accent).text_color(t.accent))
            .on_click(cx.listener(move |this, _, _, cx| {
                // atomic: ignore if something is already in flight
                let busy_now = {
                    let s = this.client.shared.lock().unwrap();
                    s.loading_model.is_some() || s.generating
                };
                if !busy_now {
                    this.client.send(Cmd::Load(name.clone()));
                }
                cx.notify();
            }))
            .child("⏵ Load")
            .into_any_element()
    };

    row_shell(
        m.active,
        m.fits,
        display_name(&m.name).map(String::from).unwrap_or_else(|| m.name.clone()),
        format!("{} · {:.0} GB disk · {:.0} GB RAM", m.name, m.disk_gb, m.ram_gb),
        div()
            .flex_none()
            .flex()
            .items_center()
            .gap_3()
            .text_xs()
            .font_family(MONO)
            .child(state)
            .child(fits_badge(m.fits))
            .into_any_element(),
    )
    .into_any_element()
}

/// Row for a catalog model that isn't installed: est. sizes, how to get it,
/// and a clear warn outline + reason when it won't fit on this machine.
fn catalog_card(e: &CatalogEntry, ram_budget: f64, disk_free: f64) -> AnyElement {
    let t = theme();
    let fits_ram = e.ram_gb <= ram_budget;
    let fits_disk = e.disk_gb <= disk_free;
    let fits = fits_ram && fits_disk;
    let state: AnyElement = if fits {
        div()
            .text_color(t.text_muted)
            .child(format!("not installed · {}", e.tool))
            .into_any_element()
    } else if !fits_ram {
        div()
            .text_color(t.warn)
            .child(format!("won't fit — needs ~{:.0} GB RAM", e.ram_gb))
            .into_any_element()
    } else {
        div()
            .text_color(t.warn)
            .child(format!("won't fit — needs ~{:.0} GB disk", e.disk_gb))
            .into_any_element()
    };
    row_shell(
        false,
        fits,
        e.display.to_string(),
        format!("{} · ~{:.0} GB disk · ~{:.0} GB RAM", e.container, e.disk_gb, e.ram_gb),
        div()
            .flex_none()
            .flex()
            .items_center()
            .gap_3()
            .text_xs()
            .font_family(MONO)
            .child(state)
            .child(fits_badge(fits))
            .into_any_element(),
    )
    .into_any_element()
}

pub fn render(root: &mut Root, cx: &mut Context<Root>) -> impl IntoElement {
    let t = theme();
    let (capacity, loading_model, load_error, load_unsupported, generating, resident_gb, events_connected) = {
        let s = root.client.shared.lock().unwrap();
        (
            s.capacity.clone(),
            s.loading_model.clone(),
            s.load_error.clone(),
            s.load_unsupported,
            s.generating,
            s.stats.resident_gb,
            s.events_connected,
        )
    };
    let busy = loading_model.is_some() || generating;

    let pane = div().size_full().flex().flex_col().gap_2().px_4().py_3().child(
        div()
            .flex()
            .items_center()
            .gap_3()
            .child(div().text_lg().child("Model library"))
            .child(
                div()
                    .id("refresh")
                    .px_2()
                    .py_0p5()
                    .rounded_md()
                    .bg(t.surface)
                    .border_1()
                    .border_color(t.border)
                    .text_xs()
                    .cursor_pointer()
                    .hover(|d| d.bg(t.surface_hover).text_color(t.accent))
                    .on_click(cx.listener(|this, _, _, _| {
                        this.client.send(Cmd::RefreshStatus);
                    }))
                    .child("refresh"),
            ),
    );

    let Some(cap) = capacity else {
        return pane.child(div().text_color(t.text_muted).child("Waiting for /status …"));
    };

    // catalog rows = known models not already installed
    let ram_budget = cap.machine_ram_gb * RAM_FIT_FRACTION;
    let downloadable: Vec<&CatalogEntry> = CATALOG
        .iter()
        .filter(|e| !cap.models.iter().any(|m| m.name == e.container))
        .collect();

    let mut changer = div()
        .id("changer")
        .flex_grow(1.)
        .min_h_0()
        .overflow_y_scroll()
        .mt_2()
        .flex()
        .flex_col()
        .gap_2();

    if cap.models.is_empty() {
        changer = changer
            .child(div().text_color(t.text_muted).child("No models reported by the daemon."));
    } else {
        changer = changer.child(div().text_xs().text_color(t.text_muted).child("installed")).child(
            div().flex().flex_col().gap_2().children(cap.models.iter().map(|m| {
                let loading = loading_model.as_deref() == Some(m.name.as_str());
                let progress = (loading && events_connected && m.ram_gb > 0.0)
                    .then(|| ((resident_gb / m.ram_gb) as f32).clamp(0.0, 1.0));
                let error = load_error
                    .as_ref()
                    .filter(|(n, _)| n == &m.name)
                    .map(|(_, e)| e.clone());
                tray(m, loading, progress, busy, error, load_unsupported, cx)
            })),
        );
    }

    if !downloadable.is_empty() {
        changer = changer
            .child(div().mt_2().text_xs().text_color(t.text_muted).child("catalog — not installed"))
            .child(
                div().flex().flex_col().gap_2().children(
                    downloadable
                        .into_iter()
                        .map(|e| catalog_card(e, ram_budget, cap.disk_free_gb)),
                ),
            );
    }

    pane.child(changer)
}
