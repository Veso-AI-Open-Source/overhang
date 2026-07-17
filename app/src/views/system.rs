//! System tab: the machine as the daemon discovered it — chip, cores, RAM,
//! GPU, Metal support, storage. Everything shown here is measured by
//! overhangd's discovery pass (GET /system); nothing is hardcoded.

use super::{Root, theme::MONO, theme::theme};
use crate::api::{Cmd, SystemInfo};
use gpui::prelude::*;
use gpui::{AnyElement, Context, IntoElement, div, px};

fn row(label: &str, value: impl Into<String>) -> AnyElement {
    let t = theme();
    div()
        .flex()
        .items_center()
        .gap_3()
        .child(div().w(px(140.)).text_color(t.text_muted).child(label.to_string()))
        .child(div().font_family(MONO).child(value.into()))
        .into_any_element()
}

/// ✓/— badge for discovered capabilities.
fn badge(label: &str, on: bool) -> AnyElement {
    let t = theme();
    div()
        .px_2()
        .py_0p5()
        .rounded_full()
        .text_xs()
        .when(on, |d| d.bg(t.ok_soft).text_color(t.ok))
        .when(!on, |d| d.bg(t.surface_hover).text_color(t.text_muted))
        .child(format!("{} {}", if on { "✓" } else { "—" }, label))
        .into_any_element()
}

fn panel(title: &str, rows: Vec<AnyElement>) -> AnyElement {
    let t = theme();
    div()
        .w(px(420.))
        .rounded_lg()
        .bg(t.surface)
        .border_1()
        .border_color(t.border)
        .p_3()
        .flex()
        .flex_col()
        .gap_1p5()
        .child(div().text_xs().text_color(t.text_muted).child(title.to_uppercase()))
        .children(rows)
        .into_any_element()
}

fn cores_line(sys: &SystemInfo) -> String {
    match (sys.perf_cores, sys.eff_cores) {
        (Some(p), Some(e)) => format!("{} ({}P + {}E)", sys.logical_cores, p, e),
        _ => format!("{}", sys.logical_cores),
    }
}

pub fn render(root: &mut Root, cx: &mut Context<Root>) -> impl IntoElement {
    let t = theme();
    let (system, system_unsupported, capacity, events_connected) = {
        let s = root.client.shared.lock().unwrap();
        (s.system.clone(), s.system_unsupported, s.capacity.clone(), s.events_connected)
    };

    let mut pane = div().size_full().flex().flex_col().gap_2().px_4().py_3().child(
        div()
            .flex()
            .items_center()
            .gap_3()
            .child(div().text_lg().child("System"))
            .child(
                div()
                    .id("rescan")
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
                        this.client.send(Cmd::RefreshStatus); // re-runs discovery fetch
                    }))
                    .child("rescan"),
            ),
    );

    if system_unsupported {
        return pane.child(
            div()
                .text_color(t.text_muted)
                .child("system discovery unavailable (update overhangd)"),
        );
    }
    let Some(sys) = system else {
        return pane.child(div().text_color(t.text_muted).child("Waiting for /system …"));
    };

    let machine = panel(
        "machine",
        vec![
            row("chip", sys.chip.clone()),
            row("architecture", sys.arch.clone()),
            row("cpu cores", cores_line(&sys)),
            row("memory", format!("{:.0} GB", sys.total_ram_gb)),
            row(
                "gpu cores",
                sys.gpu_cores.map_or("—".into(), |n| n.to_string()),
            ),
            row(
                "os",
                format!("{} {} (kernel {})", sys.os_name, sys.os_version, sys.kernel),
            ),
            div()
                .flex()
                .gap_2()
                .mt_1()
                .child(badge("Metal 4 tensor ops", sys.metal4))
                .child(badge("unified memory", sys.unified_memory))
                .into_any_element(),
        ],
    );

    let storage = panel(
        "storage (model volume)",
        vec![
            row(
                "free / total",
                format!(
                    "{:.0} / {:.0} GB",
                    sys.model_volume_free_gb, sys.model_volume_total_gb
                ),
            ),
            row("model root", sys.model_root.clone()),
        ],
    );

    let mut daemon_rows = vec![
        row("address", crate::api::addr()),
        row("events stream", if events_connected { "connected" } else { "disconnected" }),
    ];
    if let Some(cap) = &capacity {
        daemon_rows.push(row(
            "engine",
            match (&cap.engine_model, cap.engine_up) {
                (Some(m), true) => format!("{m} (up)"),
                _ => "no model loaded".into(),
            },
        ));
        let fitting = cap.models.iter().filter(|m| m.fits).count();
        daemon_rows.push(row(
            "capacity",
            format!("{} of {} installed models fit in RAM", fitting, cap.models.len()),
        ));
    }
    let daemon = panel("daemon", daemon_rows);

    pane = pane.child(
        div()
            .id("system-scroll")
            .flex_grow(1.)
            .min_h_0()
            .overflow_y_scroll()
            .mt_2()
            .flex()
            .flex_col()
            .gap_3()
            .child(machine)
            .child(storage)
            .child(daemon),
    );
    pane
}
