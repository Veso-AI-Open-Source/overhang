use super::{Root, theme::MONO, theme::theme};
use gpui::prelude::*;
use gpui::{
    Context, IntoElement, PathBuilder, canvas, div, linear_color_stop, linear_gradient, point, px,
};

fn tile(label: &'static str, value: String) -> gpui::Div {
    let t = theme();
    div()
        .min_w(px(150.))
        .px_4()
        .py_3()
        .rounded_lg()
        .bg(t.surface)
        .border_1()
        .border_color(t.border)
        .flex()
        .flex_col()
        .gap_1()
        .child(div().text_xs().text_color(t.text_muted).child(label))
        .child(div().text_2xl().font_family(MONO).child(value))
}

/// Accent sparkline with a soft gradient fill, painted with the low-level path API.
fn sparkline(history: Vec<f32>) -> impl IntoElement {
    let t = theme();
    canvas(
        |_, _, _| (),
        move |bounds, _, window, _| {
            if history.len() < 2 {
                return;
            }
            let max = history.iter().cloned().fold(1.0_f32, f32::max);
            let n = (history.len() - 1) as f32;
            let pt = |i: usize, v: f32| {
                point(
                    bounds.origin.x + bounds.size.width * (i as f32 / n),
                    bounds.origin.y + bounds.size.height
                        - px(3.)
                        - (bounds.size.height - px(6.)) * (v / max),
                )
            };
            // gradient fill under the line
            let mut fill = PathBuilder::fill();
            fill.move_to(point(bounds.origin.x, bounds.bottom_left().y));
            for (i, v) in history.iter().enumerate() {
                fill.line_to(pt(i, *v));
            }
            fill.line_to(bounds.bottom_right());
            fill.close();
            if let Ok(path) = fill.build() {
                window.paint_path(
                    path,
                    linear_gradient(
                        180.,
                        linear_color_stop(t.accent_soft, 0.),
                        linear_color_stop(gpui::transparent_black(), 1.),
                    ),
                );
            }
            // the line itself
            let mut line = PathBuilder::stroke(px(1.5));
            for (i, v) in history.iter().enumerate() {
                let p = pt(i, *v);
                if i == 0 {
                    line.move_to(p);
                } else {
                    line.line_to(p);
                }
            }
            if let Ok(path) = line.build() {
                window.paint_path(path, t.accent);
            }
        },
    )
    .w_full()
    .h(px(56.))
}

pub fn render(root: &mut Root, _cx: &mut Context<Root>) -> impl IntoElement {
    let t = theme();
    let (stats, history, connected) = {
        let s = root.client.shared.lock().unwrap();
        (s.stats, s.tok_s_history.clone(), s.events_connected)
    };

    div()
        .size_full()
        .flex()
        .flex_col()
        .gap_3()
        .px_4()
        .py_3()
        .child(
            div()
                .flex()
                .items_center()
                .gap_3()
                .child(div().text_lg().child("Live stats"))
                .child(
                    div()
                        .text_xs()
                        .text_color(if connected { t.ok } else { t.text_muted })
                        .child(if connected {
                            "ws /events connected"
                        } else {
                            "ws /events reconnecting…"
                        }),
                ),
        )
        .child(
            div()
                .flex()
                .flex_wrap()
                .gap_3()
                .items_start()
                .child(
                    // signature tile: big tok/s number with the sparkline underneath
                    tile("tok/s", format!("{:.1}", stats.tok_s))
                        .w(px(320.))
                        .child(div().text_xs().text_color(t.text_muted).child(
                            format!("last {} samples", history.len().max(1)),
                        ))
                        .child(sparkline(history)),
                )
                .child(tile("cache hit", format!("{:.0}%", stats.hit_rate * 100.0)))
                .child(tile("resident", format!("{:.1} GB", stats.resident_gb)))
                .child(tile("streamed", format!("{:.0} MB/s", stats.streamed_mb_s))),
        )
}
