use super::{Root, theme::MONO, theme::theme, thinking_phase};
use crate::api::{ChatMsg, Cmd};
use gpui::prelude::*;
use gpui::{AnyElement, Context, IntoElement, div, list, relative, rgba};

fn send(root: &mut Root) {
    let text = root.input.trim().to_string();
    let idle = {
        let s = root.client.shared.lock().unwrap();
        !s.generating && s.loading_model.is_none() // atomic: one engine op at a time
    };
    if !text.is_empty() && idle {
        root.client.send(Cmd::SendChat(text));
        root.input.clear();
    }
}

/// One message bubble: user right-aligned accent-tinted, assistant left on surface.
fn bubble(m: &ChatMsg, streaming: bool, prefill_tok_s: f64) -> AnyElement {
    let t = theme();
    let user = m.role == "user";
    let thinking = streaming && m.content.is_empty();

    let body: AnyElement = if thinking {
        let dots = ".".repeat(thinking_phase() as usize + 1);
        let label = if prefill_tok_s > 0.0 {
            format!("reading prompt{dots}  {prefill_tok_s:.0} tok/s prefill")
        } else {
            format!("reading prompt{dots}")
        };
        div().text_color(t.text_muted).font_family(MONO).child(label).into_any_element()
    } else if streaming {
        // monospace + block cursor while tokens arrive
        div()
            .font_family(MONO)
            .child(format!("{}\u{258c}", m.content))
            .into_any_element()
    } else {
        div().child(m.content.clone()).into_any_element()
    };

    div()
        .w_full()
        .py_1()
        .px_4()
        .flex()
        .when(user, |d| d.justify_end())
        .child(
            div()
                .max_w(relative(0.78))
                .px_3()
                .py_2()
                .rounded_lg()
                .border_1()
                .when(user, |d| d.bg(t.accent_soft).border_color(rgba(0x00000000)))
                .when(!user, |d| d.bg(t.surface).border_color(t.border))
                .child(body),
        )
        .into_any_element()
}

pub fn render(root: &mut Root, cx: &mut Context<Root>) -> impl IntoElement {
    let t = theme();
    let (messages, generating, gen_tok_s, daemon_up, prefill_tok_s) = {
        let s = root.client.shared.lock().unwrap();
        (s.messages.clone(), s.generating, s.gen_tok_s, s.daemon_up, s.stats.tok_s)
    };

    if root.chat_list.item_count() != messages.len() {
        root.chat_list.reset(messages.len());
    }
    let n = messages.len();
    let items = move |ix: usize, _: &mut gpui::Window, _: &mut gpui::App| {
        let streaming = generating && ix + 1 == n;
        bubble(&messages[ix], streaming, prefill_tok_s)
    };

    div()
        .size_full()
        .flex()
        .flex_col()
        .when(!daemon_up, |d| {
            d.child(
                div()
                    .px_4()
                    .py_1()
                    .border_b_1()
                    .border_color(t.border)
                    .flex()
                    .items_center()
                    .gap_2()
                    .child(div().text_color(t.warn).child("daemon unreachable —"))
                    .child(div().font_family(MONO).child("start overhangd"))
                    .child(
                        div()
                            .id("chat-retry")
                            .px_2()
                            .rounded_md()
                            .bg(t.surface)
                            .cursor_pointer()
                            .hover(|d| d.bg(t.surface_hover))
                            .on_click(cx.listener(|this, _, _, _| {
                                this.client.send(Cmd::RefreshStatus);
                            }))
                            .child("retry"),
                    ),
            )
        })
        .child(if n == 0 {
            div()
                .flex_grow(1.)
                .flex()
                .items_center()
                .justify_center()
                .text_color(t.text_muted)
                .child("No messages yet. Type below to talk to the active model.")
                .into_any_element()
        } else {
            list(root.chat_list.clone(), items)
                .flex_grow(1.)
                .py_2()
                .into_any_element()
        })
        .child(
            // composer
            div()
                .flex()
                .gap_2()
                .items_center()
                .px_4()
                .py_3()
                .border_t_1()
                .border_color(t.border)
                .when(generating, |d| {
                    d.child(
                        div()
                            .px_2()
                            .py_1()
                            .rounded_full()
                            .bg(t.accent_soft)
                            .text_xs()
                            .font_family(MONO)
                            .text_color(t.accent)
                            .child(if gen_tok_s > 0.0 {
                                format!("{gen_tok_s:.1} tok/s")
                            } else {
                                "--.- tok/s".to_string() // no rate yet (pending)
                            }),
                    )
                })
                .child(
                    div()
                        .id("chat-input")
                        .track_focus(&root.input_focus)
                        .flex_grow(1.)
                        .px_3()
                        .py_1()
                        .rounded_md()
                        .bg(t.surface)
                        .border_1()
                        .border_color(t.border)
                        .font_family(MONO)
                        .on_click(cx.listener(|this, _, window, cx| {
                            this.input_focus.focus(window, cx);
                        }))
                        .on_key_down(cx.listener(|this, ev: &gpui::KeyDownEvent, _, cx| {
                            let ks = &ev.keystroke;
                            if ks.modifiers.platform || ks.modifiers.control {
                                return;
                            }
                            match ks.key.as_str() {
                                "enter" => send(this),
                                "backspace" => {
                                    this.input.pop();
                                }
                                _ => {
                                    if let Some(ch) = &ks.key_char {
                                        this.input.push_str(ch);
                                    }
                                }
                            }
                            cx.notify();
                        }))
                        .child(if root.input.is_empty() {
                            div()
                                .text_color(t.text_muted)
                                .child("Ask the model… (click here, type, press enter)")
                                .into_any_element()
                        } else {
                            div().child(root.input.clone()).into_any_element()
                        }),
                )
                .child(
                    div()
                        .id("send")
                        .px_4()
                        .py_1()
                        .rounded_md()
                        .bg(t.surface)
                        .border_1()
                        .border_color(t.border)
                        .cursor_pointer()
                        .text_color(if generating { t.text_muted } else { t.accent })
                        .hover(|d| d.bg(t.surface_hover).border_color(t.accent))
                        .on_click(cx.listener(|this, _, _, cx| {
                            send(this);
                            cx.notify();
                        }))
                        .child("Send"),
                )
                .when(n > 0, |d| {
                    d.child(
                        div()
                            .id("clear-chat")
                            .px_3()
                            .py_1()
                            .rounded_md()
                            .border_1()
                            .border_color(t.border)
                            .cursor_pointer()
                            .text_color(t.text_muted)
                            .hover(|d| d.bg(t.warn_soft).border_color(t.warn).text_color(t.warn))
                            .on_click(cx.listener(|this, _, _, cx| {
                                // not while streaming: the backend writes into
                                // the last message by position
                                let mut s = this.client.shared.lock().unwrap();
                                if !s.generating {
                                    s.messages.clear();
                                }
                                drop(s);
                                cx.notify();
                            }))
                            .child("Clear"),
                    )
                }),
        )
}
