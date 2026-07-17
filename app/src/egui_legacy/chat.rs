use crate::api::{Client, Cmd};

pub fn show(ctx: &egui::Context, client: &Client, input: &mut String) {
    let (messages, generating, gen_tok_s, daemon_up) = {
        let s = client.shared.lock().unwrap();
        (s.messages.clone(), s.generating, s.gen_tok_s, s.daemon_up)
    };

    egui::TopBottomPanel::bottom("chat_input").show(ctx, |ui| {
        ui.add_space(6.0);
        let mut send_now = false;
        ui.horizontal(|ui| {
            let edit = egui::TextEdit::singleline(input)
                .hint_text("Ask the model…")
                .desired_width(ui.available_width() - 70.0);
            let resp = ui.add_enabled(!generating, edit);
            if resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                send_now = true;
            }
            if ui.add_enabled(!generating, egui::Button::new("Send")).clicked() {
                send_now = true;
            }
        });
        ui.add_space(6.0);
        if send_now {
            let text = input.trim().to_string();
            if !text.is_empty() {
                client.send(Cmd::SendChat(text));
                input.clear();
            }
        }
    });

    egui::CentralPanel::default().show(ctx, |ui| {
        if !daemon_up {
            ui.horizontal(|ui| {
                ui.colored_label(egui::Color32::from_rgb(0xe5, 0x39, 0x35), "daemon unreachable —");
                ui.monospace("start overhangd");
                if ui.small_button("retry").clicked() {
                    client.send(Cmd::RefreshStatus);
                }
            });
            ui.separator();
        }

        // live tok/s badge in the corner while generating
        if generating {
            egui::Area::new(egui::Id::new("toks_badge"))
                .anchor(egui::Align2::RIGHT_TOP, [-16.0, 40.0])
                .show(ctx, |ui| {
                    egui::Frame::popup(ui.style()).show(ui, |ui| {
                        ui.monospace(format!("{gen_tok_s:5.1} tok/s"));
                    });
                });
        }

        egui::ScrollArea::vertical()
            .stick_to_bottom(true)
            .auto_shrink([false, false])
            .show(ui, |ui| {
                if messages.is_empty() {
                    ui.add_space(24.0);
                    ui.weak("No messages yet. Type below to talk to the active model.");
                }
                for m in &messages {
                    let (name, color) = if m.role == "user" {
                        ("you", egui::Color32::from_rgb(0x64, 0xb5, 0xf6))
                    } else {
                        ("model", egui::Color32::from_rgb(0x81, 0xc7, 0x84))
                    };
                    ui.add_space(8.0);
                    ui.colored_label(color, name);
                    let body = if m.content.is_empty() && generating { "…" } else { &m.content };
                    ui.label(body);
                }
                ui.add_space(8.0);
            });
    });
}
