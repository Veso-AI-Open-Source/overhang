//! Interim egui/eframe UI, kept compilable behind `--features egui-fallback`.

pub mod chat;
pub mod library;
pub mod stats;

use crate::api::{self, Client, Cmd};

#[derive(PartialEq, Clone, Copy)]
enum Tab {
    Chat,
    Library,
    Stats,
}

struct OverhangApp {
    client: Client,
    tab: Tab,
    input: String,
}

impl eframe::App for OverhangApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let shared = self.client.shared.clone();
        let s = shared.lock().unwrap();

        egui::TopBottomPanel::top("tabs").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("overhang");
                ui.separator();
                ui.selectable_value(&mut self.tab, Tab::Chat, "Chat");
                ui.selectable_value(&mut self.tab, Tab::Library, "Model library");
                ui.selectable_value(&mut self.tab, Tab::Stats, "Live stats");
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let (dot, label) = if s.daemon_up {
                        (egui::Color32::from_rgb(0x4c, 0xaf, 0x50), "daemon up")
                    } else {
                        (egui::Color32::from_rgb(0xe5, 0x39, 0x35), "daemon down")
                    };
                    ui.colored_label(dot, format!("● {label}"));
                });
            });
        });

        if !s.daemon_up && self.tab != Tab::Chat {
            // Library and Stats need the daemon; Chat shows its own banner inline.
            drop(s);
            egui::CentralPanel::default().show(ctx, |ui| {
                daemon_down(ui, &self.client);
            });
        } else {
            match self.tab {
                Tab::Chat => {
                    drop(s);
                    chat::show(ctx, &self.client, &mut self.input);
                }
                Tab::Library => {
                    drop(s);
                    library::show(ctx, &self.client);
                }
                Tab::Stats => {
                    drop(s);
                    stats::show(ctx, &self.client);
                }
            }
        }
    }
}

/// Shared "daemon not reachable" panel with a retry button.
pub fn daemon_down(ui: &mut egui::Ui, client: &Client) {
    ui.vertical_centered(|ui| {
        ui.add_space(80.0);
        ui.heading("overhangd is not running");
        ui.add_space(8.0);
        ui.label("Start the daemon, then retry:");
        ui.monospace("overhangd   # listens on 127.0.0.1:11544");
        if let Some(err) = &client.shared.lock().unwrap().last_error {
            ui.add_space(4.0);
            ui.weak(err);
        }
        ui.add_space(12.0);
        if ui.button("Retry connection").clicked() {
            client.send(Cmd::RefreshStatus);
        }
    });
}

pub fn run() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([900.0, 640.0])
            .with_title("overhang"),
        ..Default::default()
    };
    eframe::run_native(
        "overhang",
        options,
        Box::new(|cc| {
            let ectx = cc.egui_ctx.clone();
            let client = api::start(move || ectx.request_repaint());
            Ok(Box::new(OverhangApp {
                client,
                tab: Tab::Chat,
                input: String::new(),
            }))
        }),
    )
}
