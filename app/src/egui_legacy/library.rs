use crate::api::{Client, Cmd};

pub fn show(ctx: &egui::Context, client: &Client) {
    let capacity = client.shared.lock().unwrap().capacity.clone();

    egui::CentralPanel::default().show(ctx, |ui| {
        ui.horizontal(|ui| {
            ui.heading("Model library");
            if ui.button("⟳ Refresh").clicked() {
                client.send(Cmd::RefreshStatus);
            }
        });
        ui.add_space(4.0);

        let Some(cap) = capacity else {
            ui.weak("Waiting for /status …");
            return;
        };

        ui.label(format!(
            "This machine: {:.0} GB RAM, {:.0} GB free disk",
            cap.machine_ram_gb, cap.disk_free_gb
        ));
        if let Some(engine) = &cap.engine_model {
            ui.label(format!(
                "Engine: {engine} ({})",
                if cap.engine_up { "up" } else { "down" }
            ));
        }
        ui.add_space(8.0);

        if cap.models.is_empty() {
            ui.weak("No models reported by the daemon.");
            return;
        }

        egui::Grid::new("capacity_ladder")
            .num_columns(5)
            .spacing([24.0, 6.0])
            .striped(true)
            .show(ui, |ui| {
                ui.strong("Model");
                ui.strong("Disk");
                ui.strong("RAM");
                ui.strong("Fits");
                ui.strong("");
                ui.end_row();
                for m in &cap.models {
                    ui.monospace(&m.name);
                    ui.label(format!("{:.1} GB", m.disk_gb));
                    ui.label(format!("{:.1} GB", m.ram_gb));
                    if m.fits {
                        ui.colored_label(egui::Color32::from_rgb(0x4c, 0xaf, 0x50), "✓ fits");
                    } else {
                        ui.colored_label(egui::Color32::from_rgb(0xe5, 0x39, 0x35), "✗ too big");
                    }
                    if m.active {
                        ui.colored_label(egui::Color32::from_rgb(0x64, 0xb5, 0xf6), "● active");
                    } else {
                        ui.label("");
                    }
                    ui.end_row();
                }
            });
    });
}
