use crate::api::Client;

pub fn show(ctx: &egui::Context, client: &Client) {
    let (stats, history, connected) = {
        let s = client.shared.lock().unwrap();
        (s.stats, s.tok_s_history.clone(), s.events_connected)
    };

    egui::CentralPanel::default().show(ctx, |ui| {
        ui.horizontal(|ui| {
            ui.heading("Live stats");
            if connected {
                ui.colored_label(egui::Color32::from_rgb(0x4c, 0xaf, 0x50), "ws /events connected");
            } else {
                ui.colored_label(egui::Color32::GRAY, "ws /events reconnecting…");
            }
        });
        ui.add_space(12.0);

        ui.horizontal(|ui| {
            readout(ui, "tok/s", format!("{:.1}", stats.tok_s));
            readout(ui, "cache hit", format!("{:.0}%", stats.hit_rate * 100.0));
            readout(ui, "resident", format!("{:.1} GB", stats.resident_gb));
            readout(ui, "streamed", format!("{:.0} MB/s", stats.streamed_mb_s));
        });

        ui.add_space(16.0);
        ui.label("tok/s (last 240 samples)");
        sparkline(ui, &history);
    });
}

fn readout(ui: &mut egui::Ui, label: &str, value: String) {
    egui::Frame::group(ui.style()).show(ui, |ui| {
        ui.vertical(|ui| {
            ui.set_min_width(120.0);
            ui.weak(label);
            ui.heading(value);
        });
    });
}

fn sparkline(ui: &mut egui::Ui, data: &[f32]) {
    let desired = egui::vec2(ui.available_width().min(720.0), 80.0);
    let (rect, _) = ui.allocate_exact_size(desired, egui::Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 4.0, ui.visuals().extreme_bg_color);
    if data.len() < 2 {
        return;
    }
    let max = data.iter().cloned().fold(1.0_f32, f32::max);
    let n = data.len() as f32;
    let pts: Vec<egui::Pos2> = data
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let x = rect.left() + (i as f32 / (n - 1.0)) * rect.width();
            let y = rect.bottom() - (v / max) * (rect.height() - 8.0) - 4.0;
            egui::pos2(x, y)
        })
        .collect();
    painter.add(egui::Shape::line(
        pts,
        egui::Stroke::new(1.5, egui::Color32::from_rgb(0x64, 0xb5, 0xf6)),
    ));
}
