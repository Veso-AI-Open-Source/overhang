// egui-fallback wins if both features are enabled (gpui-ui is on by default).
#[cfg(feature = "egui-fallback")]
fn main() -> eframe::Result {
    overhang_app::egui_legacy::run()
}

#[cfg(all(feature = "gpui-ui", not(feature = "egui-fallback")))]
fn main() {
    overhang_app::views::run();
}

#[cfg(not(any(feature = "gpui-ui", feature = "egui-fallback")))]
compile_error!("enable the `gpui-ui` (default) or `egui-fallback` feature");
