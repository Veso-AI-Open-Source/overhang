//! Design tokens. Dark only for v1; warm amber accent ("overhang" = cliff at sunset).
//! Views must pull every color from here — no inline hex in view code.

use gpui::{Rgba, rgb, rgba};
use std::sync::OnceLock;

pub const MONO: &str = "Menlo";

pub struct Theme {
    pub bg: Rgba,
    pub inset: Rgba, // recessed strips (footer slot), darker than surface
    pub surface: Rgba,
    pub surface_hover: Rgba,
    pub border: Rgba,
    pub text: Rgba,
    pub text_muted: Rgba,
    pub accent: Rgba,
    pub accent_soft: Rgba, // accent-tinted fill (user bubbles, sparkline base)
    pub ok: Rgba,
    pub ok_soft: Rgba,
    pub warn: Rgba,
    pub warn_soft: Rgba,
}

pub fn theme() -> &'static Theme {
    static T: OnceLock<Theme> = OnceLock::new();
    T.get_or_init(|| Theme {
        bg: rgb(0x141317),
        inset: rgb(0x0f0e12),
        surface: rgb(0x1d1c22),
        surface_hover: rgb(0x26242c),
        border: rgb(0x2e2b33),
        text: rgb(0xece7e1),
        text_muted: rgb(0x948f96),
        accent: rgb(0xf08c3e),
        accent_soft: rgba(0xf08c3e26),
        ok: rgb(0x5fce7b),
        ok_soft: rgba(0x5fce7b22),
        warn: rgb(0xe06c65),
        warn_soft: rgba(0xe06c6522),
    })
}
