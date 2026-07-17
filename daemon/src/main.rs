mod chat;
mod config;
mod engine;
mod models;
mod server;
mod system;

use anyhow::{bail, Context, Result};
use std::path::PathBuf;
use std::sync::atomic::AtomicU64;
use std::sync::Arc;
use tokio::sync::{broadcast, Mutex};
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .init();

    let mut config_path = PathBuf::from(config::DEFAULT_CONFIG_PATH);
    let mut engine_cmd: Option<String> = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--config" => {
                config_path = PathBuf::from(args.next().context("--config requires a path")?)
            }
            "--engine-cmd" => {
                engine_cmd = Some(args.next().context("--engine-cmd requires a command")?)
            }
            other => bail!("unknown argument: {other} (usage: overhangd [--config <path>] [--engine-cmd <cmd>])"),
        }
    }

    let cfg = Arc::new(config::load(&config_path, engine_cmd)?);
    tracing::info!(?cfg, "loaded config");

    let sysinfo = system::info(); // discover once, up front
    tracing::info!(
        "system: {} ({}), {} cores, {:.0} GB RAM, gpu_cores={:?}, metal4={}",
        sysinfo.chip, sysinfo.arch, sysinfo.logical_cores, sysinfo.total_ram_gb,
        sysinfo.gpu_cores, sysinfo.metal4
    );

    let tok = Arc::new(std::sync::RwLock::new(chat::ChatCtx {
        tok: Arc::new(chat::Tok::load(&cfg.tokenizer)?),
        template: chat::Template::for_model_type(&models::read_model_type(&cfg.model_dir)),
    }));
    let status = Arc::new(engine::SharedStatus::default());
    let (events, _) = broadcast::channel(1024);
    let engine = Arc::new(Mutex::new(engine::Engine::new(cfg.clone(), status.clone())));

    let state = server::AppState {
        cfg: cfg.clone(),
        engine,
        tok,
        events,
        status,
        req_counter: Arc::new(AtomicU64::new(1)),
    };

    let app = server::router(state);
    let listener = tokio::net::TcpListener::bind(("0.0.0.0", cfg.port))
        .await
        .with_context(|| format!("binding port {}", cfg.port))?;
    tracing::info!("overhangd listening on http://0.0.0.0:{}", cfg.port);
    axum::serve(listener, app).await?;
    Ok(())
}
