use anyhow::{Context, Result};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

pub const DEFAULT_CONFIG_PATH: &str = "./overhangd.toml";

const DEFAULT_CONFIG: &str = r#"# overhangd configuration
engine_bin = "../qwen"
model_dir = "../model"
qcache = 96
port = 11544
# tokenizer defaults to <model_dir>/tokenizer.json
# tokenizer = "../model/tokenizer.json"
# engine_cmd overrides engine_bin entirely (useful for testing):
# engine_cmd = "python3 tests/mock_engine.py"
# per-architecture engine binaries (model_type -> binary); unlisted types
# fall back to a sibling of engine_bin named after the architecture family:
# [engines]
# olmoe = "../olmoe"
"#;

#[derive(Debug, Default, Deserialize)]
struct FileConfig {
    engine_bin: Option<String>,
    engine_cmd: Option<String>,
    model_dir: Option<PathBuf>,
    qcache: Option<u64>,
    port: Option<u16>,
    tokenizer: Option<PathBuf>,
    engines: Option<BTreeMap<String, String>>,
}

#[derive(Debug, Clone)]
pub struct Config {
    pub engine_bin: String,
    /// Full command line that overrides `engine_bin --serve` (testing hook).
    pub engine_cmd: Option<String>,
    pub model_dir: PathBuf,
    pub qcache: u64,
    pub port: u16,
    pub tokenizer: PathBuf,
    /// model_type -> engine binary overrides.
    pub engines: BTreeMap<String, String>,
}

impl Config {
    /// Engine binary for a container's model_type: explicit [engines] entry,
    /// else a sibling of engine_bin named after the architecture family
    /// (qwen3_5_moe -> qwen, olmoe -> olmoe), else engine_bin itself.
    pub fn engine_for(&self, model_type: &str) -> String {
        if let Some(bin) = self.engines.get(model_type) {
            return bin.clone();
        }
        let family = ["qwen", "olmoe", "gemma", "glm"]
            .into_iter()
            .find(|f| model_type.contains(f));
        match family {
            Some(f) => Path::new(&self.engine_bin)
                .with_file_name(f)
                .to_string_lossy()
                .into_owned(),
            None => self.engine_bin.clone(),
        }
    }
}

pub fn load(path: &Path, engine_cmd_override: Option<String>) -> Result<Config> {
    let file: FileConfig = if path.exists() {
        let raw = std::fs::read_to_string(path)
            .with_context(|| format!("reading config {}", path.display()))?;
        toml::from_str(&raw).with_context(|| format!("parsing config {}", path.display()))?
    } else {
        std::fs::write(path, DEFAULT_CONFIG)
            .with_context(|| format!("writing default config {}", path.display()))?;
        tracing::info!("wrote default config to {}", path.display());
        toml::from_str(DEFAULT_CONFIG).expect("default config parses")
    };

    let model_dir = file.model_dir.unwrap_or_else(|| PathBuf::from("../model"));
    let tokenizer = file
        .tokenizer
        .unwrap_or_else(|| model_dir.join("tokenizer.json"));

    Ok(Config {
        engine_bin: file.engine_bin.unwrap_or_else(|| "../qwen".to_string()),
        engine_cmd: engine_cmd_override.or(file.engine_cmd),
        qcache: file.qcache.unwrap_or(96),
        port: file.port.unwrap_or(11544),
        model_dir,
        tokenizer,
        engines: file.engines.unwrap_or_default(),
    })
}
