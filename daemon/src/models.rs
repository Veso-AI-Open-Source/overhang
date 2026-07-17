//! Model container discovery. The capacity ladder is measured from what is
//! actually on disk — container sizes from the file system, RAM estimates
//! from safetensors headers + config.json — never hardcoded.

use serde_json::Value;
use std::collections::HashSet;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct ModelInfo {
    /// Container dir name — the canonical model id. Matches the engine's
    /// ready-line `model` (SNAP basename), so /status "active" lines up.
    pub name: String,
    pub model_type: String,
    pub disk_gb: f64,
    /// Estimated resident RAM at `qcache` expert slots per layer.
    pub est_ram_gb: f64,
}

/// Root that holds model containers: the parent of the configured model_dir
/// (containers are siblings, e.g. models/{qwen36_i8, olmoe_i4}).
pub fn model_root(model_dir: &Path) -> PathBuf {
    model_dir
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."))
        .to_path_buf()
}

pub fn read_model_type(dir: &Path) -> String {
    read_config(dir)
        .and_then(|c| c["model_type"].as_str().map(String::from))
        .unwrap_or_default()
}

fn read_config(dir: &Path) -> Option<Value> {
    let raw = std::fs::read_to_string(dir.join("config.json")).ok()?;
    serde_json::from_str(&raw).ok()
}

/// config value that may live at top level or under text_config (multimodal).
fn cfg_u64(c: &Value, key: &str) -> Option<u64> {
    c[key].as_u64().or_else(|| c["text_config"][key].as_u64())
}

/// Expert-cache slots the engine will actually use for this architecture.
/// Mirrors the engines' internal caps so the ladder's RAM estimate reflects
/// reality, not the config's global qcache (gemma.c caps at 24 — measured
/// sweep: beyond that macOS compresses the LRU slots and decode gets slower).
pub fn effective_qcache(model_type: &str, cfg_qcache: u64) -> u64 {
    if model_type.contains("gemma") { cfg_qcache.min(24) } else { cfg_qcache }
}

/// Scan `root` for containers (dirs with config.json), measured not assumed.
pub fn scan(root: &Path, qcache: u64) -> Vec<ModelInfo> {
    let mut out = Vec::new();
    let Ok(rd) = std::fs::read_dir(root) else {
        return out;
    };
    for entry in rd.flatten() {
        let path = entry.path();
        if !path.is_dir() || !path.join("config.json").exists() {
            continue;
        }
        let name = entry.file_name().to_string_lossy().to_string();
        out.push(inspect(name, path, qcache));
    }
    out.sort_by(|a, b| a.disk_gb.total_cmp(&b.disk_gb));
    out
}

fn inspect(name: String, path: PathBuf, qcache: u64) -> ModelInfo {
    let config = read_config(&path).unwrap_or(Value::Null);
    let model_type = config["model_type"].as_str().unwrap_or("").to_string();

    let mut disk_bytes: u64 = 0;
    let mut shards: Vec<PathBuf> = Vec::new();
    if let Ok(rd) = std::fs::read_dir(&path) {
        for f in rd.flatten() {
            disk_bytes += f.metadata().map(|m| m.len()).unwrap_or(0);
            let p = f.path();
            if p.extension().is_some_and(|e| e == "safetensors") {
                shards.push(p);
            }
        }
    }

    // Tally tensors across shard headers: dense stays resident (big matrices
    // quantized to int8 by the engine, embeddings/norms f32); routed experts
    // stream through a per-layer LRU of `qcache` slots (mirrors the engine's
    // own "cache %d/layer" accounting in engines/qwen.c).
    let mut dense_resident_bytes: f64 = 0.0;
    let mut expert_bytes: u64 = 0; // as stored (int8 + scales)
    let mut expert_keys: HashSet<(String, String)> = HashSet::new();
    let mut expert_int8 = false;
    let mut layer_prefixes: HashSet<String> = HashSet::new();
    for shard in &shards {
        let Some(header) = read_st_header(shard) else {
            continue;
        };
        let Some(map) = header.as_object() else {
            continue;
        };
        for (tname, t) in map {
            if tname == "__metadata__" {
                continue;
            }
            let stored = t["data_offsets"]
                .as_array()
                .and_then(|o| Some(o.get(1)?.as_u64()? - o.first()?.as_u64()?))
                .unwrap_or(0);
            let elems: u64 = t["shape"]
                .as_array()
                .map(|s| s.iter().filter_map(|d| d.as_u64()).product())
                .unwrap_or(0);
            let ndim = t["shape"].as_array().map_or(0, |s| s.len());
            if let Some((layer, rest)) = tname.split_once(".experts.") {
                layer_prefixes.insert(layer.to_string());
                expert_bytes += stored;
                let expert_id = rest.split('.').next().unwrap_or("").to_string();
                expert_keys.insert((layer.to_string(), expert_id));
                expert_int8 |= t["dtype"].as_str() == Some("I8");
            } else {
                if let Some((layer, _)) = tname.split_once(".mlp.") {
                    layer_prefixes.insert(layer.to_string());
                }
                // engine residency: embeddings + 1D (norms) f32; matrices int8
                dense_resident_bytes += if tname.contains("embed_tokens") || ndim < 2 {
                    elems as f64 * 4.0
                } else {
                    elems as f64
                };
            }
        }
    }

    let n_layers = cfg_u64(&config, "num_hidden_layers")
        .unwrap_or(layer_prefixes.len() as u64);
    let qcache = effective_qcache(&model_type, qcache);
    let est_ram_gb = if expert_int8 && !expert_keys.is_empty() {
        let per_expert = expert_bytes as f64 / expert_keys.len() as f64;
        (dense_resident_bytes + n_layers as f64 * qcache as f64 * per_expert) / 1e9
    } else {
        // non-streaming container (f32/bf16 experts, e.g. tiny oracle):
        // everything loads resident as f32
        shards
            .iter()
            .filter_map(|s| read_st_header(s))
            .filter_map(|h| {
                h.as_object().map(|m| {
                    m.iter()
                        .filter(|(k, _)| *k != "__metadata__")
                        .map(|(_, t)| {
                            t["shape"]
                                .as_array()
                                .map(|s| s.iter().filter_map(|d| d.as_u64()).product::<u64>())
                                .unwrap_or(0)
                        })
                        .sum::<u64>()
                })
            })
            .sum::<u64>() as f64
            * 4.0
            / 1e9
    };

    ModelInfo {
        name,
        model_type,
        disk_gb: disk_bytes as f64 / 1e9,
        est_ram_gb,
    }
}

/// Check a container's weights are actually loadable before we eject a
/// running engine for it: if model.safetensors.index.json exists, every
/// shard in its weight_map must be present with a parseable safetensors
/// header; otherwise at least one .safetensors with a parseable header.
/// Catches half-copied containers that would otherwise surface as an
/// opaque engine crash.
pub fn verify_container(dir: &Path) -> Result<(), String> {
    let index_path = dir.join("model.safetensors.index.json");
    if index_path.exists() {
        let raw = std::fs::read_to_string(&index_path)
            .map_err(|e| format!("reading {}: {e}", index_path.display()))?;
        let index: Value = serde_json::from_str(&raw)
            .map_err(|e| format!("parsing {}: {e}", index_path.display()))?;
        let Some(map) = index["weight_map"].as_object() else {
            return Err("index has no weight_map".to_string());
        };
        let shards: HashSet<&str> = map.values().filter_map(|v| v.as_str()).collect();
        for shard in shards {
            if read_st_header(&dir.join(shard)).is_none() {
                return Err(format!("shard {shard} missing or corrupt"));
            }
        }
        Ok(())
    } else {
        let has_valid = std::fs::read_dir(dir)
            .map_err(|e| format!("reading container dir: {e}"))?
            .flatten()
            .map(|f| f.path())
            .filter(|p| p.extension().is_some_and(|e| e == "safetensors"))
            .any(|p| read_st_header(&p).is_some());
        if has_valid {
            Ok(())
        } else {
            Err("no readable .safetensors weights".to_string())
        }
    }
}

/// safetensors header: u64 LE length prefix, then that many bytes of JSON.
fn read_st_header(path: &Path) -> Option<Value> {
    use std::io::Read;
    let mut f = std::fs::File::open(path).ok()?;
    let mut len8 = [0u8; 8];
    f.read_exact(&mut len8).ok()?;
    let len = u64::from_le_bytes(len8);
    if len == 0 || len > 256 * 1024 * 1024 {
        return None; // corrupt or not a safetensors file
    }
    let mut buf = vec![0u8; len as usize];
    f.read_exact(&mut buf).ok()?;
    serde_json::from_slice(&buf).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_st(path: &Path, header: &Value) {
        let json = serde_json::to_vec(header).unwrap();
        let mut f = std::fs::File::create(path).unwrap();
        f.write_all(&(json.len() as u64).to_le_bytes()).unwrap();
        f.write_all(&json).unwrap();
        // no tensor data needed; we only parse the header
    }

    #[test]
    fn scan_measures_streaming_container() {
        let dir = std::env::temp_dir().join(format!("mtest_{}", std::process::id()));
        let model = dir.join("tiny_i8");
        std::fs::create_dir_all(&model).unwrap();
        std::fs::write(
            model.join("config.json"),
            r#"{"model_type":"qwen3_5_moe","num_hidden_layers":2}"#,
        )
        .unwrap();
        // 2 experts on layer 0: 100 int8 bytes each (+ shapes for elems)
        write_st(
            &model.join("model.safetensors"),
            &serde_json::json!({
                "model.embed_tokens.weight": {"dtype":"BF16","shape":[10,4],"data_offsets":[0,80]},
                "model.layers.0.mlp.experts.0.gate_proj.weight": {"dtype":"I8","shape":[10,10],"data_offsets":[80,180]},
                "model.layers.0.mlp.experts.1.gate_proj.weight": {"dtype":"I8","shape":[10,10],"data_offsets":[180,280]},
            }),
        );
        let models = scan(&dir, 4);
        assert_eq!(models.len(), 1);
        let m = &models[0];
        assert_eq!(m.name, "tiny_i8");
        assert_eq!(m.model_type, "qwen3_5_moe");
        assert!(m.disk_gb > 0.0);
        // dense: embed 40 elems * 4 = 160 B; experts avg 100 B, 2 layers * 4 slots
        let expect = (160.0 + 2.0 * 4.0 * 100.0) / 1e9;
        assert!((m.est_ram_gb - expect).abs() < 1e-12, "got {}", m.est_ram_gb);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    fn temp_container(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("vtest_{}_{tag}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), r#"{"model_type":"qwen3_5_moe"}"#).unwrap();
        dir
    }

    fn valid_header() -> Value {
        serde_json::json!({"t": {"dtype":"F32","shape":[1],"data_offsets":[0,4]}})
    }

    #[test]
    fn verify_container_no_index_ok() {
        let dir = temp_container("noindex");
        write_st(&dir.join("model.safetensors"), &valid_header());
        assert!(verify_container(&dir).is_ok());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn verify_container_no_weights() {
        let dir = temp_container("noweights");
        let err = verify_container(&dir).unwrap_err();
        assert!(err.contains("no readable"), "got: {err}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn verify_container_corrupt_header() {
        let dir = temp_container("corrupt");
        // zero length prefix = corrupt per read_st_header
        std::fs::write(dir.join("model.safetensors"), 0u64.to_le_bytes()).unwrap();
        assert!(verify_container(&dir).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn verify_container_index_missing_shard() {
        let dir = temp_container("index");
        std::fs::write(
            dir.join("model.safetensors.index.json"),
            r#"{"weight_map":{"w1":"a.safetensors","w2":"b.safetensors"}}"#,
        )
        .unwrap();
        write_st(&dir.join("a.safetensors"), &valid_header());
        let err = verify_container(&dir).unwrap_err();
        assert!(err.contains("b.safetensors"), "got: {err}");
        write_st(&dir.join("b.safetensors"), &valid_header());
        assert!(verify_container(&dir).is_ok());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
