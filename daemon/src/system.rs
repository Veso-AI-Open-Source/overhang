//! System discovery: everything the daemon reports about the machine is
//! measured here, never hardcoded. Static facts (chip, cores, GPU) are
//! discovered once at startup; volatile ones (disk, available RAM) per call.

use serde::Serialize;
use std::path::Path;
use std::sync::OnceLock;

#[derive(Debug, Clone, Serialize)]
pub struct SystemInfo {
    pub chip: String,
    pub arch: &'static str,
    pub os_name: String,
    pub os_version: String,
    pub kernel: String,
    pub logical_cores: usize,
    pub physical_cores: Option<usize>,
    /// Performance/efficiency core split (Apple Silicon; None elsewhere).
    pub perf_cores: Option<u32>,
    pub eff_cores: Option<u32>,
    pub total_ram_gb: f64,
    /// GPU core count from IORegistry (Apple Silicon; None elsewhere).
    pub gpu_cores: Option<u32>,
    /// Metal 4 tensor ops need macOS 26+ on Apple Silicon.
    pub metal4: bool,
    pub unified_memory: bool,
}

fn sysctl(key: &str) -> Option<String> {
    if !cfg!(target_os = "macos") {
        return None;
    }
    let out = std::process::Command::new("sysctl")
        .args(["-n", key])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    (!s.is_empty()).then_some(s)
}

/// GPU core count via IORegistry (AGXAccelerator carries "gpu-core-count").
fn gpu_core_count() -> Option<u32> {
    if !cfg!(target_os = "macos") {
        return None;
    }
    let out = std::process::Command::new("ioreg")
        .args(["-rc", "AGXAccelerator", "-d", "1"])
        .output()
        .ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    for line in text.lines() {
        if let Some((_, v)) = line.split_once("\"gpu-core-count\"") {
            if let Some(n) = v.split('=').nth(1) {
                if let Ok(n) = n.trim().parse::<u32>() {
                    return Some(n);
                }
            }
        }
    }
    None
}

fn discover() -> SystemInfo {
    let mut sys = sysinfo::System::new();
    sys.refresh_memory();
    sys.refresh_cpu_all();

    let os_version = sysinfo::System::os_version().unwrap_or_default();
    let os_major: u32 = os_version
        .split('.')
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let apple_silicon = cfg!(target_os = "macos") && std::env::consts::ARCH == "aarch64";
    let chip = sysctl("machdep.cpu.brand_string")
        .or_else(|| sys.cpus().first().map(|c| c.brand().to_string()))
        .unwrap_or_else(|| "unknown".into());

    SystemInfo {
        chip,
        arch: std::env::consts::ARCH,
        os_name: sysinfo::System::name().unwrap_or_default(),
        os_version,
        kernel: sysinfo::System::kernel_version().unwrap_or_default(),
        logical_cores: sys.cpus().len(),
        physical_cores: sys.physical_core_count(),
        perf_cores: sysctl("hw.perflevel0.logicalcpu").and_then(|s| s.parse().ok()),
        eff_cores: sysctl("hw.perflevel1.logicalcpu").and_then(|s| s.parse().ok()),
        total_ram_gb: sys.total_memory() as f64 / 1e9,
        gpu_cores: gpu_core_count(),
        metal4: apple_silicon && os_major >= 26,
        unified_memory: apple_silicon,
    }
}

/// Discovered once, cached for the daemon's lifetime.
pub fn info() -> &'static SystemInfo {
    static INFO: OnceLock<SystemInfo> = OnceLock::new();
    INFO.get_or_init(discover)
}

/// (free_gb, total_gb) of the volume holding `path` (longest matching mount).
pub fn disk_for(path: &Path) -> (f64, f64) {
    let path = std::fs::canonicalize(path)
        .or_else(|_| std::env::current_dir())
        .unwrap_or_default();
    let disks = sysinfo::Disks::new_with_refreshed_list();
    let mut best: Option<(usize, u64, u64)> = None;
    for d in disks.list() {
        let mount = d.mount_point();
        if path.starts_with(mount) {
            let len = mount.as_os_str().len();
            if best.is_none_or(|(l, _, _)| len > l) {
                best = Some((len, d.available_space(), d.total_space()));
            }
        }
    }
    best.map(|(_, free, total)| (free as f64 / 1e9, total as f64 / 1e9))
        .unwrap_or((0.0, 0.0))
}
