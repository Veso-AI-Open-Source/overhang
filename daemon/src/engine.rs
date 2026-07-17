use crate::config::Config;
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, Lines};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::sync::broadcast;

/// Per-token event published on /events.
#[derive(Debug, Clone, Serialize)]
pub struct TokenEvent {
    /// Decode tokens/sec, running average since the FIRST token of the
    /// request. Instantaneous 1/dt was meaningless: pipe lines arrive
    /// batched, so dt ~ 1 ms produced bogus 1000 tok/s spikes in the UI.
    pub tok_s: f64,
    /// Cache hit rate from the last known `done` line (None until first done).
    pub hit_rate: Option<f64>,
    /// Tokens generated so far in the current request.
    pub n_out: u64,
}

/// Status shared with HTTP handlers without holding the engine mutex.
#[derive(Default)]
pub struct SharedStatus {
    pub up: AtomicBool,
    pub model: std::sync::Mutex<Option<String>>,
    pub last_hit: std::sync::Mutex<Option<f64>>,
}

/// Error reported by the engine for one request (process stays alive).
#[derive(Debug)]
pub struct EngineRequestError(pub String);

impl std::fmt::Display for EngineRequestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "engine error: {}", self.0)
    }
}

impl std::error::Error for EngineRequestError {}

#[derive(Debug, Deserialize)]
struct ReadyLine {
    ready: bool,
    model: String,
}

#[derive(Debug, Deserialize)]
struct RespLine {
    id: u64,
    #[serde(default)]
    tok: Option<u32>,
    #[serde(default)]
    done: Option<bool>,
    #[serde(default)]
    error: Option<String>,
    #[serde(default)]
    n_out: Option<u64>,
    #[serde(default)]
    prefill_s: Option<f64>,
    #[serde(default)]
    decode_s: Option<f64>,
    #[serde(default)]
    hit: Option<f64>,
}

#[derive(Serialize)]
struct RequestLine<'a> {
    id: u64,
    ids: &'a [u32],
    n: u64,
    reset: bool,
}

#[derive(Debug, Clone)]
pub struct DoneStats {
    pub n_out: u64,
    pub prefill_s: Option<f64>,
    pub decode_s: Option<f64>,
    pub hit: Option<f64>,
}

struct Proc {
    child: Child,
    stdin: ChildStdin,
    lines: Lines<BufReader<ChildStdout>>,
}

pub struct Engine {
    cfg: Arc<Config>,
    status: Arc<SharedStatus>,
    proc: Option<Proc>,
    next_id: u64,
    /// Currently selected container + engine binary; /engine/load swaps them.
    model_dir: std::path::PathBuf,
    engine_bin: String,
    /// Token ids currently represented in the engine's state (last prompt +
    /// its output). When a new request's ids extend this exactly, generate()
    /// sends reset:false with only the suffix — per-turn prefill becomes
    /// O(new tokens) instead of O(entire history).
    history: Vec<u32>,
}

impl Engine {
    pub fn new(cfg: Arc<Config>, status: Arc<SharedStatus>) -> Self {
        Self {
            model_dir: cfg.model_dir.clone(),
            engine_bin: cfg.engine_bin.clone(),
            cfg,
            status,
            proc: None,
            next_id: 0,
            history: Vec::new(),
        }
    }

    /// Switch to another container: eject the current engine, spawn eagerly
    /// on the new one so failures surface here (not on the next chat).
    /// On spawn failure the previous container selection is restored, so the
    /// next request lazily respawns the previous model (no eager re-spawn:
    /// that's seconds of RAM churn on a path where the caller is likely about
    /// to retry a different container anyway).
    /// Returns the engine-reported model name.
    pub async fn load(
        &mut self,
        model_dir: std::path::PathBuf,
        engine_bin: String,
    ) -> Result<String> {
        let prev = (self.model_dir.clone(), self.engine_bin.clone());
        self.eject().await;
        self.model_dir = model_dir;
        self.engine_bin = engine_bin;
        if let Err(e) = self.spawn().await {
            (self.model_dir, self.engine_bin) = prev;
            tracing::warn!("load failed, reverted to previous container: {e:#}");
            return Err(e);
        }
        Ok(self
            .status
            .model
            .lock()
            .unwrap()
            .clone()
            .unwrap_or_default())
    }

    async fn spawn(&mut self) -> Result<()> {
        let mut cmd = if let Some(cmdline) = &self.cfg.engine_cmd {
            let mut parts = cmdline.split_whitespace();
            let prog = parts.next().context("empty engine_cmd")?;
            let mut c = Command::new(prog);
            c.args(parts);
            c
        } else {
            let mut c = Command::new(&self.engine_bin);
            c.arg("--serve");
            c
        };
        cmd.env("SNAP", &self.model_dir)
            .env("QCACHE", self.cfg.qcache.to_string())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        tracing::info!("spawning engine: {:?}", cmd.as_std());
        let mut child = cmd.spawn().context("failed to spawn engine")?;

        let stdin = child.stdin.take().expect("piped stdin");
        let stdout = child.stdout.take().expect("piped stdout");
        let stderr = child.stderr.take().expect("piped stderr");

        // Forward engine stderr (stats/logs) to tracing.
        tokio::spawn(async move {
            let mut lines = BufReader::new(stderr).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                tracing::info!(target: "engine", "{line}");
            }
        });

        let mut lines = BufReader::new(stdout).lines();
        let ready_line = tokio::time::timeout(Duration::from_secs(300), lines.next_line())
            .await
            .context("timed out waiting for engine ready line")?
            .context("reading engine ready line")?
            .context("engine exited before printing ready line")?;
        let ready: ReadyLine = serde_json::from_str(&ready_line)
            .with_context(|| format!("bad engine ready line: {ready_line}"))?;
        if !ready.ready {
            bail!("engine reported ready:false");
        }
        tracing::info!("engine ready, model={}", ready.model);

        self.history.clear(); // fresh process = empty engine state
        *self.status.model.lock().unwrap() = Some(ready.model);
        self.status.up.store(true, Ordering::SeqCst);
        self.proc = Some(Proc {
            child,
            stdin,
            lines,
        });
        Ok(())
    }

    /// Eject: kill the engine child and wait for it to exit so its memory is
    /// actually released before anything else (notably a swap-in spawn)
    /// starts allocating. The next request cold-starts it again (lazy
    /// respawn, reset:true as always).
    pub async fn eject(&mut self) -> bool {
        let had = self.proc.is_some();
        if let Some(mut p) = self.proc.take() {
            let _ = p.child.start_kill();
            // SIGKILL can't be caught; wait() returning is the kernel
            // confirming teardown. No timeout — one would silently
            // reintroduce the double-residency this exists to prevent.
            let _ = p.child.wait().await;
        }
        self.status.up.store(false, Ordering::SeqCst);
        *self.status.model.lock().unwrap() = None;
        self.history.clear();
        if had { tracing::info!("engine ejected"); }
        had
    }

    fn mark_down(&mut self) {
        // Drop relies on kill_on_drop; no wait needed — the child has almost
        // always already died here, and respawn is lazy (next request),
        // unlike the eject-then-spawn in load where double-residency bites.
        // Process state is lost: the next request respawns lazily and must
        // send reset:true with the full history.
        self.proc = None;
        self.status.up.store(false, Ordering::SeqCst);
        self.history.clear();
        tracing::warn!("engine marked down; will restart lazily on next request");
    }

    /// Run one generation. Serial by construction: callers hold the engine
    /// mutex. `on_tok` is called synchronously per generated token.
    ///
    /// Warm-append: when the new ids extend the ids already in the engine's
    /// state (previous prompt + output), only the suffix is sent with
    /// reset:false. If the engine refuses a warm continuation before emitting
    /// anything (e.g. "context full"), one cold retry with the full ids runs.
    pub async fn generate(
        &mut self,
        ids: &[u32],
        n: u64,
        events: &broadcast::Sender<TokenEvent>,
        mut on_tok: impl FnMut(u32),
    ) -> Result<(Vec<u32>, DoneStats)> {
        if self.proc.is_none() {
            self.spawn().await?; // clears history: fresh state
        }
        let warm = !self.history.is_empty()
            && ids.len() > self.history.len()
            && ids[..self.history.len()] == self.history[..];
        let suffix_start = if warm { self.history.len() } else { 0 };
        if warm {
            tracing::debug!(reused = suffix_start, sent = ids.len() - suffix_start, "warm-append");
        }
        let mut emitted = 0usize;
        let mut res = self
            .generate_once(&ids[suffix_start..], !warm, n, events, &mut |t| {
                emitted += 1;
                on_tok(t);
            })
            .await;
        if warm && emitted == 0 {
            if let Err(e) = &res {
                if e.is::<EngineRequestError>() {
                    // warm continuation refused before any token: cold retry
                    tracing::info!("warm-append refused ({e}); retrying with full reset");
                    self.history.clear();
                    res = self.generate_once(ids, true, n, events, &mut on_tok).await;
                }
            }
        }
        match &res {
            Ok((out, _)) => {
                self.history.clear();
                self.history.extend_from_slice(ids);
                self.history.extend_from_slice(out);
            }
            // engine state is indeterminate after a failed request
            Err(_) => self.history.clear(),
        }
        res
    }

    async fn generate_once(
        &mut self,
        ids: &[u32],
        reset: bool,
        n: u64,
        events: &broadcast::Sender<TokenEvent>,
        on_tok: &mut impl FnMut(u32),
    ) -> Result<(Vec<u32>, DoneStats)> {
        self.next_id += 1;
        let id = self.next_id;
        let req = serde_json::to_string(&RequestLine { id, ids, n, reset })? + "\n";
        let last_hit = *self.status.last_hit.lock().unwrap();

        let res: Result<(Vec<u32>, DoneStats)> = async {
            let proc = self.proc.as_mut().expect("ensured above");
            proc.stdin.write_all(req.as_bytes()).await?;
            proc.stdin.flush().await?;

            let mut out: Vec<u32> = Vec::new();
            let mut first_t: Option<Instant> = None;
            loop {
                let Some(line) = proc.lines.next_line().await? else {
                    bail!("engine closed stdout (process died)");
                };
                let resp: RespLine = serde_json::from_str(&line)
                    .with_context(|| format!("bad engine response line: {line}"))?;
                if resp.id != id {
                    tracing::warn!("engine line for stale id {} (expected {id})", resp.id);
                    continue;
                }
                if let Some(err) = resp.error {
                    return Err(EngineRequestError(err).into());
                }
                if let Some(t) = resp.tok {
                    let first = *first_t.get_or_insert_with(Instant::now);
                    out.push(t);
                    // running average from the first token (0 until there are
                    // two points; smooth and truthful from then on)
                    let dt = first.elapsed().as_secs_f64();
                    let _ = events.send(TokenEvent {
                        tok_s: if out.len() > 1 && dt > 0.0 {
                            (out.len() - 1) as f64 / dt
                        } else {
                            0.0
                        },
                        hit_rate: last_hit,
                        n_out: out.len() as u64,
                    });
                    on_tok(t);
                }
                if resp.done == Some(true) {
                    let n_out = resp.n_out.unwrap_or(out.len() as u64);
                    let stats = DoneStats {
                        n_out,
                        prefill_s: resp.prefill_s,
                        decode_s: resp.decode_s,
                        hit: resp.hit,
                    };
                    return Ok((out, stats));
                }
            }
        }
        .await;

        match &res {
            Ok((_, stats)) => {
                tracing::info!(
                    n_out = stats.n_out,
                    prefill_s = stats.prefill_s,
                    decode_s = stats.decode_s,
                    hit = stats.hit,
                    "generation done"
                );
                if let Some(h) = stats.hit {
                    *self.status.last_hit.lock().unwrap() = Some(h);
                }
            }
            Err(e) => {
                // Per-request engine errors keep the process alive; anything
                // else (I/O, EOF, garbage) means the child is unusable.
                if !e.is::<EngineRequestError>() {
                    self.mark_down();
                }
            }
        }
        res
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use std::path::PathBuf;

    /// Engine driven by the python mock (cwd during `cargo test` is daemon/).
    /// The mock exits before the ready line for containers named *_bad.
    fn test_engine() -> (Engine, Arc<SharedStatus>) {
        let cfg = Arc::new(Config {
            engine_bin: "unused".into(),
            engine_cmd: Some("python3 tests/mock_engine.py".into()),
            model_dir: PathBuf::from(GOOD),
            qcache: 96,
            port: 0,
            tokenizer: PathBuf::from("tests/fixtures/tokenizer.json"),
            engines: Default::default(),
        });
        let status = Arc::new(SharedStatus::default());
        (Engine::new(cfg, status.clone()), status)
    }

    const GOOD: &str = "tests/fixtures/models/mock";

    #[tokio::test]
    async fn load_failure_rolls_back_container() {
        let (mut e, status) = test_engine();
        let name = e.load(PathBuf::from(GOOD), "unused".into()).await.unwrap();
        assert_eq!(name, "mock");
        assert!(e.load(PathBuf::from("container_bad"), "unused".into()).await.is_err());
        assert_eq!(e.model_dir, PathBuf::from(GOOD), "selection must revert");
        assert!(!status.up.load(Ordering::SeqCst));
        // Lazy respawn on the rolled-back container serves the next request.
        let (tx, _rx) = broadcast::channel(16);
        let (out, _stats) = e.generate(&[1, 2, 3], 4, &tx, |_| {}).await.unwrap();
        assert_eq!(out.len(), 3);
        assert_eq!(status.model.lock().unwrap().as_deref(), Some("mock"));
    }

    #[tokio::test]
    async fn load_failure_with_no_prior_engine() {
        let (mut e, _status) = test_engine();
        assert!(e.load(PathBuf::from("container_bad"), "unused".into()).await.is_err());
        assert_eq!(e.model_dir, PathBuf::from(GOOD), "reverts to config default");
        let name = e.load(PathBuf::from(GOOD), "unused".into()).await.unwrap();
        assert_eq!(name, "mock");
    }

    #[tokio::test]
    async fn warm_append_reuses_prefix() {
        let (mut e, _status) = test_engine();
        let (tx, _rx) = broadcast::channel(16);
        // cold start: full ids, mock emits [9906, 11, 220]
        e.generate(&[1, 2, 3], 4, &tx, |_| {}).await.unwrap();
        assert_eq!(e.history, vec![1, 2, 3, 9906, 11, 220]);
        // extending the history exactly -> warm append of just the suffix
        let mut ids = e.history.clone();
        ids.extend([7, 8]);
        let (out, _) = e.generate(&ids, 4, &tx, |_| {}).await.unwrap();
        assert_eq!(out.len(), 3);
        assert_eq!(e.history.len(), ids.len() + 3);
        assert_eq!(&e.history[..ids.len()], &ids[..]);
        // unrelated ids -> cold reset, history rebuilt from scratch
        e.generate(&[42], 4, &tx, |_| {}).await.unwrap();
        assert_eq!(e.history, vec![42, 9906, 11, 220]);
        // eject clears engine state tracking
        e.eject().await;
        assert!(e.history.is_empty());
    }

    #[tokio::test]
    async fn eject_waits_for_exit() {
        let (mut e, status) = test_engine();
        e.load(PathBuf::from(GOOD), "unused".into()).await.unwrap();
        assert!(e.eject().await);
        assert!(e.proc.is_none());
        assert!(!status.up.load(Ordering::SeqCst));
        assert!(!e.eject().await, "second eject is a no-op");
    }
}
