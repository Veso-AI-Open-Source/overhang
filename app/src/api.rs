//! Daemon client for overhangd (http://127.0.0.1:11544).
//!
//! Runs a tokio runtime on a background thread. The UI thread reads a shared
//! `Shared` snapshot (Mutex) and sends `Cmd`s down a channel. This keeps the
//! view layer swappable: nothing here knows about the UI framework.

use futures_util::StreamExt;
use serde::Deserialize;
use serde_json::{json, Value};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Daemon address; override host:port with OVERHANG_ADDR (e.g. for tests).
pub fn addr() -> String {
    std::env::var("OVERHANG_ADDR").unwrap_or_else(|_| "127.0.0.1:11544".into())
}
fn base() -> String {
    format!("http://{}", addr())
}
fn ws_events() -> String {
    format!("ws://{}/events", addr())
}

// ---------- shared state (UI reads, backend writes) ----------

#[derive(Clone, Debug)]
pub struct ChatMsg {
    pub role: String, // "user" | "assistant"
    pub content: String,
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct ModelRow {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub disk_gb: f64,
    #[serde(default)]
    pub ram_gb: f64,
    #[serde(default)]
    pub fits: bool,
    #[serde(default)]
    pub active: bool,
}

#[derive(Clone, Debug, Default)]
pub struct Capacity {
    pub machine_ram_gb: f64,
    pub disk_free_gb: f64,
    pub engine_model: Option<String>, // active engine, e.g. "qwen36_i8"
    pub engine_up: bool,
    pub models: Vec<ModelRow>,
}

/// GET /system: hardware discovered by the daemon (never hardcoded).
#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
pub struct SystemInfo {
    #[serde(default)]
    pub chip: String,
    #[serde(default)]
    pub arch: String,
    #[serde(default)]
    pub os_name: String,
    #[serde(default)]
    pub os_version: String,
    #[serde(default)]
    pub kernel: String,
    #[serde(default)]
    pub logical_cores: u32,
    #[serde(default)]
    pub physical_cores: Option<u32>,
    #[serde(default)]
    pub perf_cores: Option<u32>,
    #[serde(default)]
    pub eff_cores: Option<u32>,
    #[serde(default)]
    pub total_ram_gb: f64,
    #[serde(default)]
    pub gpu_cores: Option<u32>,
    #[serde(default)]
    pub metal4: bool,
    #[serde(default)]
    pub unified_memory: bool,
    #[serde(default)]
    pub model_volume_free_gb: f64,
    #[serde(default)]
    pub model_volume_total_gb: f64,
    #[serde(default)]
    pub model_root: String,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct LiveStats {
    pub tok_s: f64,
    pub hit_rate: f64,
    pub resident_gb: f64,
    pub streamed_mb_s: f64,
}

#[derive(Default)]
pub struct Shared {
    pub daemon_up: bool,
    pub last_error: Option<String>,
    pub capacity: Option<Capacity>,
    pub messages: Vec<ChatMsg>,
    pub generating: bool,
    pub gen_tok_s: f64, // live tok/s of the current generation (SSE-derived)
    pub stats: LiveStats,
    pub tok_s_history: Vec<f32>, // for the sparkline (ws /events)
    pub events_connected: bool,
    pub eject_unsupported: bool, // POST /engine/eject returned 404 (older daemon)
    pub load_unsupported: bool,  // POST /engine/load returned 404 (older daemon)
    pub loading_model: Option<String>, // model name while a load is in flight
    pub load_error: Option<(String, String)>, // (model, message) from the last failed load
    pub system: Option<SystemInfo>, // GET /system discovery report
    pub system_unsupported: bool, // GET /system returned 404 (older daemon)
}

pub enum Cmd {
    SendChat(String),
    RefreshStatus,
    Eject,
    Load(String),
}

#[derive(Clone)]
pub struct Client {
    pub shared: Arc<Mutex<Shared>>,
    tx: Sender<Cmd>,
}

impl Client {
    pub fn send(&self, cmd: Cmd) {
        let _ = self.tx.send(cmd);
    }
}

// ---------- backend ----------

/// `wake` is called whenever shared state changes, so the UI can repaint.
pub fn start(wake: impl Fn() + Send + Sync + 'static) -> Client {
    let shared = Arc::new(Mutex::new(Shared::default()));
    let (tx, rx) = channel::<Cmd>();
    let client = Client { shared: shared.clone(), tx };
    let wake: Arc<dyn Fn() + Send + Sync> = Arc::new(wake);

    std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .expect("tokio runtime");
        rt.block_on(backend(shared, rx, wake));
    });
    client
}

async fn backend(shared: Arc<Mutex<Shared>>, rx: Receiver<Cmd>, wake: Arc<dyn Fn() + Send + Sync>) {
    let http = reqwest::Client::builder()
        .timeout(Duration::from_secs(300))
        .connect_timeout(Duration::from_secs(2))
        .build()
        .unwrap();

    // events websocket: reconnect forever in the background
    tokio::spawn(events_loop(shared.clone(), wake.clone()));

    // initial status probe
    fetch_status(&http, &shared, &wake).await;
    fetch_system(&http, &shared, &wake).await;

    // command loop (std channel; poll it from async without blocking the rt).
    // Between commands, re-poll /status so daemon-side changes made by other
    // clients (curl, another app) show up without the manual refresh button.
    const STATUS_POLL: Duration = Duration::from_secs(2);
    let mut last_status = std::time::Instant::now();
    loop {
        let cmd = tokio::task::block_in_place(|| rx.recv_timeout(Duration::from_millis(200)));
        let fetched = match cmd {
            Ok(Cmd::RefreshStatus) => {
                {
                    // explicit refresh: re-probe endpoints the daemon may have gained
                    let mut s = shared.lock().unwrap();
                    s.eject_unsupported = false;
                    s.load_unsupported = false;
                    s.system_unsupported = false;
                    s.load_error = None;
                }
                fetch_status(&http, &shared, &wake).await;
                fetch_system(&http, &shared, &wake).await;
                true
            }
            Ok(Cmd::SendChat(text)) => {
                run_chat(&http, &shared, &wake, text).await;
                false
            }
            Ok(Cmd::Eject) => {
                eject(&http, &shared, &wake).await;
                true
            }
            Ok(Cmd::Load(name)) => {
                load(&http, &shared, &wake, name).await;
                true
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                if last_status.elapsed() >= STATUS_POLL {
                    fetch_status(&http, &shared, &wake).await;
                    true
                } else {
                    false
                }
            }
            Err(_) => break,
        };
        if fetched {
            last_status = std::time::Instant::now();
        }
    }
}

/// POST /engine/eject: kill the engine child, free RAM. 404 => older daemon.
async fn eject(
    http: &reqwest::Client,
    shared: &Arc<Mutex<Shared>>,
    wake: &Arc<dyn Fn() + Send + Sync>,
) {
    let res = http.post(format!("{}/engine/eject", base())).send().await;
    {
        let mut s = shared.lock().unwrap();
        match &res {
            Ok(r) if r.status().as_u16() == 404 => s.eject_unsupported = true,
            Ok(_) => {}
            Err(e) => {
                s.daemon_up = false;
                s.last_error = Some(e.to_string());
            }
        }
    }
    wake();
    fetch_status(http, shared, wake).await; // reconcile engine state
}

/// POST /engine/load {"model": name}: spin up the engine on a container.
/// 404 => older daemon without the endpoint.
async fn load(
    http: &reqwest::Client,
    shared: &Arc<Mutex<Shared>>,
    wake: &Arc<dyn Fn() + Send + Sync>,
    name: String,
) {
    shared.lock().unwrap().loading_model = Some(name.clone());
    wake();
    let res = http
        .post(format!("{}/engine/load", base()))
        .json(&json!({ "model": name }))
        .send()
        .await;
    // Classify before locking: reading a failed response's body is async.
    let mut unsupported = false;
    let mut daemon_err = None;
    let mut load_err = None;
    match res {
        Ok(r) if r.status().as_u16() == 404 => unsupported = true,
        Ok(r) if !r.status().is_success() => {
            // daemon error envelope: {"error":{"message":...}}; fall back to raw body
            let body = r.text().await.unwrap_or_default();
            let msg = serde_json::from_str::<Value>(&body)
                .ok()
                .and_then(|v| v["error"]["message"].as_str().map(String::from))
                .unwrap_or(body);
            load_err = Some((name.clone(), msg));
        }
        Ok(_) => {}
        Err(e) => daemon_err = Some(e.to_string()),
    }
    {
        let mut s = shared.lock().unwrap();
        if unsupported {
            s.load_unsupported = true;
        }
        if let Some(e) = daemon_err {
            s.daemon_up = false;
            s.last_error = Some(e);
        }
        s.load_error = load_err; // success clears any previous failure
        s.loading_model = None;
    }
    wake();
    fetch_status(http, shared, wake).await; // reconcile engine state
}

async fn fetch_status(
    http: &reqwest::Client,
    shared: &Arc<Mutex<Shared>>,
    wake: &Arc<dyn Fn() + Send + Sync>,
) {
    let res = async {
        http.get(format!("{}/status", base())).send().await?.json::<Value>().await
    }
    .await;
    {
        let mut s = shared.lock().unwrap();
        match res {
            Ok(v) => {
                s.daemon_up = true;
                s.last_error = None;
                s.capacity = Some(parse_capacity(&v));
            }
            Err(e) => {
                s.daemon_up = false;
                s.last_error = Some(e.to_string());
            }
        }
    }
    wake();
}

/// GET /system: discovery report. 404 => older daemon without the endpoint.
async fn fetch_system(
    http: &reqwest::Client,
    shared: &Arc<Mutex<Shared>>,
    wake: &Arc<dyn Fn() + Send + Sync>,
) {
    if shared.lock().unwrap().system_unsupported {
        return;
    }
    let res = http.get(format!("{}/system", base())).send().await;
    let outcome = match res {
        Ok(r) if r.status().as_u16() == 404 => Some(Err(())),
        Ok(r) => Some(Ok(r.json::<SystemInfo>().await)),
        Err(_) => None, // /status already tracked daemon_up
    };
    {
        let mut s = shared.lock().unwrap();
        match outcome {
            Some(Err(())) => s.system_unsupported = true,
            Some(Ok(Ok(info))) => s.system = Some(info),
            Some(Ok(Err(e))) => s.last_error = Some(e.to_string()),
            None => {}
        }
    }
    wake();
}

/// overhangd /status: {"capacity":{"free_disk_gb":..,"total_ram_gb":..,
///   "ladder":[{"name":..,"disk_gb":..,"ram_gb":..,"fits_disk":..,"fits_ram":..}]},
///   "engine":{"model":"..","up":true}}
fn parse_capacity(v: &Value) -> Capacity {
    let c = &v["capacity"];
    let mut cap = Capacity {
        machine_ram_gb: c["total_ram_gb"].as_f64().or(v["ram_gb"].as_f64()).unwrap_or(0.0),
        disk_free_gb: c["free_disk_gb"].as_f64().or(v["disk_free_gb"].as_f64()).unwrap_or(0.0),
        engine_model: v["engine"]["model"].as_str().map(String::from),
        engine_up: v["engine"]["up"].as_bool().unwrap_or(false),
        models: vec![],
    };
    let list = c["ladder"].as_array().or(v["models"].as_array());
    if let Some(list) = list {
        for m in list {
            let fits = m["fits"].as_bool().unwrap_or_else(|| {
                m["fits_disk"].as_bool().unwrap_or(false) && m["fits_ram"].as_bool().unwrap_or(false)
            });
            let name = m["name"].as_str().or(m["id"].as_str()).unwrap_or("?").to_string();
            cap.models.push(ModelRow {
                active: m["active"].as_bool().unwrap_or(false)
                    || cap.engine_model.as_deref() == Some(name.as_str()),
                name,
                disk_gb: m["disk_gb"].as_f64().unwrap_or(0.0),
                ram_gb: m["ram_gb"].as_f64().unwrap_or(0.0),
                fits,
            });
        }
    }
    cap
}

async fn run_chat(
    http: &reqwest::Client,
    shared: &Arc<Mutex<Shared>>,
    wake: &Arc<dyn Fn() + Send + Sync>,
    text: String,
) {
    let history: Vec<Value> = {
        let mut s = shared.lock().unwrap();
        s.messages.push(ChatMsg { role: "user".into(), content: text });
        s.messages.push(ChatMsg { role: "assistant".into(), content: String::new() });
        s.generating = true;
        s.gen_tok_s = 0.0;
        s.stats.tok_s = 0.0; // don't show the previous request's rate as "current"
        s.messages[..s.messages.len() - 1]
            .iter()
            .map(|m| json!({"role": m.role, "content": m.content}))
            .collect()
    };
    wake();

    let body = json!({"messages": history, "stream": true});
    let res = http
        .post(format!("{}/v1/chat/completions", base()))
        .json(&body)
        .send()
        .await;

    let fail = |msg: String| {
        let mut s = shared.lock().unwrap();
        s.generating = false;
        if let Some(last) = s.messages.last_mut() {
            last.content = format!("[error: {msg}]");
        }
        s.last_error = Some(msg);
        drop(s);
        wake();
    };

    let resp = match res {
        Ok(r) if r.status().is_success() => r,
        Ok(r) => return fail(format!("HTTP {}", r.status())),
        Err(e) => {
            shared.lock().unwrap().daemon_up = false;
            return fail(e.to_string());
        }
    };

    // SSE parse: lines "data: {json}" separated by blank lines; "[DONE]" ends.
    let started = Instant::now();
    let mut first_token = started; // re-stamped on the first real delta
    let mut n_tokens: u64 = 0;
    let mut buf = String::new();
    let mut stream = resp.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = match chunk {
            Ok(c) => c,
            Err(e) => return fail(e.to_string()),
        };
        buf.push_str(&String::from_utf8_lossy(&chunk));
        while let Some(pos) = buf.find('\n') {
            let line = buf[..pos].trim().to_string();
            buf.drain(..=pos);
            let Some(data) = line.strip_prefix("data:") else { continue };
            let data = data.trim();
            if data == "[DONE]" {
                let mut s = shared.lock().unwrap();
                s.generating = false;
                drop(s);
                wake();
                return;
            }
            if let Ok(v) = serde_json::from_str::<Value>(data) {
                if let Some(delta) = v["choices"][0]["delta"]["content"].as_str() {
                    // the stream opens with an empty role-priming delta —
                    // counting it produced a bogus "1 token in ~1ms" spike
                    if delta.is_empty() {
                        continue;
                    }
                    n_tokens += 1;
                    if n_tokens == 1 {
                        first_token = std::time::Instant::now();
                    }
                    let mut s = shared.lock().unwrap();
                    if let Some(last) = s.messages.last_mut() {
                        last.content.push_str(delta);
                    }
                    // running average since the first real token; 0 (= shown
                    // as "--.-") until there are two points
                    let dt = first_token.elapsed().as_secs_f64();
                    s.gen_tok_s = if n_tokens > 1 && dt > 0.0 {
                        (n_tokens - 1) as f64 / dt
                    } else {
                        0.0
                    };
                    drop(s);
                    wake();
                }
            }
        }
    }
    shared.lock().unwrap().generating = false;
    wake();
}

async fn events_loop(shared: Arc<Mutex<Shared>>, wake: Arc<dyn Fn() + Send + Sync>) {
    loop {
        match tokio_tungstenite::connect_async(ws_events()).await {
            Ok((mut ws, _)) => {
                {
                    let mut s = shared.lock().unwrap();
                    s.events_connected = true;
                    s.daemon_up = true;
                }
                wake();
                while let Some(Ok(msg)) = ws.next().await {
                    if let tokio_tungstenite::tungstenite::Message::Text(txt) = msg {
                        if let Ok(v) = serde_json::from_str::<Value>(&txt) {
                            let mut s = shared.lock().unwrap();
                            s.stats = LiveStats {
                                tok_s: v["tok_s"].as_f64().unwrap_or(0.0),
                                hit_rate: v["hit_rate"].as_f64().unwrap_or(0.0),
                                resident_gb: v["resident_gb"].as_f64().unwrap_or(0.0),
                                streamed_mb_s: v["streamed_mb_s"].as_f64().unwrap_or(0.0),
                            };
                            let t = s.stats.tok_s as f32;
                            s.tok_s_history.push(t);
                            let len = s.tok_s_history.len();
                            if len > 240 {
                                s.tok_s_history.drain(..len - 240);
                            }
                            drop(s);
                            wake();
                        }
                    }
                }
            }
            Err(_) => {}
        }
        {
            let mut s = shared.lock().unwrap();
            s.events_connected = false;
        }
        wake();
        tokio::time::sleep(Duration::from_secs(2)).await;
    }
}
