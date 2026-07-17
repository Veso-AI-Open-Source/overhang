use crate::chat::{strip_stop, ChatCtx, StreamDecoder, Tok};
use crate::config::Config;
use crate::engine::{Engine, SharedStatus, TokenEvent};
use crate::{models, system};
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::{json, Value};
use std::convert::Infallible;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::{broadcast, mpsc, Mutex};
use tokio_stream::wrappers::UnboundedReceiverStream;

const REQUEST_TIMEOUT: Duration = Duration::from_secs(600);

/// A model needs headroom next to the OS and other apps; require the RAM
/// estimate to fit in this fraction of total RAM.
const RAM_FIT_FRACTION: f64 = 0.9;

#[derive(Clone)]
pub struct AppState {
    pub cfg: Arc<Config>,
    pub engine: Arc<Mutex<Engine>>,
    /// Tokenizer + chat template; swapped when /engine/load switches containers.
    pub tok: Arc<std::sync::RwLock<ChatCtx>>,
    pub events: broadcast::Sender<TokenEvent>,
    pub status: Arc<SharedStatus>,
    pub req_counter: Arc<AtomicU64>,
}

impl AppState {
    fn chat(&self) -> ChatCtx {
        self.tok.read().unwrap().clone()
    }
}

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/models", get(models_list))
        .route("/status", get(status))
        .route("/system", get(system_info))
        .route("/events", get(events_ws))
        .route("/engine/eject", post(eject))
        .route("/engine/load", post(load))
        .with_state(state)
}

#[derive(Debug, Deserialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct ChatRequest {
    #[serde(default)]
    model: Option<String>,
    messages: Vec<ChatMessage>,
    #[serde(default)]
    max_tokens: Option<u64>,
    #[serde(default)]
    stream: Option<bool>,
}

fn error_response(code: StatusCode, msg: impl std::fmt::Display) -> Response {
    (
        code,
        Json(json!({"error": {"message": msg.to_string(), "type": "overhangd_error"}})),
    )
        .into_response()
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

async fn chat_completions(State(st): State<AppState>, Json(req): Json<ChatRequest>) -> Response {
    let max_tokens = req.max_tokens.unwrap_or(256);
    let messages: Vec<(String, String)> = req
        .messages
        .iter()
        .map(|m| (m.role.clone(), m.content.clone()))
        .collect();
    let completion_id = format!("chatcmpl-{}", st.req_counter.fetch_add(1, Ordering::SeqCst));
    let created = unix_now();
    let model_name = req
        .model
        .clone()
        .or_else(|| st.status.model.lock().unwrap().clone())
        .unwrap_or_else(|| "overhang".to_string());

    if req.stream.unwrap_or(false) {
        stream_completion(st, messages, max_tokens, completion_id, created, model_name)
    } else {
        blocking_completion(st, messages, max_tokens, completion_id, created, model_name).await
    }
}

async fn blocking_completion(
    st: AppState,
    messages: Vec<(String, String)>,
    max_tokens: u64,
    completion_id: String,
    created: u64,
    model_name: String,
) -> Response {
    let gen = tokio::time::timeout(REQUEST_TIMEOUT, async {
        let mut engine = st.engine.lock().await;
        // One ChatCtx per request, read under the engine mutex so template
        // and tokenizer match the engine (/engine/load swaps them while
        // holding this mutex).
        let cc = st.chat();
        let prompt = cc.render(messages.iter().map(|(r, c)| (r.as_str(), c.as_str())));
        let ids = cc.tok.encode(&prompt)?;
        let r = engine.generate(&ids, max_tokens, &st.events, |_| {}).await?;
        anyhow::Ok((cc, ids, r))
    })
    .await;
    let (cc, ids, (tokens, stats)) = match gen {
        Err(_) => return error_response(StatusCode::GATEWAY_TIMEOUT, "generation timed out"),
        Ok(Err(e)) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")),
        Ok(Ok(r)) => r,
    };
    let raw = match cc.tok.decode(&tokens) {
        Ok(t) => t,
        Err(e) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, e),
    };
    let (content, stopped) = strip_stop(&raw, cc.template.stops());
    let finish_reason = if !stopped && stats.n_out >= max_tokens {
        "length"
    } else {
        "stop"
    };
    Json(json!({
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model_name,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": finish_reason,
        }],
        "usage": {
            "prompt_tokens": ids.len(),
            "completion_tokens": stats.n_out,
            "total_tokens": ids.len() as u64 + stats.n_out,
        },
    }))
    .into_response()
}

fn stream_completion(
    st: AppState,
    messages: Vec<(String, String)>,
    max_tokens: u64,
    completion_id: String,
    created: u64,
    model_name: String,
) -> Response {
    let (tx, rx) = mpsc::unbounded_channel::<Result<Event, Infallible>>();

    tokio::spawn(async move {
        let chunk = |delta: Value, finish: Option<&str>| {
            json!({
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [{
                    "index": 0,
                    "delta": delta,
                    "finish_reason": finish,
                }],
            })
            .to_string()
        };

        let _ = tx.send(Ok(
            Event::default().data(chunk(json!({"role": "assistant", "content": ""}), None))
        ));

        let gen = tokio::time::timeout(REQUEST_TIMEOUT, async {
            let mut engine = st.engine.lock().await;
            // One ChatCtx per request, read under the engine mutex so
            // template and tokenizer match the engine (/engine/load swaps
            // them while holding this mutex).
            let cc = st.chat();
            let prompt = cc.render(messages.iter().map(|(r, c)| (r.as_str(), c.as_str())));
            let ids = cc.tok.encode(&prompt)?;
            let mut decoder = StreamDecoder::new(cc);
            let r = engine
                .generate(&ids, max_tokens, &st.events, |tok_id| {
                    // TODO: send {"stop":true} to the engine once a stop
                    // string is seen instead of draining remaining tokens.
                    if let Ok(delta) = decoder.push(tok_id) {
                        if !delta.is_empty() {
                            let _ = tx.send(Ok(
                                Event::default().data(chunk(json!({"content": delta}), None))
                            ));
                        }
                    }
                })
                .await?;
            anyhow::Ok((r, decoder.finished))
        })
        .await;

        match gen {
            Err(_) => {
                tracing::warn!("streaming generation timed out");
                let _ = tx.send(Ok(Event::default().data(chunk(json!({}), Some("length")))));
            }
            Ok(Err(e)) => {
                tracing::warn!("streaming generation failed: {e:#}");
                let _ = tx.send(Ok(Event::default().data(
                    json!({"error": {"message": format!("{e:#}"), "type": "overhangd_error"}})
                        .to_string(),
                )));
            }
            Ok(Ok(((_, stats), finished))) => {
                let finish = if !finished && stats.n_out >= max_tokens {
                    "length"
                } else {
                    "stop"
                };
                let _ = tx.send(Ok(Event::default().data(chunk(json!({}), Some(finish)))));
            }
        }
        let _ = tx.send(Ok(Event::default().data("[DONE]")));
    });

    Sse::new(UnboundedReceiverStream::new(rx))
        .keep_alive(KeepAlive::default())
        .into_response()
}

async fn eject(State(st): State<AppState>) -> Response {
    let mut engine = st.engine.lock().await;
    let was_loaded = engine.eject().await;
    Json(serde_json::json!({ "ejected": was_loaded })).into_response()
}

#[derive(Debug, Deserialize)]
struct LoadRequest {
    model: String,
}

/// POST /engine/load {"model": "<container dir name>"}: eject the current
/// engine and spawn on the named container (tokenizer swapped to match).
async fn load(State(st): State<AppState>, Json(req): Json<LoadRequest>) -> Response {
    // container name, not a path: refuse separators outright
    if req.model.is_empty() || req.model.contains(['/', '\\']) || req.model.contains("..") {
        return error_response(StatusCode::BAD_REQUEST, "model must be a container name");
    }
    let dir = models::model_root(&st.cfg.model_dir).join(&req.model);
    if !dir.join("config.json").exists() {
        return error_response(
            StatusCode::NOT_FOUND,
            format!("no container named {:?} under the model root", req.model),
        );
    }
    // Refuse half-copied containers before ejecting anything — otherwise the
    // failure surfaces as an opaque engine crash after the working engine is
    // already gone.
    if let Err(msg) = models::verify_container(&dir) {
        return error_response(
            StatusCode::UNPROCESSABLE_ENTITY,
            format!("container {:?} is incomplete: {msg}", req.model),
        );
    }
    let model_type = models::read_model_type(&dir);
    let engine_bin = st.cfg.engine_for(&model_type);

    // Load the container's tokenizer before touching the engine, so a bad
    // container leaves the current model running.
    let tok_path = if dir.join("tokenizer.json").exists() {
        dir.join("tokenizer.json")
    } else {
        st.cfg.tokenizer.clone()
    };
    let new_cc = match Tok::load(&tok_path) {
        Ok(t) => ChatCtx {
            tok: Arc::new(t),
            template: crate::chat::Template::for_model_type(&model_type),
        },
        Err(e) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, e),
    };

    let mut engine = st.engine.lock().await;
    match engine.load(dir, engine_bin).await {
        Ok(model) => {
            // Swap tok+template while still holding the engine mutex: chat
            // captures them under the same mutex, so they can never be
            // observed mismatched with the engine.
            *st.tok.write().unwrap() = new_cc;
            Json(json!({ "loaded": true, "model": model })).into_response()
        }
        Err(e) => error_response(StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")),
    }
}

async fn models_list(State(st): State<AppState>) -> Response {
    let root = models::model_root(&st.cfg.model_dir);
    let mut data: Vec<Value> = models::scan(&root, st.cfg.qcache)
        .into_iter()
        .map(|m| {
            json!({
                "id": m.name,
                "object": "model",
                "owned_by": "overhang",
            })
        })
        .collect();
    data.sort_by(|a, b| a["id"].as_str().cmp(&b["id"].as_str()));
    Json(json!({"object": "list", "data": data})).into_response()
}

/// GET /system: discovered hardware — nothing here is hardcoded.
async fn system_info(State(st): State<AppState>) -> Response {
    let info = system::info();
    let (free_gb, total_gb) = system::disk_for(&st.cfg.model_dir);
    let mut v = serde_json::to_value(info).unwrap_or_default();
    v["model_volume_free_gb"] = json!(free_gb);
    v["model_volume_total_gb"] = json!(total_gb);
    v["model_root"] = json!(models::model_root(&st.cfg.model_dir).to_string_lossy());
    Json(v).into_response()
}

async fn status(State(st): State<AppState>) -> Response {
    let info = system::info();
    let (free_disk_gb, _) = system::disk_for(&st.cfg.model_dir);
    let engine_model = st.status.model.lock().unwrap().clone();
    let engine_up = st.status.up.load(Ordering::SeqCst);

    // Capacity ladder measured from the containers actually on disk.
    let root = models::model_root(&st.cfg.model_dir);
    let ladder: Vec<Value> = models::scan(&root, st.cfg.qcache)
        .into_iter()
        .map(|m| {
            json!({
                "name": m.name,
                "model_type": m.model_type,
                "disk_gb": m.disk_gb,
                "ram_gb": m.est_ram_gb,
                "fits_ram": m.est_ram_gb <= info.total_ram_gb * RAM_FIT_FRACTION,
                "fits_disk": true, // already installed on disk
                "active": engine_up && engine_model.as_deref() == Some(m.name.as_str()),
            })
        })
        .collect();

    Json(json!({
        "engine": {
            "up": engine_up,
            "model": engine_model,
        },
        "capacity": {
            "total_ram_gb": info.total_ram_gb,
            "free_disk_gb": free_disk_gb,
            "ladder": ladder,
        },
    }))
    .into_response()
}

async fn events_ws(ws: WebSocketUpgrade, State(st): State<AppState>) -> Response {
    ws.on_upgrade(move |socket| handle_events_socket(socket, st))
}

async fn handle_events_socket(mut socket: WebSocket, st: AppState) {
    let mut rx = st.events.subscribe();
    loop {
        match rx.recv().await {
            Ok(ev) => {
                let text = match serde_json::to_string(&ev) {
                    Ok(t) => t,
                    Err(_) => continue,
                };
                if socket.send(Message::Text(text.into())).await.is_err() {
                    break;
                }
            }
            Err(broadcast::error::RecvError::Lagged(_)) => continue,
            Err(broadcast::error::RecvError::Closed) => break,
        }
    }
}
