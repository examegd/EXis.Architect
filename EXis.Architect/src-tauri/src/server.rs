use axum::{
    routing::{get, post},
    Router, Json, response::sse::{Event, Sse},
};
use tower_http::cors::CorsLayer;
use serde::Deserialize;
use serde_json::json;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::path::PathBuf;
use futures_util::Stream;
use tokio::sync::mpsc::{self, Receiver};

use crate::ExiqState;
use crate::exiq::{ExiqModel, generate_stream, DeviceSelection};

pub struct ReceiverStream<T> {
    inner: Receiver<T>,
}

impl<T> ReceiverStream<T> {
    pub fn new(inner: Receiver<T>) -> Self {
        Self { inner }
    }
}

impl<T> Stream for ReceiverStream<T> {
    type Item = T;
    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.inner.poll_recv(cx)
    }
}

#[derive(Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub repeat_penalty: Option<f32>,
    pub max_tokens: Option<usize>,
}

fn resolve_model_path(raw_name: &str) -> Option<PathBuf> {
    let clean = raw_name.trim().trim_start_matches("exiq:").trim();
    if clean.is_empty() {
        return None;
    }

    let file_name = if clean.ends_with(".exiq") {
        clean.to_string()
    } else {
        format!("{}.exiq", clean)
    };

    let candidates = [
        PathBuf::from(&file_name),
        PathBuf::from("src-tauri").join(&file_name),
        PathBuf::from("..").join(&file_name),
    ];

    for p in &candidates {
        if p.is_file() {
            return Some(p.clone());
        }
    }

    if let Ok(home) = std::env::var("USERPROFILE") {
        let p_doc = PathBuf::from(home).join("Documents").join("exis").join("models").join(&file_name);
        if p_doc.is_file() {
            return Some(p_doc);
        }
    }

    None
}

pub async fn start_api_server(state: Arc<ExiqState>, port: u16) {
    let app = Router::new()
        .route("/v1/models", get(list_models))
        .route("/api/tags", get(list_models))
        .route("/v1/chat/completions", post(move |body| handle_chat_openai(body, state.clone())))
        .layer(CorsLayer::permissive());

    let listener = tokio::net::TcpListener::bind(format!("127.0.0.1:{}", port))
        .await
        .unwrap();

    println!("[Exis Server] http://127.0.0.1:{}", port);
    let _ = axum::serve(listener, app).await;
}

async fn list_models() -> Json<serde_json::Value> {
    Json(json!({
        "object": "list",
        "data": [{ "id": "exiq:exisq.exiq", "object": "model", "owned_by": "exis" }]
    }))
}

async fn handle_chat_openai(
    Json(req): Json<ChatCompletionRequest>,
    state: Arc<ExiqState>,
) -> Sse<impl Stream<Item = Result<Event, std::convert::Infallible>>> {
    let (tx, rx) = mpsc::channel(100);

    
    state.is_aborted.store(false, Ordering::Relaxed);
    let abort_flag = state.is_aborted.clone();

    let mut prompt = String::new();
    for m in &req.messages {
        if !m.content.trim().is_empty() {
            prompt.push_str(&format!("<|{}|>\n{}</s>\n", m.role, m.content));
        }
    }
    prompt.push_str("<|assistant|>\n");

    let temp = req.temperature.unwrap_or(0.7);
    let top_p = req.top_p.unwrap_or(0.9);
    let repeat_penalty = req.repeat_penalty.unwrap_or(1.1);
    let max_tok = req.max_tokens.unwrap_or(512);
    let requested_model_name = req.model.clone();

    let state_clone = state.clone();
    tokio::task::spawn_blocking(move || {
        let mut model_lock = state_clone.model.lock().unwrap_or_else(|e| e.into_inner());
        let mut name_lock = state_clone.active_model_name.lock().unwrap_or_else(|e| e.into_inner());

        if model_lock.is_none() || *name_lock != requested_model_name {
            if let Some(model_path) = resolve_model_path(&requested_model_name) {
                if model_lock.is_some() {
                    println!("[Exis Server] 🧹 Выгрузка старой модели '{}'...", *name_lock);
                    model_lock.take();
                }

                println!("[Exis Server] 🚀 Загрузка модели {:?} в VRAM...", model_path);
                match ExiqModel::load(&model_path) {
                    Ok(m) => {
                        *model_lock = Some(m);
                        *name_lock = requested_model_name.clone();
                        println!("[Exis Server] ✅ Модель готова к генерации!");
                    }
                    Err(e) => {
                        eprintln!("[Exis Server] ❌ Ошибка загрузки модели: {}", e);
                        let err_msg = json!({
                            "error": { "message": format!("Load error: {}", e), "type": "server_error" }
                        });
                        let _ = tx.blocking_send(Ok(Event::default().data(err_msg.to_string())));
                        return;
                    }
                }
            } else {
                eprintln!("[Exis Server] ❌ Файл модели '{}' не найден!", requested_model_name);
                let err_msg = json!({
                    "error": { "message": format!("Model '{}' not found on server", requested_model_name), "type": "model_not_found" }
                });
                let _ = tx.blocking_send(Ok(Event::default().data(err_msg.to_string())));
                return;
            }
        }

        if let Some(ref mut model) = *model_lock {
            let tx_clone = tx.clone();
            
            let on_token = move |tok: String| {
                let chunk = json!({
                    "choices": [{
                        "delta": { "content": tok },
                        "index": 0,
                        "finish_reason": null
                    }]
                });
                let _ = tx_clone.blocking_send(Ok(Event::default().data(chunk.to_string())));
            };

            let _ = generate_stream(
                model,
                &prompt,
                max_tok,
                temp,
                top_p,
                repeat_penalty,
                4,
                DeviceSelection::Auto,
                true, 
                4096,
                abort_flag,
                on_token,
            );
        }
    });

    Sse::new(ReceiverStream::new(rx))
}