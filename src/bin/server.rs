use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{
    Router,
    extract::{Json, State},
};
use rs_candle::Error as InternalError;
use rs_candle::inference::InferenceEngine;
use rs_candle::types::{MnistImage, MnistPrediction};
use std::sync::Arc;

enum AppError {
    InternalError(InternalError),
}

impl From<InternalError> for AppError {
    fn from(err: InternalError) -> Self {
        AppError::InternalError(err)
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> axum::response::Response {
        let result = match self {
            AppError::InternalError(msg) => (
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                format!("Internal server error: {}", msg),
            )
                .into_response(),
        };
        println!("{:?}", result);
        result
    }
}

#[derive(Clone)]
struct AppState {
    model: Arc<InferenceEngine>,
}

async fn ping() -> impl IntoResponse {
    "pong"
}

async fn predict(
    State(AppState { model }): State<AppState>,
    Json(request): Json<MnistImage>,
) -> Result<Json<MnistPrediction>, AppError> {
    let prediction = model.predict(request)?;
    Ok(Json(prediction))
}

fn app() -> Router {
    let state = AppState {
        model: Arc::new(
            InferenceEngine::new(candle_core::Device::Cpu, "models/mnist_mlp.safetensors")
                .expect("Failed to load model"),
        ),
    };

    Router::new()
        .route("/ping", get(ping))
        .route("/predict", post(predict))
        .with_state(state)
}

#[tokio::main]
async fn main() {
    let app = app();

    let listener = tokio::net::TcpListener::bind("127.0.0.1:8080")
        .await
        .unwrap();
    axum::serve(listener, app).await.unwrap();
}
