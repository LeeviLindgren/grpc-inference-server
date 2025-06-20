use std::sync::Arc;

use crate::middleware::tracing_layer;
use crate::{AppError, Result};
use axum::{
    Json, Router,
    extract::State,
    response::IntoResponse,
    routing::{get, post},
};
use axum_extra::extract::WithRejection;
use candle_core::{DType, Device};
use mnist::{
    inference::InferenceEngine,
    types::{MnistImage, MnistPrediction},
};

pub enum WeightsProvider {
    LocalFile(String),
    S3 { bucket: String, key: String },
}

pub struct AppConfig {
    pub device: Device,
    pub dtype: DType,
    pub weights_provider: WeightsProvider,
}

#[derive(Clone)]
struct AppState {
    model: Arc<InferenceEngine>,
}

impl AppState {
    pub fn new(config: AppConfig) -> Result<Self> {
        let provider = match config.weights_provider {
            WeightsProvider::LocalFile(path) => {
                mnist::weights_provider::LocalFileProvider::from_str(&path)?
            }
            WeightsProvider::S3 { .. } => {
                // Implement S3 provider logic here
                unimplemented!()
            }
        };

        Ok(Self {
            model: Arc::new(
                InferenceEngine::builder()
                    .device(config.device)
                    .dtype(config.dtype)
                    .build(provider)?,
            ),
        })
    }
}

async fn ping() -> impl IntoResponse {
    (axum::http::StatusCode::OK, "OK")
}

async fn predict(
    State(AppState { model }): State<AppState>,
    WithRejection(Json(request), _): WithRejection<Json<MnistImage>, AppError>,
) -> Result<Json<MnistPrediction>> {
    let prediction = model.predict(request)?;
    Ok(Json(prediction))
}

pub fn app(config: AppConfig) -> Router {
    let state = AppState::new(config).expect("Failed to create app state");
    Router::new()
        .route("/ping", get(ping))
        .route("/predict", post(predict))
        .with_state(state)
        .layer(axum::middleware::from_fn(tracing_layer))
}
