use axum::extract::rejection::JsonRejection;
use axum::response::IntoResponse;
use derive_more::{Display, From};
use mnist::Error as MnistError;

pub type Result<T> = std::result::Result<T, AppError>;

#[derive(Debug, Display, From)]
pub enum AppError {
    // External
    #[from]
    ModelError(MnistError),

    #[from]
    JsonError(JsonRejection),
}

impl IntoResponse for AppError {
    fn into_response(self) -> axum::response::Response {
        tracing::error!(error = ?self, "server error");

        match self {
            AppError::JsonError(err) => {
                let (status, message) = (
                    axum::http::StatusCode::BAD_REQUEST,
                    format!("JSON error: {}", err),
                );
                (status, message).into_response()
            }
            _ => {
                let (status, message) = (
                    axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                    "Internal server error".to_string(),
                );
                (status, message).into_response()
            }
        }
    }
}

impl std::error::Error for AppError {}
