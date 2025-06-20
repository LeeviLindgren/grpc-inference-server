use axum::{extract::Request, middleware::Next, response::Response};
use std::time::Instant;
use tracing::{info, info_span};

pub async fn tracing_layer(request: Request, next: Next) -> Response {
    // Start a timer and create a tracing span for the request
    let start = Instant::now();
    let request_id = uuid::Uuid::new_v4();
    let span = info_span!(
        "request",
        method = %request.method(),
        uri = %request.uri(),
        request_id = %request_id,
    );
    let _enter = span.enter();

    // Pass forward
    let response = next.run(request).await;

    // Log the request status

    let elapsed = start.elapsed().as_micros();

    info!(elapsed = format!("{elapsed}µs"), status = %response.status(), "request completed",);

    response
}
