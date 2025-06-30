use std::time::Instant;
use tonic::{Request, Status, service::Interceptor};

pub fn tracing_interceptor(mut req: Request<()>) -> Result<Request<()>, Status> {
    let start = Instant::now();
    let request_id = uuid::Uuid::new_v4();
    let span = tracing::info_span!(
        "request",
        from = ?req.remote_addr(),
        request_id = %request_id,
    );
    span.in_scope(|| tracing::info!("gRPC request"));

    Ok(req)
}
