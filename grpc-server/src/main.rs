use std::time::Duration;

use clap::Parser;
use grpc_server::cli::Args;
use grpc_server::proto::mnist_server::MnistServer;
use grpc_server::service::{MnistService, ServiceConfig};
use http::{Request, Response};
use tonic::transport::Server;
use tower_http::trace::TraceLayer;
use tracing::Span;
use uuid::Uuid;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    let args = Args::parse();
    let config = ServiceConfig {
        device: args.get_device().unwrap(),
        dtype: args.get_dtype().unwrap(),
        provider: args.get_weights_provider().unwrap(),
        model_architectur: args.model_architecture,
    };
    let addr = "[::1]:50051".parse().unwrap();
    let service = MnistService::new(config).unwrap();

    tracing::info!("starting server...");
    Server::builder()
        .layer(
            TraceLayer::new_for_grpc()
                .make_span_with(|_req: &http::Request<tonic::body::Body>| {
                    tracing::info_span!(
                        "grpc-request",
                        status_code = tracing::field::Empty,
                        request_id = tracing::field::Empty,
                    )
                })
                .on_request(|request: &Request<tonic::body::Body>, span: &Span| {
                    let request_id = Uuid::new_v4().to_string();
                    span.record("request_id", request_id);
                    tracing::info!(path = ?request.uri().path())
                })
                .on_response(
                    |response: &Response<tonic::body::Body>, latency: Duration, span: &Span| {
                        span.record("status_code", response.status().as_u16());
                        tracing::info!(latency = format!("{}ms", latency.as_millis()))
                    },
                ),
        )
        .add_service(MnistServer::new(service))
        .serve(addr)
        .await
        .unwrap();
}
