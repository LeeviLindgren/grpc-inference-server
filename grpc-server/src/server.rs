use std::time::Duration;

use http::{Request, Response};
use tonic::transport::Server;
use tower_http::trace::TraceLayer;
use tracing::Span;
use uuid::Uuid;

use crate::config::{ServerConfig, TracingConfig};
use crate::proto::mnist_server::MnistServer;
use crate::service::MnistService;
use crate::{Error, Result};

/// MNIST gRPC Server
pub struct MnistGrpcServer {
    config: ServerConfig,
    service: MnistService,
}

impl MnistGrpcServer {
    /// Create a new server instance with the given configuration
    pub fn new(config: ServerConfig) -> Result<Self> {
        let service = MnistService::new(config.service.clone())?;

        Ok(Self { config, service })
    }

    /// Initialize tracing based on the configuration
    pub fn init_tracing(&self) -> Result<()> {
        let builder = tracing_subscriber::fmt().with_max_level(self.config.tracing.level);
        match self.config.tracing.format {
            crate::cli::LogFormat::Pretty => builder.pretty().init(),
            crate::cli::LogFormat::Json => builder.json().init(),
            crate::cli::LogFormat::Compact => builder.compact().init(),
        }
        Ok(())
    }

    /// Start the server and listen for incoming requests
    pub async fn serve(self) -> Result<()> {
        tracing::info!("Starting MNIST gRPC server on {}", self.config.address);

        let server = Server::builder()
            .layer(
                TraceLayer::new_for_grpc()
                    .make_span_with(|_req: &Request<tonic::body::Body>| {
                        tracing::info_span!(
                            "grpc-request",
                            status_code = tracing::field::Empty,
                            request_id = tracing::field::Empty,
                        )
                    })
                    .on_request(|request: &Request<tonic::body::Body>, span: &Span| {
                        let request_id = Uuid::new_v4().to_string();
                        span.record("request_id", request_id);
                        tracing::info!(
                            method = %request.method(),
                            uri = %request.uri(),
                            version = ?request.version(),
                            "Started processing request"
                        );
                    })
                    .on_response(
                        |response: &Response<tonic::body::Body>, latency: Duration, span: &Span| {
                            span.record("status_code", response.status().as_u16());
                            tracing::info!(
                                status = %response.status(),
                                latency_ms = latency.as_millis(),
                                "Finished processing request"
                            );
                        },
                    ),
            )
            .add_service(MnistServer::new(self.service))
            .serve(self.config.address);

        // Handle graceful shutdown
        tokio::select! {
            result = server => {
                result.map_err(|e| Error::custom(format!("Server error: {}", e)))?;
            }
            _ = tokio::signal::ctrl_c() => {
                tracing::info!("Received shutdown signal, stopping server...");
            }
        }

        tracing::info!("Server stopped");
        Ok(())
    }
}

/// Server builder for convenient server construction
pub struct ServerBuilder {
    config: Option<ServerConfig>,
}

impl ServerBuilder {
    pub fn new() -> Self {
        Self { config: None }
    }

    pub fn with_config(mut self, config: ServerConfig) -> Self {
        self.config = Some(config);
        self
    }

    pub fn build(self) -> Result<MnistGrpcServer> {
        let config = self.config.unwrap_or_else(|| ServerConfig::default());

        MnistGrpcServer::new(config)
    }
}

impl Default for ServerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{ConfigBuilder, ServiceConfig};
    use crate::inference_engine::{ModelArchitecture, weights_provider::LocalFileProvider};
    use candle_core::{DType, Device};
    use std::str::FromStr;

    #[test]
    fn test_server_creation() {
        let provider = LocalFileProvider::from_str("test.safetensors").unwrap();
        let config = ConfigBuilder::new()
            .device(Device::Cpu)
            .dtype(DType::F32)
            .weights_provider(provider)
            .model_architecture(ModelArchitecture::MLP)
            .build()
            .unwrap();

        // This will fail in tests because we don't have actual model weights,
        // but it tests the configuration flow
        let result = MnistGrpcServer::new(config);
        assert!(result.is_err()); // Expected to fail due to missing weights file
    }

    #[test]
    fn test_server_builder() {
        let builder = ServerBuilder::new();
        assert!(builder.config.is_none());
    }

    #[test]
    fn test_server_builder_with_config() {
        let config = ServerConfig::default();
        let builder = ServerBuilder::new().with_config(config);
        assert!(builder.config.is_some());
    }
}
