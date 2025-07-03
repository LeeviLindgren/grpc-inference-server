use std::net::SocketAddr;
use std::str::FromStr;

use crate::cli::LogFormat;
use crate::inference_engine::ModelArchitecture;
use crate::inference_engine::weights_provider::LocalFileProvider;
use crate::{Error, Result};
use candle_core::{DType, Device};

/// Server configuration
#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub address: SocketAddr,
    pub service: ServiceConfig,
    pub tracing: TracingConfig,
}

/// Service-specific configuration
#[derive(Debug, Clone)]
pub struct ServiceConfig {
    pub device: Device,
    pub dtype: DType,
    pub weights_provider: LocalFileProvider,
    pub model_architecture: ModelArchitecture,
}

/// Tracing configuration
#[derive(Debug, Clone)]
pub struct TracingConfig {
    pub level: tracing::Level,
    pub format: LogFormat,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            address: "[::1]:50051".parse().unwrap(),
            service: ServiceConfig::default(),
            tracing: TracingConfig::default(),
        }
    }
}

impl Default for ServiceConfig {
    fn default() -> Self {
        Self {
            device: Device::Cpu,
            dtype: DType::F32,
            weights_provider: LocalFileProvider::from_str("model.safetensors").unwrap(),
            model_architecture: ModelArchitecture::MLP,
        }
    }
}

impl Default for TracingConfig {
    fn default() -> Self {
        Self {
            level: tracing::Level::INFO,
            format: LogFormat::Pretty,
        }
    }
}

impl ServerConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_address(mut self, address: SocketAddr) -> Self {
        self.address = address;
        self
    }

    pub fn with_service_config(mut self, service: ServiceConfig) -> Self {
        self.service = service;
        self
    }

    pub fn with_tracing_config(mut self, tracing: TracingConfig) -> Self {
        self.tracing = tracing;
        self
    }
}

impl ServiceConfig {
    pub fn new(
        device: Device,
        dtype: DType,
        weights_provider: LocalFileProvider,
        model_architecture: ModelArchitecture,
    ) -> Self {
        Self {
            device,
            dtype,
            weights_provider,
            model_architecture,
        }
    }
}

impl TracingConfig {
    pub fn new(level: tracing::Level, format: LogFormat) -> Self {
        Self { level, format }
    }

    pub fn with_level(mut self, level: tracing::Level) -> Self {
        self.level = level;
        self
    }
}

/// Configuration builder for easy construction from CLI args or environment
pub struct ConfigBuilder {
    address: Option<SocketAddr>,
    device: Option<Device>,
    dtype: Option<DType>,
    weights_provider: Option<LocalFileProvider>,
    model_architecture: Option<ModelArchitecture>,
    tracing_level: Option<tracing::Level>,
    format: Option<LogFormat>,
}

impl ConfigBuilder {
    pub fn new() -> Self {
        Self {
            address: None,
            device: None,
            dtype: None,
            weights_provider: None,
            model_architecture: None,
            tracing_level: None,
            format: None,
        }
    }

    pub fn address(mut self, address: SocketAddr) -> Self {
        self.address = Some(address);
        self
    }

    pub fn device(mut self, device: Device) -> Self {
        self.device = Some(device);
        self
    }

    pub fn dtype(mut self, dtype: DType) -> Self {
        self.dtype = Some(dtype);
        self
    }

    pub fn weights_provider(mut self, provider: LocalFileProvider) -> Self {
        self.weights_provider = Some(provider);
        self
    }

    pub fn model_architecture(mut self, arch: ModelArchitecture) -> Self {
        self.model_architecture = Some(arch);
        self
    }

    pub fn tracing_level(mut self, level: tracing::Level) -> Self {
        self.tracing_level = Some(level);
        self
    }

    pub fn format(mut self, format: LogFormat) -> Self {
        self.format = Some(format);
        self
    }

    pub fn build(self) -> Result<ServerConfig> {
        let service = ServiceConfig {
            device: self.device.unwrap_or(Device::Cpu),
            dtype: self.dtype.unwrap_or(DType::F32),
            weights_provider: self
                .weights_provider
                .ok_or_else(|| Error::custom("Weights provider must be specified"))?,
            model_architecture: self
                .model_architecture
                .ok_or_else(|| Error::custom("Model architecture must be specified"))?,
        };

        let tracing = TracingConfig {
            level: self.tracing_level.unwrap_or(tracing::Level::INFO),
            format: self.format.unwrap_or(LogFormat::Pretty),
        };

        Ok(ServerConfig {
            address: self
                .address
                .unwrap_or_else(|| "[::1]:50051".parse().unwrap()),
            service,
            tracing,
        })
    }
}

impl Default for ConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    #[test]
    fn test_default_config() {
        let config = ServerConfig::default();
        assert_eq!(config.address.to_string(), "[::1]:50051");
        assert!(matches!(config.service.device, Device::Cpu));
        assert_eq!(config.service.dtype, DType::F32);
        assert!(matches!(
            config.service.model_architecture,
            ModelArchitecture::MLP
        ));
        assert_eq!(config.tracing.level, tracing::Level::INFO);
    }

    #[test]
    fn test_config_builder() {
        let provider = LocalFileProvider::from_str("test.safetensors").unwrap();
        let config = ConfigBuilder::new()
            .address("127.0.0.1:8080".parse().unwrap())
            .device(Device::Cpu)
            .dtype(DType::F32)
            .weights_provider(provider)
            .model_architecture(ModelArchitecture::Conv)
            .tracing_level(tracing::Level::DEBUG)
            .build()
            .unwrap();

        assert_eq!(config.address.to_string(), "127.0.0.1:8080");
        assert!(matches!(
            config.service.model_architecture,
            ModelArchitecture::Conv
        ));
        assert_eq!(config.tracing.level, tracing::Level::DEBUG);
    }

    #[test]
    fn test_config_builder_missing_required() {
        let result = ConfigBuilder::new()
            .address("127.0.0.1:8080".parse().unwrap())
            .device(Device::Cpu)
            .build();

        assert!(result.is_err());
    }
}
