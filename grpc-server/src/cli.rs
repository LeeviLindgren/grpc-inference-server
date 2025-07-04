use crate::config::{ConfigBuilder, ServerConfig};
use crate::inference_engine::ModelArchitecture;
use clap::{Parser, ValueEnum};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::str::FromStr;

use crate::Result;
use crate::inference_engine::weights_provider::LocalFileProvider;
use candle_core::{DType, Device};

#[derive(Debug, Parser)]
#[command(name = "Mnist Inference Server")]
#[command(about = "A Rust ML inference server for MNIST predictions using candle")]
pub struct Args {
    /// Model architecture to use
    #[arg(long, value_enum)]
    pub model_architecture: ModelArchitecture,

    /// Path to model weights file
    #[arg(long)]
    pub model_weights: PathBuf,

    /// Device to use for inference (cpu, cuda)
    #[arg(long, default_value = "cpu")]
    pub device: String,

    /// Data type to use for computations
    #[arg(long, default_value = "f32")]
    pub dtype: String,

    /// Server bind address
    #[arg(long, default_value = "[::1]:50051")]
    pub address: String,

    /// Tracing level (trace, debug, info, warn, error)
    #[arg(long, default_value = "info")]
    pub log_level: String,

    /// Log format (pretty, json, compact)
    #[arg(long, default_value = "pretty")]
    pub log_format: LogFormat,
}

#[derive(Debug, Clone, ValueEnum)]
pub enum LogFormat {
    Pretty,
    Json,
    Compact,
}

impl Args {
    /// Convert the device string to a candle_core::Device
    pub fn get_device(&self) -> Result<Device> {
        match self.device.to_lowercase().as_str() {
            "cpu" => Ok(Device::Cpu),
            "cuda" => Ok(Device::cuda_if_available(0)?),
            _ => Err(crate::Error::custom(format!(
                "Unsupported device: {}. Supported devices: cpu, cuda",
                self.device
            ))),
        }
    }

    /// Convert the dtype string to a candle_core::DType
    pub fn get_dtype(&self) -> Result<DType> {
        match self.dtype.to_lowercase().as_str() {
            "f16" => Ok(DType::F16),
            "f32" => Ok(DType::F32),
            "f64" => Ok(DType::F64),
            _ => Err(crate::Error::custom(format!(
                "Unsupported dtype: {}. Supported dtypes: f16, f32, f64",
                self.dtype
            ))),
        }
    }

    /// Convert the model weights path to a LocalFileProvider
    pub fn get_weights_provider(&self) -> Result<LocalFileProvider> {
        LocalFileProvider::from_str(
            self.model_weights
                .to_str()
                .ok_or_else(|| crate::Error::custom("Invalid UTF-8 in weights path"))?,
        )
    }

    /// Get the server address
    pub fn get_address(&self) -> Result<SocketAddr> {
        self.address
            .parse()
            .map_err(|e| crate::Error::custom(format!("Invalid address '{}': {}", self.address, e)))
    }

    /// Get the tracing level
    pub fn get_tracing_level(&self) -> Result<tracing::Level> {
        match self.log_level.to_lowercase().as_str() {
            "trace" => Ok(tracing::Level::TRACE),
            "debug" => Ok(tracing::Level::DEBUG),
            "info" => Ok(tracing::Level::INFO),
            "warn" => Ok(tracing::Level::WARN),
            "error" => Ok(tracing::Level::ERROR),
            _ => Err(crate::Error::custom(format!(
                "Invalid log level '{}'. Supported levels: trace, debug, info, warn, error",
                self.log_level
            ))),
        }
    }

    /// Convert CLI args to ServerConfig
    pub fn to_server_config(&self) -> Result<ServerConfig> {
        ConfigBuilder::new()
            .address(self.get_address()?)
            .device(self.get_device()?)
            .dtype(self.get_dtype()?)
            .weights_provider(self.get_weights_provider()?)
            .model_architecture(self.model_architecture)
            .tracing_level(self.get_tracing_level()?)
            .build()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn test_parse_args() {
        let args = Args::try_parse_from(&[
            "rs-candle",
            "--model-architecture",
            "conv",
            "--model-weights",
            "/path/to/weights.bin",
            "--device",
            "cpu",
            "--dtype",
            "f32",
        ])
        .unwrap();

        assert!(matches!(args.model_architecture, ModelArchitecture::Conv));
        assert_eq!(args.model_weights, PathBuf::from("/path/to/weights.bin"));
        assert_eq!(args.device, "cpu");
        assert_eq!(args.dtype, "f32");
    }

    #[test]
    fn test_default_values() {
        let args = Args::try_parse_from(&[
            "rs-candle",
            "--model-architecture",
            "mlp",
            "--model-weights",
            "/path/to/weights.bin",
        ])
        .unwrap();

        assert!(matches!(args.model_architecture, ModelArchitecture::MLP));
        assert_eq!(args.device, "cpu");
        assert_eq!(args.dtype, "f32");
    }

    #[test]
    fn test_get_device() {
        let args = Args {
            model_architecture: ModelArchitecture::Conv,
            model_weights: PathBuf::from("test.bin"),
            device: "cpu".to_string(),
            dtype: "f32".to_string(),
            address: "[::1]:50051".to_string(),
            log_level: "info".to_string(),
            log_format: LogFormat::Pretty,
        };

        let device = args.get_device().unwrap();
        assert!(matches!(device, Device::Cpu));
    }

    #[test]
    fn test_get_dtype() {
        let args = Args {
            model_architecture: ModelArchitecture::Conv,
            model_weights: PathBuf::from("test.bin"),
            device: "cpu".to_string(),
            dtype: "f32".to_string(),
            address: "[::1]:50051".to_string(),
            log_level: "info".to_string(),
            log_format: LogFormat::Pretty,
        };

        let dtype = args.get_dtype().unwrap();
        assert_eq!(dtype, DType::F32);
    }

    #[test]
    fn test_get_address() {
        let args = Args {
            model_architecture: ModelArchitecture::Conv,
            model_weights: PathBuf::from("test.bin"),
            device: "cpu".to_string(),
            dtype: "f32".to_string(),
            address: "127.0.0.1:8080".to_string(),
            log_level: "info".to_string(),
            log_format: LogFormat::Pretty,
        };

        let addr = args.get_address().unwrap();
        assert_eq!(addr.to_string(), "127.0.0.1:8080");
    }

    #[test]
    fn test_get_tracing_level() {
        let args = Args {
            model_architecture: ModelArchitecture::Conv,
            model_weights: PathBuf::from("test.bin"),
            device: "cpu".to_string(),
            dtype: "f32".to_string(),
            address: "[::1]:50051".to_string(),
            log_level: "debug".to_string(),
            log_format: LogFormat::Pretty,
        };

        let level = args.get_tracing_level().unwrap();
        assert_eq!(level, tracing::Level::DEBUG);
    }
}
