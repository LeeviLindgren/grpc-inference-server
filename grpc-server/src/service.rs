use crate::{Error, Result};
use tonic::{Request, Response, Status};

use crate::proto::mnist_server::Mnist;
use crate::proto::{MnistImage, MnistPrediction};

use crate::config::ServiceConfig;
use crate::inference_engine::weights_provider::{LocalFileProvider, WeightsProvider};
use crate::inference_engine::{InferenceEngine, InferenceEngineBuilder, ModelArchitecture};
use candle_core::{DType, Device};

#[derive(Debug)]
pub struct MnistService {
    inference_engine: InferenceEngine,
}

impl MnistService {
    pub fn new(config: ServiceConfig) -> Result<Self> {
        let inference_engine = InferenceEngineBuilder::new()
            .model_architecture(config.model_architecture)
            .device(config.device)
            .dtype(config.dtype)
            .build(config.weights_provider)?;

        Ok(MnistService { inference_engine })
    }
}

#[tonic::async_trait]
impl Mnist for MnistService {
    async fn predict(
        &self,
        request: Request<MnistImage>,
    ) -> std::result::Result<Response<MnistPrediction>, Status> {
        let processed_image = preprocess_image(&request.into_inner().data);

        let prediction = self.inference_engine.predict(processed_image)?;

        Ok(Response::new(MnistPrediction {
            label: prediction.digit as i32,
            probabilities: prediction.probabilities,
        }))
    }
}

/// Convert the image bytes to a vector of f32
fn preprocess_image(image_bytes: &[u8]) -> Vec<f32> {
    // Decode image
    let img = image::load_from_memory(image_bytes).unwrap();

    // Convert to grayscale and resize to 28x28
    let gray = img.to_luma8();
    let resized = image::imageops::resize(&gray, 28, 28, image::imageops::FilterType::Triangle);
    let data: Vec<f32> = resized
        .into_raw()
        .into_iter()
        // Converts to black bacground and white pencil by substracting from one
        .map(|b| 1.0 - (b as f32) / 255.0)
        .collect();

    data
}
