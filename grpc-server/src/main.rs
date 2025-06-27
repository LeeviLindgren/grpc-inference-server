#![allow(unused)]
use image::Pixel;
use tonic::{Request, Response, Status, transport::Server};

use proto::mnist_server::{Mnist, MnistServer};
use proto::{MnistImage, MnistPrediction};

use candle_core::{DType, Device};
use mnist::inference::{InferenceEngine, InferenceEngineBuilder};
use mnist::weights_provider::LocalFileProvider;

pub mod proto {
    tonic::include_proto!("mnist");
}

struct MnistService {
    inference_engine: InferenceEngine,
}

impl MnistService {
    fn new() -> Self {
        let provider = LocalFileProvider::from_str("models/mnist_mlp.safetensors").unwrap();
        let inference_engine = InferenceEngineBuilder::new()
            .device(Device::Cpu)
            .dtype(DType::F32)
            .build(provider)
            .unwrap();

        MnistService { inference_engine }
    }
}

#[tonic::async_trait]
impl Mnist for MnistService {
    async fn predict(
        &self,
        request: Request<MnistImage>,
    ) -> Result<Response<MnistPrediction>, Status> {
        // Preprocess the image data by converting to grayscale and resizing
        // and return pixels as a vector of f32
        let processed_image = preprocess_image(&request.into_inner().data);

        // Convert to domain type
        let input_image = mnist::types::MnistImage::try_from(processed_image).unwrap();

        let prediction = self.inference_engine.predict(input_image).unwrap();

        Ok(Response::new(MnistPrediction {
            label: prediction.digit as i32,
            probabilities: prediction.probabilities,
        }))
    }
}

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
#[tokio::main]
async fn main() {
    let addr = "[::1]:50051".parse().unwrap();
    let service = MnistService::new();

    Server::builder()
        .add_service(MnistServer::new(service))
        .serve(addr)
        .await
        .unwrap();
}
