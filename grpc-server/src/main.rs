#![allow(unused)]
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
        let input_image = request
            .into_inner()
            .data
            .iter()
            .map(|&x| x as f32)
            .collect::<Vec<f32>>();
        let input_image = mnist::types::MnistImage::try_from(input_image).unwrap();

        let prediction = self.inference_engine.predict(input_image).unwrap();

        Ok(Response::new(MnistPrediction {
            label: prediction.digit as i32,
            probabilities: prediction.probabilities,
        }))
    }
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
