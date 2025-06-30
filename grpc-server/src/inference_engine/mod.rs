use crate::Error;
use crate::Result;
use candle_core::Tensor;
use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};
use mnist::ConvNet;
use mnist::MnistMLP;
use weights_provider::WeightsProvider;

pub mod weights_provider;

/// InferenceEngine struct to encapsulate the model and device
///
/// It is responsible for loading the model and performing inference
#[derive(Debug)]
pub struct InferenceEngine {
    device: Device,
    dtype: DType,
    model: MnistModel,
}

impl InferenceEngine {
    pub fn builder() -> InferenceEngineBuilder {
        InferenceEngineBuilder::new()
    }

    /// Predict method takes a vector of f32 as input and returns a Prediction struct
    ///
    /// # Arguments:
    /// - `input` -  vector of f32 representing the input image data, should be of size 784 (28x28 pixels flattened)
    ///
    pub fn predict(&self, input: Vec<f32>) -> Result<Prediction> {
        let tensor = Tensor::from_vec(input, (1, 1, 28, 28), &self.device)?;
        let output = self.model.forward(&tensor)?;
        let output = output.flatten_all()?;
        let class_pred = output.argmax(0)?.to_scalar::<u32>()?;
        let probabilities = output.to_vec1()?;
        Ok(Prediction {
            digit: class_pred,
            probabilities,
        })
    }
}

#[derive(Debug, Clone, Copy, clap::ValueEnum)]
pub enum ModelArchitecture {
    MLP,
    Conv,
}

/// Builder class for the InferenceEngine
pub struct InferenceEngineBuilder {
    model_architecture: Option<ModelArchitecture>,
    device: Option<Device>,
    dtype: Option<DType>,
}

impl InferenceEngineBuilder {
    pub fn new() -> Self {
        Self {
            model_architecture: None,
            device: None,
            dtype: None,
        }
    }

    pub fn model_architecture(mut self, arch: ModelArchitecture) -> InferenceEngineBuilder {
        self.model_architecture = Some(arch);
        InferenceEngineBuilder {
            model_architecture: self.model_architecture,
            device: self.device,
            dtype: self.dtype,
        }
    }

    pub fn device(mut self, device: Device) -> Self {
        self.device = Some(device);
        self
    }

    pub fn dtype(mut self, dtype: DType) -> Self {
        self.dtype = Some(dtype);
        self
    }

    /// Build the InferenceEngine with the specified weights provider that loads the model weights
    pub fn build<P: WeightsProvider>(self, provider: P) -> Result<InferenceEngine> {
        let device = self.device.unwrap_or(Device::Cpu);
        let dtype = self.dtype.unwrap_or(DType::F32);

        // Load the weights from the provider
        let weights = provider.load_weights()?;
        let varbuilder = VarBuilder::from_buffered_safetensors(weights, dtype, &device)?;

        // Initialize the model based on the architecture
        // Errors if the architecture is not set
        let model = match self.model_architecture {
            Some(arch) => MnistModel::new(varbuilder, arch)?,
            None => return Err(Error::custom("Model architecture not set")),
        };
        Ok(InferenceEngine {
            device,
            dtype,
            model,
        })
    }
}

/// Prediction struct to hold the result of the inference
pub struct Prediction {
    pub digit: u32,
    pub probabilities: Vec<f32>,
}

/// Model enum to encapsulate different architectures
#[derive(Debug)]
enum MnistModel {
    MLP(MnistMLP),
    Conv(ConvNet),
}

impl MnistModel {
    pub fn new(varbuilder: VarBuilder, arch: ModelArchitecture) -> Result<Self> {
        match arch {
            ModelArchitecture::MLP => {
                let model = MnistMLP::new(varbuilder)?;
                Ok(MnistModel::MLP(model))
            }
            ModelArchitecture::Conv => {
                let model = ConvNet::new(varbuilder)?;
                Ok(MnistModel::Conv(model))
            }
        }
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Map the candle error to our internal error type
        match self {
            MnistModel::MLP(model) => model.forward(input).map_err(|e| e.into()),
            MnistModel::Conv(model) => model.forward(input).map_err(|e| e.into()),
        }
    }
}
