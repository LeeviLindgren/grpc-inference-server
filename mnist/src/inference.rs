use crate::{
    Result,
    types::{MnistImage, MnistPrediction},
    weights_provider::WeightsProvider,
};
use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};

use crate::model::MnistMLP;

pub struct InferenceEngine {
    device: Device,
    dtype: DType,
    model: MnistMLP,
}

impl InferenceEngine {
    pub fn builder() -> InferenceEngineBuilder {
        InferenceEngineBuilder::new()
    }

    pub fn predict(&self, input: MnistImage) -> Result<MnistPrediction> {
        let tensor = input.try_into()?;
        let output = self.model.forward(&tensor)?;
        let output = output.flatten_all()?;
        let class_pred = output.argmax(0)?.to_scalar::<u32>()?;
        let probabilities = output.to_vec1()?;
        Ok(MnistPrediction {
            digit: class_pred,
            probabilities,
        })
    }
}

pub struct InferenceEngineBuilder {
    device: Option<Device>,
    dtype: Option<DType>,
}

impl InferenceEngineBuilder {
    pub fn new() -> Self {
        Self {
            device: None,
            dtype: None,
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

    pub fn build<P: WeightsProvider>(self, provider: P) -> Result<InferenceEngine> {
        let device = self.device.unwrap_or(Device::Cpu);
        let dtype = self.dtype.unwrap_or(DType::F32);
        let weights = provider.load_weights()?;
        let varbuilder = VarBuilder::from_buffered_safetensors(weights, dtype, &device)?;
        let model = MnistMLP::new(varbuilder)?;
        Ok(InferenceEngine {
            device,
            dtype,
            model,
        })
    }
}
