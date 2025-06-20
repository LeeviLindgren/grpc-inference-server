//! Candle model definitions for MNIST classification.

use crate::{Error, Result};
use candle_core::{Device, Tensor};
use candle_nn::Module;
use candle_nn::{self as nn};

pub struct MnistImage(Vec<f32>);

impl MnistImage {
    pub fn new(data: Vec<f32>) -> Self {
        Self(data)
    }
}

impl TryInto<Tensor> for MnistImage {
    type Error = Error;

    fn try_into(self) -> Result<Tensor> {
        let data = Tensor::from_vec(self.0, (1, 784), &Device::Cpu)?;
        Ok(data)
    }
}

pub trait MnistModel {
    fn predict(&self, image: MnistImage) -> Result<u32>;
    fn predict_proba(&self, image: MnistImage) -> Result<Vec<f32>>;
}

pub struct MnistMLP {
    fc1: nn::linear::Linear,
    fc2: nn::linear::Linear,
    fc3: nn::linear::Linear,
}

impl MnistMLP {
    pub fn new(varbuilder: nn::VarBuilder) -> Result<Self> {
        let fc1 = nn::linear::linear(784, 128, varbuilder.pp("fc1"))?;
        let fc2 = nn::linear::linear(128, 64, varbuilder.pp("fc2"))?;
        let fc3 = nn::linear::linear(64, 10, varbuilder.pp("fc3"))?;

        Ok(Self { fc1, fc2, fc3 })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.fc1.forward(x)?.relu()?;
        let x = self.fc2.forward(&x)?.relu()?;
        let x = self.fc3.forward(&x)?;
        let out = nn::ops::softmax_last_dim(&x)?;
        Ok(out)
    }
}
