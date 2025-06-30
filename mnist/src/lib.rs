use candle_core::Tensor;
use candle_nn::Module;
use candle_nn::{self as nn};

type Result<T> = std::result::Result<T, candle_core::Error>;

#[derive(Debug)]
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

#[derive(Debug)]
pub struct ConvNet {
    conv2d_1: nn::Conv2d,
    conv2d_2: nn::Conv2d,
    linear_1: nn::Linear,
    linear_2: nn::Linear,
}

impl ConvNet {
    pub fn new(varbuilder: nn::VarBuilder) -> Result<Self> {
        let conv2d_1 = nn::conv2d(
            1,
            32,
            3,
            nn::Conv2dConfig {
                padding: 1,
                ..Default::default()
            },
            varbuilder.pp("conv2d_1"),
        )?;
        let conv2d_2 = nn::conv2d(
            32,
            64,
            3,
            nn::Conv2dConfig {
                padding: 1,
                ..Default::default()
            },
            varbuilder.pp("conv2d_2"),
        )?;
        let linear_1 = nn::linear(64 * 7 * 7, 128, varbuilder.pp("linear_1"))?;
        let linear_2 = nn::linear(128, 10, varbuilder.pp("linear_2"))?;
        Ok(Self {
            conv2d_1,
            conv2d_2,
            linear_1,
            linear_2,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.conv2d_1.forward(x)?;
        let x = x.relu()?;
        let x = x.max_pool2d((2, 2))?;
        let x = self.conv2d_2.forward(&x)?;
        let x = x.relu()?;
        let x = x.max_pool2d((2, 2))?;
        let x = x.flatten(1, 3)?;
        let x = self.linear_1.forward(&x)?;
        let x = x.relu()?;
        let x = self.linear_2.forward(&x)?;
        candle_nn::ops::softmax_last_dim(&x)
    }
}
