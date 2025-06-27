use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{
    Conv2d, Conv2dConfig, Module, VarBuilder, VarMap, conv2d,
    linear::{Linear, linear},
};

struct ConvNet {
    conv2d_1: Conv2d,
    conv2d_2: Conv2d,
    linear_1: Linear,
    linear_2: Linear,
}

impl ConvNet {
    fn new(varbuilder: VarBuilder) -> Result<Self> {
        let conv2d_1 = conv2d(
            1,
            32,
            3,
            Conv2dConfig {
                padding: 1,
                ..Default::default()
            },
            varbuilder.pp("conv2d_1"),
        )?;
        let conv2d_2 = conv2d(
            32,
            64,
            3,
            Conv2dConfig {
                padding: 1,
                ..Default::default()
            },
            varbuilder.pp("conv2d_2"),
        )?;
        let linear_1 = linear(64 * 7 * 7, 128, varbuilder.pp("linear_1"))?;
        let linear_2 = linear(128, 10, varbuilder.pp("linear_2"))?;
        Ok(Self {
            conv2d_1,
            conv2d_2,
            linear_1,
            linear_2,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
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

fn main() -> Result<()> {
    let device = Device::Cpu;

    let mut varmap = VarMap::new();
    let varbuilder = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = ConvNet::new(varbuilder)?;
    varmap.load("models/mnist_convnet.safetensors")?;

    let input = Tensor::ones((1, 1, 28, 28), DType::F32, &device)?;
    let output = model.forward(&input)?;

    println!("Output: {:?}", output.to_vec2::<f32>()?);

    // varmap.save("models/mlp.safetensors")?;
    Ok(())
}
