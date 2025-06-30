use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{VarBuilder, VarMap};

use mnist::ConvNet;

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
