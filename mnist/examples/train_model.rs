/// Dummy script to just create a model with random weights and save them.
use candle_core::{DType, Device, Result};
use candle_nn::{VarBuilder, VarMap};

use mnist::MnistMLP;

fn main() -> Result<()> {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let varbuilder = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let _ = MnistMLP::new(varbuilder).expect("Failed to create MLP model");

    varmap
        .save("models/mnist_mlp.safetensors")
        .expect("Failed to save model weights");

    Ok(())
}
