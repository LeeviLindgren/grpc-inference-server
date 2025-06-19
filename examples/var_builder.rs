use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{
    Module, VarBuilder, VarMap,
    linear::{Linear, linear},
};

struct MLP {
    layer1: Linear,
    layer2: Linear,
}

impl MLP {
    fn new(varbuilder: VarBuilder) -> Result<Self> {
        let layer1 = linear(5, 5, varbuilder.pp("layer1"))?;
        let layer2 = linear(5, 1, varbuilder.pp("layer2"))?;
        Ok(Self { layer1, layer2 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.layer1.forward(x)?;
        let x = x.relu()?;
        self.layer2.forward(&x)
    }
}

fn main() -> Result<()> {
    let device = Device::Cpu;

    let mut varmap = VarMap::new();
    let varbuilder = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    // varmap.load("models/mlp.safetensors")?; output is now random!
    let model = MLP::new(varbuilder)?;
    // Loading after creating the model fixes the output
    varmap.load("models/mlp.safetensors")?;

    let input = Tensor::ones((1, 5), DType::F32, &device)?;
    let output = model.forward(&input)?;

    println!("Output: {:?}", output.to_vec2::<f32>()?);

    // varmap.save("models/mlp.safetensors")?;
    Ok(())
}
