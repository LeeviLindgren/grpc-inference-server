use crate::{
    Result,
    types::{MnistImage, MnistPrediction},
};
use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};

use crate::model::MnistMLP;

// pl.scan_parquet("s3://bucket/account_id=1234/**/*.parquet")

pub struct InferenceEngine {
    device: Device,
    model: MnistMLP,
}

impl InferenceEngine {
    pub fn new(device: Device, weights_path: &str) -> Result<Self> {
        let mut varmap = VarMap::new();
        let varbuilder = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let model = MnistMLP::new(varbuilder)?;
        varmap.load(weights_path)?;
        Ok(Self { device, model })
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
