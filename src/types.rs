use crate::{Error, Result};
use candle_core::{Device, Tensor};
use serde::{Deserialize, Serialize, de};

const MNIST_IMAGE_SIZE: usize = 784;

#[derive(Debug, Deserialize)]
pub struct MnistImage {
    #[serde(deserialize_with = "deserialize_mnist")]
    data: Vec<f32>,
}

fn deserialize_mnist<'de, D>(deserializer: D) -> std::result::Result<Vec<f32>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let data: Vec<f32> = Deserialize::deserialize(deserializer)?;

    let data_len = data.len();
    if data_len != MNIST_IMAGE_SIZE {
        return Err(serde::de::Error::custom(format!(
            "Expected {} elements, got {}",
            MNIST_IMAGE_SIZE, data_len
        )));
    }
    Ok(data)
}

impl TryInto<Tensor> for MnistImage {
    type Error = Error;

    fn try_into(self) -> Result<Tensor> {
        let data = Tensor::from_vec(self.data, (1, 784), &Device::Cpu)?;
        Ok(data)
    }
}

#[derive(Serialize, Debug)]
pub struct MnistPrediction {
    pub digit: u32,
    pub probabilities: Vec<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_deserialize_mnist_invalid_size() {
        let json_data = json!({
            "data": [0.0, 1.0]
        });

        let image: MnistImage = serde_json::from_value(json_data).unwrap();
        assert_eq!(image.data.len(), MNIST_IMAGE_SIZE);
    }
}
