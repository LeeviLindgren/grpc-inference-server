use std::path::PathBuf;

use crate::{Error, Result};

/// WeightsProvider trait defines a contract for providing model weights
pub trait WeightsProvider {
    fn load_weights(&self) -> Result<Vec<u8>>;
}

pub struct LocalFileProvider {
    path: PathBuf,
}

impl LocalFileProvider {
    pub fn from_str(path: &str) -> Result<Self> {
        let path = PathBuf::from(path);
        if !path.exists() {
            return Err(Error::custom(format!(
                "Weights file does not exist: {}",
                path.display()
            )));
        }
        Ok(Self { path })
    }
}

impl WeightsProvider for LocalFileProvider {
    fn load_weights(&self) -> Result<Vec<u8>> {
        std::fs::read(&self.path).map_err(|e| Error::Custom(e.to_string()))
    }
}
