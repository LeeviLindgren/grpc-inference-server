#![allow(unused)]

pub mod inference;
pub mod model;
pub mod types;
pub mod weights_provider;

use derive_more::{Display, From};

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Display, From)]
pub enum Error {
    #[from]
    Custom(String),

    // External errors
    #[from]
    CandleError(candle_core::Error),
}

impl Error {
    pub fn custom<S: Into<String>>(msg: S) -> Self {
        Error::Custom(msg.into())
    }
}

impl From<&str> for Error {
    fn from(value: &str) -> Self {
        Error::Custom(value.to_string())
    }
}

impl std::error::Error for Error {}
