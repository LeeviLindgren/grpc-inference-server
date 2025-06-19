#![allow(unused)]

pub mod inference;
pub mod model;
pub mod types;

use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Error, Debug)]
pub enum Error {
    #[error("Candle error: {0}")]
    Candle(#[from] candle_core::Error),
}
