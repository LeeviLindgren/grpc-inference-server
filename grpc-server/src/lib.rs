#![allow(unused)]
pub mod cli;
pub mod error;
pub mod inference_engine;
pub mod interceptors;
pub mod service;

pub use error::{Error, Result};

pub mod proto {
    tonic::include_proto!("mnist");
}
