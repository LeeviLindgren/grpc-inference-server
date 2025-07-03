#![allow(unused)]
pub mod cli;
pub mod config;
pub mod error;
pub mod inference_engine;
pub mod interceptors;
pub mod server;
pub mod service;

pub use error::{Error, Result};

pub mod proto {
    tonic::include_proto!("mnist");
}
