use tonic::Status;

pub type Result<T> = std::result::Result<T, Error>;

use derive_more::{Display, From};

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

impl From<Error> for Status {
    fn from(error: Error) -> Status {
        match error {
            // Map your custom error variants to appropriate gRPC status codes
            _ => Status::unknown("An unknown error occurred"),
        }
    }
}

impl std::error::Error for Error {}
