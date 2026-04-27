use khal::backend::GpuBackendError;
pub use crate::error::Error;

pub type GpuResult<T> = core::result::Result<T, GpuBackendError>;
pub type Result<T> = core::result::Result<T, Error>;