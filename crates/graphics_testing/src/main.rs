pub mod shader;
pub mod error;
mod prelude;
mod util;

use include_dir::{include_dir, Dir};
use khal::backend::{Backend, Buffer, Encoder, GpuBackend, WebGpu};
use khal::{AsGpuSlice, Shader};
use kiss3d::prelude::*;
use rand::{RngExt, SeedableRng};
use std::ops::{Div, Mul};
use shader::{run_fdtd_1d, run_fdtd_3d};

pub static SPIRV_DIR: Dir<'static> = include_dir!("$CARGO_MANIFEST_DIR/shaders-spirv");

#[kiss3d::main]
async fn main() {
    let webgpu = WebGpu::default().await.unwrap();
    let backend = GpuBackend::WebGpu(webgpu);

    run_fdtd_3d::run_fdtd_3d(&backend).await;
    // run_fdtd_1d::run_fdtd_1d(&backend).await;
}