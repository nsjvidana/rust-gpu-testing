use khal::backend::{Backend, Buffer, DispatchGrid, GpuBackend, GpuBackendError, GpuBuffer, GpuPass};
use shader_crate::DoubleMe;

pub fn create_buffer(
    backend: &GpuBackend,
    floats: &[f32]
) -> Result<GpuBuffer<f32>, GpuBackendError> {
    backend.init_buffer(
        floats,
        khal::BufferUsages::STORAGE | khal::BufferUsages::COPY_SRC
    )
}

pub fn encode_double_me(
    kernel: &DoubleMe,
    pass: &mut GpuPass,
    floats_buffer: &mut GpuBuffer<f32>,
) -> Result<(), GpuBackendError> {
    let workgroup_count = floats_buffer.len().div_ceil(64) as u32;
    kernel.call(
        pass,
        DispatchGrid::Grid([workgroup_count, 1, 1]),
        floats_buffer,
    )
}