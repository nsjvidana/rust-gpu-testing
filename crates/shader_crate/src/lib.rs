#![no_std]

use spirv_std::glam::UVec3;
use spirv_std::spirv;

#[spirv(compute(threads(64)))]
pub fn main_cs(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] floats: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] floats_out: &mut [f32],
) {
    let idx = id.x as usize;
    floats_out[idx] = 1.0;
}