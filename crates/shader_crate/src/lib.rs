#![no_std]

use spirv_std::glam::UVec3;
use spirv_std::spirv;

#[spirv(compute(threads(64)))]
pub fn double_me(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] floats: &mut [f32],
) {
    let idx = id.x as usize;
    floats[idx] *= 2.0;
}