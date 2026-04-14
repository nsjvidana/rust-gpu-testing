#![cfg_attr(target_arch = "spirv", no_std)]

use khal_std::glamx::UVec3;

pub mod fdtd_1d;
pub mod fdtd_3d;

/// Select between two numerical values `tru` and `fals` of type `typ` depending on a bool `b`.
///
/// Returns `tru` if `b` is true, `fals` otherwise
#[macro_export]
macro_rules! select_val {
    ($b:expr, $tru:expr, $fals:expr, $typ:ty) => {{
        let b_n = (!$b) as usize as $typ;
        let b = $b as usize as $typ;
        $tru * b + $fals * b_n
    }};
}

pub fn flat_idx_to_vector(idx: u32, grid_dim: UVec3) -> UVec3 {
    UVec3::new(
        idx % grid_dim.x,
        (idx / grid_dim.x) % grid_dim.y,
        idx / (grid_dim.x * grid_dim.y),
    )
}

pub fn vector_to_flat_idx(v: UVec3, grid_dim: UVec3) -> u32 {
    v.z * grid_dim.x * grid_dim.y +
        v.y * grid_dim.x +
        v.x
}