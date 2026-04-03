#![no_std]

use spirv_std::glam::{UVec3, UVec4, Vec3, Vec3Swizzles, Vec4, Vec4Swizzles};
use spirv_std::spirv;
use spirv_std::num_traits::Float;

/// The Coulomb constant 1/(4πε_0)
const COULOMB_K: f32 = 8.98755178597214e9;

// TODO: replace spirv with cfg_attr(feature = "dim2/3", spirv(compute(threads(64, 64,)) etc.)
#[spirv(compute(threads(4, 4, 4)))]
pub fn e_field_compute(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] e_field: &mut [Vec4], // Vec4 for 16-byte alignment
    // TODO: magnetic field (b_field)
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] pt_charges: &mut [PointCharge],
    #[spirv(uniform, descriptor_set = 0, binding = 2)] grid: &GridValues,
) {
    if id.cmpge(grid.grid_dimensions.xyz()).any() {
        return;
    }

    //Flat index of this invocation's grid cell
    let flat_idx = (id.z * grid.grid_dimensions.x * grid.grid_dimensions.y +
              id.y * grid.grid_dimensions.x +
              id.x) as usize;
    let cell_position = grid.position + Vec4::from((id.as_vec3(), 0.)) * grid.cell_size;
    for i in 0..pt_charges.len() {
        let pt = &pt_charges[i];
        let r = cell_position - pt.position;
        let r_magnitude_sq = r.length_squared();
        let r_hat =
            if r_magnitude_sq != 0. { r / r_magnitude_sq.sqrt() } else { Vec4::ZERO };
        let e = (COULOMB_K * pt.q / r_magnitude_sq) * r_hat;
        e_field[flat_idx] += Vec4::new(e.x, e.y, e.z, 0.);
    }
}

pub struct PointCharge {
    /// The charge of the point charge (unit: Coulomb)
    pub q: f32,
    /// Mass of this point charge (unit: kg)
    pub mass: f32,
    /// The position of the charge (unit: m)
    pub position: Vec4,
}

pub struct GridValues {
    /// Position of the grid's origin cell (the "0,0" cell. NOT the cell at the center)
    pub position: Vec4,
    pub grid_dimensions: UVec4,
    /// The length of one dimension of the grid's cell
    pub cell_size: f32,
}

#[spirv(compute(threads(64)))]
pub fn double_me(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] floats: &mut [f32],
) {
    let idx = id.x as usize;
    floats[idx] *= 2.0;
}