#![no_std]

use bytemuck::{Pod, Zeroable};
use spirv_std::glam::{UVec3, Vec3, Vec3Swizzles};
use spirv_std::num_traits::Float;
use spirv_std::spirv;

/// The Coulomb constant 1/(4πε_0)
const COULOMB_K: f32 = 8.98755178597214e9;

// TODO: replace spirv with cfg_attr(feature = "dim2/3", spirv(compute(threads(64, 64,)) etc.)
#[spirv(compute(threads(4, 4, 4)))]
pub fn e_field_compute(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] cells: &mut [GridCell],
    // TODO: magnetic field (b_field)
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] pt_charges: &mut [PointCharge],
    #[spirv(uniform, descriptor_set = 0, binding = 2)] grid: &GridInfo,
) {
    if id.cmpge(grid.grid_dimensions.xyz()).any() {
        return;
    }

    //Flat index of this invocation's grid cell
    let flat_idx = (id.z * grid.grid_dimensions.x * grid.grid_dimensions.y +
              id.y * grid.grid_dimensions.x +
              id.x) as usize;
    let cell_position = grid.position + id.as_vec3() * grid.cell_size;
    for i in 0..pt_charges.len() {
        let pt = &pt_charges[i];
        let r = cell_position - pt.position;
        let r_magnitude_sq = r.length_squared();
        let r_hat =
            if r_magnitude_sq != 0. { r / r_magnitude_sq.sqrt() } else { Vec3::ZERO };
        cells[flat_idx].e += (COULOMB_K * pt.q / r_magnitude_sq) * r_hat;
    }
}

#[derive(Copy, Clone, Pod, Zeroable, Default)]
#[repr(C)]
pub struct GridCell {
    /// Electric field vector
    pub e: Vec3,
    _padding0: u32,
    /// The magnetic field vector at a grid position offset by half the cell size.
    pub b: Vec3,
    _padding1: u32,
}

#[derive(Copy, Clone, Pod, Zeroable, Default)]
#[repr(C)]
pub struct PointCharge {
    pub velocity: Vec3,
    /// The charge of the point charge (unit: Coulomb)
    pub q: f32,
    /// The position of the charge (unit: m)
    pub position: Vec3,
    /// Mass of this point charge (unit: kg)
    pub mass: f32,
}

#[derive(Copy, Clone, Pod, Zeroable, Default)]
#[repr(C)]
pub struct GridInfo {
    /// Position of the grid's origin cell (the "0,0" cell. NOT the cell at the center)
    pub position: Vec3,
    _padding: u32,
    pub grid_dimensions: UVec3,
    /// The length of one dimension of the grid's cell
    pub cell_size: f32,
}

impl GridInfo {
    pub fn new(
        position: Vec3,
        grid_dimensions: UVec3,
        cell_size: f32,
    ) -> Self {
        Self {
            position,
            grid_dimensions,
            cell_size,
            ..Default::default()
        }
    }
}

#[spirv(compute(threads(64)))]
pub fn double_me(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] floats: &mut [f32],
) {
    let idx = id.x as usize;
    floats[idx] *= 2.0;
}