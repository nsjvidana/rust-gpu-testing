#![cfg_attr(target_arch = "spirv", no_std)]

use bytemuck::{Pod, Zeroable};
use khal_std::glamx::{Mat3, UVec3, Vec3, Vec3Swizzles};
use khal_std::macros::{spirv, spirv_bindgen};
use khal_std::num_traits::Float;

/// The Coulomb constant 1/(4πε_0)
const COULOMB_K: f32 = 8.98755178597214e9;

// TODO: replace spirv with cfg_attr(feature = "dim2/3", spirv(compute(threads(64, 64,)) etc.)
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn e_field_compute(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] cells: &mut [GridCell],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] pt_charges: &mut [PointCharge],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] material: &[MaterialConstants],
    // TODO: make this parameter into a uniform (khal doesn't support uniform buffers right now).
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] grid: &[GridInfo],
) {
    let grid = &grid[0];
    if id.cmpge(grid.grid_dimensions.xyz()).any() {
        return;
    }

    //Flat index of this invocation's grid cell
    let flat_idx = (id.z * grid.grid_dimensions.x * grid.grid_dimensions.y +
              id.y * grid.grid_dimensions.x +
              id.x) as usize;
}

fn pt_charge_e_field(pt: &PointCharge, cell_position: Vec3) -> Vec3 {
    let r = cell_position - pt.position;
    let r_magnitude_sq = r.length_squared();
    let r_hat =
        if r_magnitude_sq != 0. { r / r_magnitude_sq.sqrt() } else { Vec3::ZERO };
    (COULOMB_K * pt.q / r_magnitude_sq) * r_hat
}

#[derive(Copy, Clone, Pod, Zeroable, Default)]
#[repr(C)]
pub struct GridCell {
    /// Electric field vector
    pub e: Vec3,
    /// The index of the material this cell simulates. Material constants will be located using
    /// this index.
    ///
    /// Material 0 is expected to be permittivity/permeability of free space. The user has control
    /// over which material constants are passed to the shader though.
    pub material_idx: u32,
    /// Magnetic Field vector (staggered by half-timestep and half-cell-size)
    pub b: Vec3,
    pub _padding0: u32,
    /// Magnetic Field Intensity vector (staggered by half-timestep and half-cell-size)
    pub h: Vec3,
    pub _padding1: u32,
    /// Electric Field Intensity vector
    pub d: Vec3,
    pub _padding2: u32,
}

/// Material constants for linear, isotropic, and non-dispersive materials ONLY.
///
/// TODO: The vectors here hold the values of the diagonals of the permittivity and
///       permeability tensors only.
#[derive(Copy, Clone, Pod, Zeroable)]
#[repr(C)]
pub struct MaterialConstants {
    /// Permittivity of the material.
    pub eps: f32,
    /// Permeability of the material.
    pub mu: f32,
    pub _padding0: [u32; 2],
}

impl MaterialConstants {
    /// Permittivity of free space
    const EPS_0: f32 = 8.854187818814e-12;
    /// Permeability of free space
    const MU_0: f32 = 1.2566370612720e-6;
    /// Speed of light in free space
    const C_0: f32 = 299792458.0;
    /// Impedance of free space
    const N_0: f32 = 376.73031346177;
}

impl Default for MaterialConstants {
    /// Free space is default material.
    fn default() -> Self {
        Self {
            eps: Self::EPS_0,
            mu: Self::MU_0,
            _padding0: [0; 2],
        }
    }
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

#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn double_me(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] floats: &mut [f32],
) {
    let idx = id.x as usize;
    floats[idx] *= 2.0;
}