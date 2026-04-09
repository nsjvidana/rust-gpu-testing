#![cfg_attr(target_arch = "spirv", no_std)]

use bytemuck::{Pod, Zeroable};
use khal_std::glamx::{BVec3, Mat3, Mat3A, Mat4, MatExt, USizeVec3, UVec3, Vec3, Vec3Swizzles, Vec4, Vec4Swizzles};
use khal_std::macros::{spirv, spirv_bindgen};
use khal_std::num_traits::Float;

/// The Coulomb constant 1/(4πε_0)
const COULOMB_K: f32 = 8.98755178597214e9;

// TODO: replace spirv with cfg_attr(feature = "dim2/3", spirv(compute(threads(64, 64,)) etc.)
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn h_field_compute(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] cells: &mut [GridCell],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] pt_charges: &mut [PointCharge],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] material: &[MaterialConstants],
    // TODO: make this parameter into a uniform (khal doesn't support uniform buffers right now).
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] grid: &[GridInfo],
) {
    let grid = &grid[0];
    if id.cmpge(grid.grid_dimensions).any() {
        return;
    }

    let idx = vector_to_flat_idx(id, grid.grid_dimensions) as usize; // Flat index of this invocation's grid cell

    let dim = grid.grid_dimensions.as_usizevec3();

    // Get E-field components from surrounding cells with Dirichlet Boundary Condition
    // Hn from E stage
    {
        let is_not_boundary = id.cmplt(grid.grid_dimensions - UVec3::ONE);
        let is_not_boundary =
            USizeVec3::new(is_not_boundary.x as usize, is_not_boundary.y as usize, is_not_boundary.z as usize);
        let mut incr = USizeVec3::new(1, dim.x, dim.x*dim.y) * is_not_boundary;
        let e_i = cells[idx + incr.x].e * is_not_boundary.x as f32; // E at the adjacent cell in the +x direction
        let e_j = cells[idx + incr.y].e * is_not_boundary.y as f32; // E at the adjacent cell in the +y direction
        let e_k = cells[idx + incr.z].e * is_not_boundary.z as f32; // E at the adjacent cell in the +z direction
        let e = cells[idx].e;

        // Compute change in normalized H-field and apply it
        let hn_coeff_inv = material[cells[idx].material_idx as usize].hn_update_coeff_inv;
        let e_curl = Vec4::new(
            (e_j.z - e.z) - (e_k.y - e.y),
            (e_k.x - e.x) - (e_i.z - e.z),
            (e_i.y - e.y) - (e_j.x - e.x),
            0.
        ) / grid.cell_size;
        let delta_hn = hn_coeff_inv * e_curl;
        cells[idx].hn += delta_hn.xyz();
    }

    // E from Hn stage
    {
        let is_not_boundary = id.cmpgt(UVec3::ZERO);
        let is_not_boundary =
            USizeVec3::new(is_not_boundary.x as usize, is_not_boundary.y as usize, is_not_boundary.z as usize);
        let mut incr = USizeVec3::new(1, dim.x, dim.x*dim.y) * is_not_boundary;
        let hn_i = cells[idx - incr.x].e * is_not_boundary.x as f32; // Hn at the adjacent cell in the -x direction
        let hn_j = cells[idx - incr.y].e * is_not_boundary.y as f32; // Hn at the adjacent cell in the -y direction
        let hn_k = cells[idx - incr.z].e * is_not_boundary.z as f32; // Hn at the adjacent cell in the -z direction
        let hn = cells[idx].hn;

        // Compute change in E-field and apply it
        let e_coeff_inv = material[cells[idx].material_idx as usize].e_update_coeff_inv;
        let hn_curl = Vec4::new(
            (hn_j.z - hn.z) - (hn_k.y - hn.y),
            (hn_k.x - hn.x) - (hn_i.z - hn.z),
            (hn_i.y - hn.y) - (hn_j.x - hn.x),
            0.
        ) / grid.cell_size;
        let delta_e = e_coeff_inv * hn_curl;
        cells[idx].e += delta_e.xyz();
    }
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
    /// Normalized Magnetic Field Intensity (staggered by half-timestep and half-cell-size)
    ///
    /// Normalized in a way such that `hn / MaterialConstants::N_0` is the actual H-field vector
    /// (Hn and E are of the same order of magnitude)
    pub hn: Vec3,
    pub _padding1: u32,
    /// Electric Field Intensity vector
    pub d: Vec3,
    pub _padding2: u32,
}

/// Material constants for linear, isotropic, and non-dispersive materials ONLY.
///
/// All values (eps/permittivity and mu/permeability) are relative to free space.
#[derive(Copy, Clone, Pod, Zeroable, Debug)]
#[repr(C)]
pub struct MaterialConstants {
    /// Update coefficient for the Hn-field solving stage
    pub hn_update_coeff_inv: Mat4,
    /// Update coefficient for the E-field solving stage
    pub e_update_coeff_inv: Mat4,
}

impl MaterialConstants {
    /// Permittivity of free space
    pub const EPS_0: f32 = 8.854187818814e-12;
    /// Permeability of free space
    pub const MU_0: f32 = 1.2566370612720e-6;
    /// Speed of light in free space
    pub const C_0: f32 = 299792458.0;
    /// Impedance of free space
    pub const N_0: f32 = 376.73031346177;

    /// Material constants of free space
    pub fn free_space(dt: f32) -> Self {
        Self::new_linear(1., 1., dt).unwrap()
    }

    /// Creates material constants for a linear/isotropic/non-dispersive material
    ///
    /// `eps_r` and `mu_r` are relative to eps and mu of free space.
    pub fn new_linear(eps_r: f32, mu_r: f32, dt: f32) -> Option<Self> {
        if eps_r == 0. || mu_r == 0. {
            return None;
        }
        Some(
            Self {
                hn_update_coeff_inv: Mat4::from_diagonal(
                    Vec4::from((Vec3::splat(-Self::C_0*dt / mu_r), 1.))
                ),
                e_update_coeff_inv: Mat4::from_diagonal(
                    Vec4::from((Vec3::splat(Self::C_0*dt / eps_r), 1.))
                ),
            }
        )
    }

    /// Creates material constants for an anisotropic material
    ///
    /// `eps_r` and `mu_r` are relative to eps and mu of free space.
    ///
    /// # Panics
    /// When either `eps` or `mu` matrices are non-invertible
    pub fn new_anisotropic(eps_r: Mat3, mu_r: Mat3, dt: f32) -> Option<Self> {
        let Some((e_field_update_coeff_inv, h_field_update_coeff_inv)) =
            (eps_r / (Self::C_0 * dt))
                .try_inverse()
                .zip((-mu_r / (Self::C_0 * dt)).try_inverse()) else { return None };
        Some(
            Self {
                hn_update_coeff_inv: Mat4::from_mat3(h_field_update_coeff_inv),
                e_update_coeff_inv: Mat4::from_mat3(e_field_update_coeff_inv),
                ..Default::default()
            }
        )
    }
}

impl Default for MaterialConstants {
    fn default() -> Self {
        Self::free_space(GridInfo::DEFAULT_DT)
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

#[derive(Copy, Clone, Pod, Zeroable)]
#[repr(C)]
pub struct GridInfo {
    /// Position of the grid's origin cell (the "0,0" cell. NOT the cell at the center)
    pub position: Vec3,
    pub dt: f32,
    pub grid_dimensions: UVec3,
    /// The length of one dimension of the grid's cell
    pub cell_size: f32,
}

impl GridInfo {
    /// 1μs is the default time step.
    pub const DEFAULT_DT: f32 = 1.0e-6;
    pub fn new(
        position: Vec3,
        grid_dimensions: UVec3,
        cell_size: f32,
        dt: f32
    ) -> Self {
        Self {
            position,
            grid_dimensions,
            cell_size,
            dt
        }
    }
}

impl Default for GridInfo {
    fn default() -> Self {
        Self {
            dt: Self::DEFAULT_DT,
            cell_size: 1.,
            grid_dimensions: UVec3::ONE,
            position: Vec3::ZERO,
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