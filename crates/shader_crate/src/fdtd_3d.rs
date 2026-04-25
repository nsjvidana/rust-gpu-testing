use bytemuck::{Pod, Zeroable};
use khal_std::glamx::{Mat3, Mat3A, Mat4, USizeVec3, UVec3, UVec4, Vec3, Vec3A, Vec4, Vec4Swizzles};
use khal_std::macros::{spirv, spirv_bindgen};
use khal_std::num_traits::Float;
use crate::vector_to_flat_idx;

#[spirv_bindgen]
#[spirv(compute(threads(4, 4, 4)))]
pub fn fdtd_3d(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] cells: &mut [GridCell],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] materials: &mut [MaterialConstants],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] source_vals: &[Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] sources: &mut [GpuSource],
    #[spirv(uniform, descriptor_set = 0, binding = 4)] grid: &GridInfo,
) {
    let idx_dims = grid.idx_dims.xyz();
    if id.cmpge(idx_dims).any() { return; }

    let idx = vector_to_flat_idx(id, idx_dims) as usize;

    let mat = materials[cells[idx].material_idx as usize];

    let d = Vec3A::from(grid.cell_size);
    let incr = USizeVec3::new(1, idx_dims.x as _, (idx_dims.x*idx_dims.y) as _);

    // Update Hn from E
    {
        let not_boundary = USizeVec3::from(id.cmplt(idx_dims - UVec3::ONE));

        let incr = incr * not_boundary;
        let e_i_1 = Vec3A::from(cells[idx + incr.x].e) * not_boundary.x as f32;
        let e_j_1 = Vec3A::from(cells[idx + incr.y].e) * not_boundary.y as f32;
        let e_k_1 = Vec3A::from(cells[idx + incr.z].e) * not_boundary.z as f32;
        let e = Vec3A::from(cells[idx].e);

        let e_curl = Vec3A::new(
            (e_j_1.z - e.z)/d.y - (e_k_1.y - e.y)/d.z,
            (e_k_1.x - e.x)/d.z - (e_i_1.z - e.z)/d.x,
            (e_i_1.y - e.y)/d.x - (e_j_1.x - e.x)/d.y,
        );
        cells[idx].hn += Vec3::from(Mat3A::from_mat4(mat.hn_update_coeff) * e_curl);
    }

    // Update E from Hn
    {
        let not_boundary = USizeVec3::from(id.cmpgt(UVec3::ZERO));

        let incr = incr * not_boundary;
        let hn_i_1 = Vec3A::from(cells[idx - incr.x].hn) * not_boundary.x as f32;
        let hn_j_1 = Vec3A::from(cells[idx - incr.y].hn) * not_boundary.y as f32;
        let hn_k_1 = Vec3A::from(cells[idx - incr.z].hn) * not_boundary.z as f32;
        let hn = Vec3A::from(cells[idx].hn);

        let hn_curl = Vec3A::new(
            (hn.z - hn_j_1.z)/d.y - (hn.y - hn_k_1.y)/d.z,
            (hn.x - hn_k_1.x)/d.z - (hn.z - hn_i_1.z)/d.x,
            (hn.y - hn_i_1.y)/d.x - (hn.x - hn_j_1.x)/d.y,
        );
        cells[idx].e += Vec3::from(Mat3A::from_mat4(mat.e_update_coeff) * hn_curl);
    }

    // Soft source injection
    if idx != 0 { return; }
    for i in 0..sources.len() {
        let src = &mut sources[i];
        let source_not_finished = (src.curr_idx <= (src.end_idx - src.start_idx)) as u32;

        let val_idx = (src.start_idx + src.curr_idx) as usize;
        cells[src.cell_idx as usize].e += source_vals[val_idx].xyz() * source_not_finished as f32;
        src.curr_idx += source_not_finished;
    }
}

#[derive(Copy, Clone, Pod, Zeroable, Default, Debug)]
#[repr(C)]
pub struct GridInfo {
    pub dimensions: Vec4,
    pub idx_dims: UVec4,
    pub cell_size: Vec3,
    pub dt: f32,
}

impl GridInfo {
    pub fn max_values(dimensions: Vec3) -> Self {
        Self {
            dimensions: Vec4::from((dimensions, 0.)),
            idx_dims: UVec4::ZERO,
            cell_size: Vec3::splat(f32::MAX),
            dt: f32::MAX,
        }
    }

    pub fn set_dimensions(&mut self, dimensions: Vec3) -> &mut Self {
        self.dimensions = Vec4::from((dimensions, 0.));
        self.update_cell_count()
    }

    pub fn update_cell_count(&mut self) -> &mut Self {
        let new_dims = (self.dimensions.xyz() / self.cell_size).ceil().as_uvec3();
        self.idx_dims = UVec4::from((new_dims, 0));
        self
    }

    pub fn min_wavelength(&mut self, f_max: f32, n_max: f32, cells_per_wavelength: u32) -> &mut Self {
        let min_wavelen = MaterialConstants::C_0 / (f_max * n_max);
        self.cell_size = self.cell_size.min(Vec3::splat(min_wavelen / cells_per_wavelength as f32));
        self.update_cell_count()
    }

    pub fn min_feature_length(&mut self, min_feature_length: f32, cells_per_min_len: u32) -> &mut Self {
        self.cell_size = self.cell_size.min(Vec3::splat(min_feature_length / cells_per_min_len as f32));
        self.update_cell_count()
    }

    pub fn snap_to_critical_dims(&mut self, critical_dims: Vec3) -> &mut Self {
        let cells_per_crit_dim = (critical_dims / self.cell_size).ceil();
        self.cell_size = critical_dims / cells_per_crit_dim;
        self.update_cell_count()
    }

    pub fn courant_stability_condition(&mut self, n_min: f32, safety_margin: f32) -> &mut Self {
        let new_dt = n_min /
            (safety_margin * MaterialConstants::C_0 * self.cell_size.map(|v| v*v).recip().element_sum().sqrt());
        self.dt = self.dt.min(new_dt);
        self
    }

    /// Adjust dt to account for gaussian pulse.
    ///
    /// Have `cells_resolution >= 10` for better results
    pub fn account_for_pulse(&mut self, tau: f32, cells_resolution: u32) -> &mut Self {
        self.dt = self.dt.min(tau / cells_resolution as f32);
        self
    }
}

#[derive(Copy, Clone, Pod, Zeroable, Default)]
#[repr(C)]
pub struct GridCell {
    pub e: Vec3,
    pub _padding0: u32,
    pub hn: Vec3,
    pub material_idx: u32,
}

#[derive(Copy, Clone, Pod, Zeroable)]
#[repr(C)]
pub struct MaterialConstants {
    pub hn_update_coeff: Mat4,
    pub e_update_coeff: Mat4,
}

impl MaterialConstants {
    /// Speed of light in free space
    pub const C_0: f32 = 299792458.0;

    pub fn new_linear(eps_r: f32, mu_r: f32, dt: f32) -> Self {
        let hn = -(Self::C_0 * dt)/mu_r;
        let e = (Self::C_0 * dt)/eps_r;
        Self {
            hn_update_coeff: Mat4::from_diagonal(Vec4::new(hn, hn, hn, 0.)),
            e_update_coeff: Mat4::from_diagonal(Vec4::new(e, e, e, 0.)),
        }
    }

    pub fn anisotropic(eps_r: Mat3A, mu_r: Mat3A, dt: f32) -> Option<Self> {
        let hn = mu_r / (-Self::C_0 * dt);
        let e = eps_r / (Self::C_0 * dt);
        Some(Self {
            hn_update_coeff: Mat4::from_mat3a(hn.try_inverse()?),
            e_update_coeff: Mat4::from_mat3a(e.try_inverse()?),
        })
    }
}

#[derive(Copy, Clone, Pod, Zeroable)]
#[repr(C)]
pub struct GpuSource {
    pub start_idx: u32,
    pub end_idx: u32,
    pub curr_idx: u32,
    pub cell_idx: u32,
}