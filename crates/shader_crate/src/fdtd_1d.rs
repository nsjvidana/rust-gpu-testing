use crate::{e_i, select_val, GpuComplexPolar};
use bytemuck::{Pod, Zeroable};
use khal_std::glamx::{UVec3, Vec2, Vec4, Vec4Swizzles};
use khal_std::macros::{spirv, spirv_bindgen};
use khal_std::num_traits::Float;
use khal_std::sync::workgroup_memory_barrier_with_group_sync;

#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn fdtd_1d(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] cells: &mut [GridCell1D],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] materials: &mut [MaterialConstants1D],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] source_vals: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] tfsf_source: &mut GpuSource1D,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] boundary: &mut PerfectBoundaryData,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] timestep_counter: &mut u32,
    #[spirv(uniform, descriptor_set = 0, binding = 6)] grid_info: &GridInfo1D,
) {
    let idx = (id.x as usize).min(cells.len() - 1);

    let mat = materials[cells[idx].material_idx as usize];
    // Update Hn from E
    if idx == 0 {
        *timestep_counter += 1;
        boundary.hn_x2 = boundary.hn_x1;
        boundary.hn_x1 = cells[idx].hn_x;
    }
    workgroup_memory_barrier_with_group_sync();
    let is_not_boundary = idx < cells.len()-1;
    let e_y1 = select_val!(is_not_boundary, cells[idx + is_not_boundary as usize].e_y, boundary.e_y2, f32);
    cells[idx].hn_x += mat.hn_update_coeff * (e_y1 - cells[idx].e_y) / grid_info.cell_size;

    // Update E from Hn
    if idx == 0 {
        boundary.e_y2 = boundary.e_y1;
        boundary.e_y1 = cells[cells.len() - 1].e_y;
    }
    workgroup_memory_barrier_with_group_sync();
    let is_not_boundary = idx > 0;
    let hn_x1 = select_val!(is_not_boundary, cells[idx - is_not_boundary as usize].hn_x, boundary.hn_x2, f32);
    cells[idx].e_y += mat.e_update_coeff * (cells[idx].hn_x - hn_x1) / grid_info.cell_size;

    // Soft source injection
    let src = tfsf_source;
    let src_cell_idx = src.cell_idx as usize;
    if src_cell_idx == idx {
        let source_not_finished = (src.curr_idx <= (src.end_idx - src.start_idx)) as u32;
        let val_idx = (src.start_idx + src.curr_idx) as usize;
        cells[src_cell_idx].e_y += source_vals[val_idx] * source_not_finished as f32;

        // TF/SF correction terms
        let mat_idx_1 = cells[src_cell_idx - 1].material_idx as usize;
        let mat_1 = materials[mat_idx_1];
        let src_e_y = source_vals[val_idx];
        cells[src_cell_idx - 1].hn_x -= mat_1.hn_update_coeff * src_e_y / grid_info.cell_size;

        let dt_delay = (mat.n * grid_info.cell_size / MaterialConstants1D::C_0 + grid_info.dt) / 2.;
        let dt_delay_idx = (dt_delay / grid_info.dt) as usize;
        let src_val_idx = (val_idx + dt_delay_idx).min(source_vals.len() - 1);
        let src_hn_x = Float::sqrt(mat.eps_r / mat.mu_r) * source_vals[src_val_idx];
        cells[src_cell_idx].e_y -= mat.e_update_coeff * src_hn_x / grid_info.cell_size;

        src.curr_idx += source_not_finished;
    }
    workgroup_memory_barrier_with_group_sync()
}

#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn precompute_dft_kernels_1d(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] dft_kernels: &mut [GpuComplexPolar],
    #[spirv(uniform, descriptor_set = 0, binding = 1)] grid: &GridInfo1D,
    #[spirv(uniform, descriptor_set = 0, binding = 2)] dft: &DftInfo1D,
) {
    let i = id.x as usize;
    if i >= dft_kernels.len() { return; }

    let f = dft.f_start + dft.f_increment * i as f32;
    dft_kernels[i] = e_i!(-core::f32::consts::TAU * f * grid.dt);
}

#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn dft_1d(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] dft_kernels: &[GpuComplexPolar],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] reflectance: &mut [GpuComplexPolar],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] transmittance: &mut [GpuComplexPolar],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] source: &mut [GpuComplexPolar],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] source_vals: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] cells: &[GridCell1D],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] timestep_counter: &u32,
) {
    let i = id.x as usize;
    if i >= dft_kernels.len() { return; }

    let m = *timestep_counter as f32;
    let src_i = (*timestep_counter as usize).min(source_vals.len() - 1);
    let src = source_vals[src_i];

    let k = dft_kernels[i].powf(m);
    reflectance[i] += k * cells[0].e_y;
    transmittance[i] += k * cells[cells.len()-1].e_y;
    source[i] += k * src;
}

#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn finish_dft_1d(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] reflectance: &mut [GpuComplexPolar],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] transmittance: &mut [GpuComplexPolar],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] source: &mut [GpuComplexPolar],
    #[spirv(uniform, descriptor_set = 0, binding = 3)] grid: &GridInfo1D,
) {
    let i = id.x as usize;
    if i >= reflectance.len() { return; }

    transmittance[i] *= grid.dt;
    reflectance[i] *= grid.dt;
    source[i] *= grid.dt;
}

#[derive(Copy, Clone, Pod, Zeroable, Default, Debug)]
#[repr(C)]
pub struct GridInfo1D {
    pub dimensions: f32,
    pub num_cells: u32,
    pub cell_size: f32,
    pub dt: f32,
    pub steps_per_call: u32,
}

impl GridInfo1D {
    pub fn max_values(dimensions: f32) -> Self {
        Self {
            dimensions,
            num_cells: 0,
            cell_size: f32::MAX,
            dt: f32::MAX,
            steps_per_call: 1,
        }
    }
}

#[derive(Copy, Clone, Pod, Zeroable, Default)]
#[repr(C)]
pub struct GridCell1D {
    pub e_y: f32,
    pub hn_x: f32,
    pub material_idx: u32,
}

#[derive(Copy, Clone, Pod, Zeroable, Default)]
#[repr(C)]
pub struct DftInfo1D {
    pub f_start: f32,
    pub f_increment: f32,
}

#[derive(Copy, Clone, Pod, Zeroable, Default)]
#[repr(C)]
pub struct PerfectBoundaryData {
    pub hn_x1: f32,
    pub hn_x2: f32,
    pub e_y1: f32,
    pub e_y2: f32,
}

#[derive(Copy, Clone, Pod, Zeroable, Default)]
#[repr(C)]
pub struct MaterialConstants1D {
    pub hn_update_coeff: f32,
    pub e_update_coeff: f32,
    pub eps_r: f32,
    pub mu_r: f32,
    /// Refractive index
    pub n: f32
}

impl MaterialConstants1D {
    /// Speed of light in free space
    pub const C_0: f32 = 299792458.0;

    pub fn new(eps_r: f32, mu_r: f32, dt: f32) -> Self {
        Self {
            e_update_coeff: Self::compute_e_update_coeff(eps_r, dt),
            hn_update_coeff: Self::compute_hn_update_coeff(mu_r, dt),
            eps_r,
            mu_r,
            n: Float::sqrt(eps_r * mu_r)
        }
    }

    pub fn compute_e_update_coeff(eps_r: f32, dt: f32) -> f32 {
        (Self::C_0 * dt) / eps_r
    }

    pub fn compute_hn_update_coeff(mu_r: f32, dt: f32) -> f32 {
        (Self::C_0 * dt) / mu_r
    }
}

/// A Total Field/Scatter Field (TF/SF) electric field source the user can inject into the
/// simulation.
#[derive(Copy, Clone, Pod, Zeroable, Default)]
#[repr(C)]
pub struct GpuSource1D {
    pub start_idx: u32,
    pub end_idx: u32,
    pub curr_idx: u32,
    pub cell_idx: u32,
}