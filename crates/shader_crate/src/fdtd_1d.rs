use bytemuck::{Pod, Zeroable};
use khal_std::glamx::UVec3;
use khal_std::macros::{spirv, spirv_bindgen};
use khal_std::num_traits::Float;
use khal_std::sync::workgroup_memory_barrier_with_group_sync;
use crate::select_val;

#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn fdtd_1d(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] cells: &mut [GridCell1D],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] materials: &mut [MaterialConstants1D],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] source_vals: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] tfsf_sources: &mut [GpuSource1D],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] boundary: &mut PerfectBoundaryData,
    #[spirv(uniform, descriptor_set = 0, binding = 5)] grid_info: &GridInfo1D,
) {
    let idx = (id.x as usize).min(cells.len() - 1);

    let mat = materials[cells[idx].material_idx as usize];
    // Update Hn from E
    if idx == 0 {
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
    for i in 0..tfsf_sources.len() {
        let src = &mut tfsf_sources[i];
        let src_cell_idx = src.cell_idx as usize;
        if src_cell_idx != idx { continue; }

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

#[derive(Copy, Clone, Pod, Zeroable, Default, Debug)]
#[repr(C)]
pub struct GridInfo1D {
    pub dimensions: f32,
    pub num_cells: u32,
    pub cell_size: f32,
    pub dt: f32,
    pub step_count: u32,
}

impl GridInfo1D {
    pub fn max_values(dimensions: f32) -> Self {
        Self {
            dimensions,
            num_cells: 0,
            cell_size: f32::MAX,
            dt: f32::MAX,
            step_count: 1,
        }
    }

    pub fn set_dimensions(&mut self, dimensions: f32) -> &mut Self {
        self.dimensions = dimensions;
        self.update_cell_count()
    }

    pub fn update_cell_count(&mut self) -> &mut Self {
        self.num_cells = (self.dimensions / self.cell_size).ceil() as u32;
        self
    }

    pub fn min_wavelength(&mut self, f_max: f32, n_max: f32, cells_per_wavelength: u32) -> &mut Self {
        let min_wavelen = MaterialConstants1D::C_0 / (f_max * n_max);
        self.cell_size = self.cell_size.min(min_wavelen / cells_per_wavelength as f32);
        self.update_cell_count()
    }

    pub fn min_feature_length(&mut self, min_feature_length: f32, cells_per_min_len: u32) -> &mut Self {
        self.cell_size = self.cell_size.min(min_feature_length / cells_per_min_len as f32);
        self.update_cell_count()
    }

    pub fn snap_to_critical_dim(&mut self, critical_dim: f32) -> &mut Self {
        let cells_per_crit_dim = (critical_dim / self.cell_size).ceil();
        self.cell_size = critical_dim / cells_per_crit_dim;
        self.update_cell_count()
    }

    /// Sets up `dt` for simulating with a perfect boundary condition.
    ///
    /// Guarantees that the fastest wave in the simulation travels 1 grid cell in exactly
    /// two timesteps.
    pub fn set_cfl_perfect_boundary(&mut self, n_boundary:f32) -> &mut Self {
        self.dt = self.compute_cfl_upper_bound(n_boundary, 2.);
        self
    }

    pub fn set_cfl_condition(&mut self, n_min: f32, safety_margin: f32) -> &mut Self {
        self.dt = self.dt.min(self.compute_cfl_upper_bound(n_min, safety_margin));
        self
    }

    pub fn compute_cfl_upper_bound(&self, n_min: f32, safety_margin: f32) -> f32 {
        (n_min * self.cell_size) / (safety_margin * MaterialConstants1D::C_0)
    }

    /// Set the amount of `dt` time steps per shader dispatch.
    pub fn set_step_count(&mut self, step_count: u32) -> &mut Self {
        self.step_count = step_count;
        self
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
            e_update_coeff: (Self::C_0 * dt) / eps_r,
            hn_update_coeff: (Self::C_0 * dt) / mu_r,
            eps_r,
            mu_r,
            n: Float::sqrt(eps_r * mu_r)
        }
    }
}

/// A Total Field/Scatter Field (TF/SF) electric field source the user can inject into the
/// simulation.
#[derive(Copy, Clone, Pod, Zeroable)]
#[repr(C)]
pub struct GpuSource1D {
    pub start_idx: u32,
    pub end_idx: u32,
    pub curr_idx: u32,
    pub cell_idx: u32,
}