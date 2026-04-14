use bytemuck::{Pod, Zeroable};
use khal_std::glamx::UVec3;
use khal_std::macros::{spirv, spirv_bindgen};
use khal_std::num_traits::Float;
use crate::select_val;

#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn fdtd_1d(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] cells: &mut [GridCell1D],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] materials: &mut [MaterialConstants1D],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] source_vals: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] sources: &mut [GpuSource1D],
    #[spirv(uniform, descriptor_set = 0, binding = 4)] grid_info: &GridInfo1D,
) {
    let idx = id.x as usize;
    if idx >= cells.len() { return; }

    let mat = materials[cells[idx].material_idx as usize];

    let is_not_boundary = idx < cells.len()-1;
    let next_idx = idx + is_not_boundary as usize;
    let e_y1 = select_val!(is_not_boundary, cells[next_idx].e_y, 0., f32);
    cells[idx].hn_x += mat.hn_update_coeff * (e_y1 - cells[idx].e_y) / grid_info.cell_size;

    let is_not_boundary = idx > 0;
    let prev_idx = idx - is_not_boundary as usize;
    let hn_x1 = select_val!(is_not_boundary, cells[prev_idx].hn_x, 0., f32);
    cells[idx].e_y += mat.e_update_coeff * (cells[idx].hn_x - hn_x1) / grid_info.cell_size;

    // Soft source injection
    if idx != 0 { return; }
    for i in 0..sources.len() {
        let src = &mut sources[i];
        let source_not_finished = (src.curr_idx <= (src.end_idx - src.start_idx)) as u32;
        let val_idx = (src.start_idx + src.curr_idx) as usize;
        cells[src.cell_idx as usize].e_y += source_vals[val_idx] * source_not_finished as f32;
        src.curr_idx += source_not_finished;
    }
}

#[derive(Copy, Clone, Pod, Zeroable, Default, Debug)]
#[repr(C)]
pub struct GridInfo1D {
    pub dimensions: f32,
    pub num_cells: u32,
    pub cell_size: f32,
    pub dt: f32,
}

impl GridInfo1D {
    pub fn max_values(dimensions: f32) -> Self {
        Self {
            dimensions,
            num_cells: 0,
            cell_size: f32::MAX,
            dt: f32::MAX,
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

    pub fn courant_stability_condition(&mut self, n_min: f32, safety_margin: f32) -> &mut Self {
        let cell_size_min = self.cell_size;
        self.dt = self.dt.min((n_min * cell_size_min) / (safety_margin * MaterialConstants1D::C_0));
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
pub struct GridCell1D {
    pub e_y: f32,
    pub hn_x: f32,
    pub material_idx: u32,
}

#[derive(Copy, Clone, Pod, Zeroable)]
#[repr(C)]
pub struct MaterialConstants1D {
    pub hn_update_coeff: f32,
    pub e_update_coeff: f32,
}

impl MaterialConstants1D {
    /// Speed of light in free space
    pub const C_0: f32 = 299792458.0;

    pub fn new(eps_r: f32, mu_r: f32, dt: f32) -> Self {
        Self {
            e_update_coeff: (Self::C_0 * dt) / eps_r,
            hn_update_coeff: -(Self::C_0 * dt) / mu_r
        }
    }
}

#[derive(Copy, Clone, Pod, Zeroable)]
#[repr(C)]
pub struct GpuSource1D {
    pub start_idx: u32,
    pub end_idx: u32,
    pub curr_idx: u32,
    pub cell_idx: u32,
}