use crate::{vector_to_flat_idx, USizeVec2};
use bytemuck::{Pod, Zeroable};
use khal_std::glamx::{UVec2, UVec3, Vec2, Vec3Swizzles, Vec4};
use khal_std::macros::{spirv, spirv_bindgen};
use khal_std::sync::workgroup_memory_barrier_with_group_sync;

#[spirv_bindgen]
#[spirv(compute(threads(8, 8, 1)))]
pub fn fdtd2(
    #[spirv(global_invocation_id)] id3: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] field_values: &mut [FieldValues2],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] update_coeffs: &[MaterialConstants2],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] source: &GpuSource2,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] source_vals: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] step_counter: &mut u32,
    #[spirv(uniform, descriptor_set = 0, binding = 5)] grid: &GridInfo2,
) {
    let id2 = id3.xy();
    let grid_dim3 = UVec3::from((grid.grid_dim, 1));

    let outside_grid = id3.cmpge(grid_dim3).any();
    if outside_grid { return; }

    let d = grid.cell_size;
    let idx = (vector_to_flat_idx!(id3, grid_dim3) as usize).min(field_values.len() - 1);
    let cell_incr = USizeVec2::from(grid.i_incr);

    let update_coeffs = update_coeffs[idx];

    // Update H field
    let idx_incremented = (cell_incr + idx)
        .min(USizeVec2::splat(field_values.len() - 1));
    let is_boundary = id2.cmpeq(grid.grid_dim - 1);
    let en_z = field_values[idx].en_z;
    let en_z_i1 = if is_boundary.x { 0. } else { field_values[idx_incremented.x].en_z };
    let en_z_j1 = if is_boundary.y { 0. } else { field_values[idx_incremented.y].en_z };
    let e_curl = Vec2::new(
        (en_z_j1 - en_z) / d.y,
        -(en_z_i1 - en_z) / d.x
    );
    field_values[idx].h += update_coeffs.h_update_coeff * e_curl;

    // Update Dn field
    let idx_decremented = cell_incr.map(|incr| {
        if incr > idx { 0 } else { idx.wrapping_sub(incr) }
    });
    let is_boundary = id2.cmpeq(UVec2::ZERO);
    let h = field_values[idx].h;
    let h_y_i1 = if is_boundary.x { 0. } else { field_values[idx_decremented.x].h.y };
    let h_x_j1 = if is_boundary.y { 0. } else { field_values[idx_decremented.y].h.x };
    let h_curl = (h.y - h_y_i1) / d.x - (h.x - h_x_j1) / d.y;
    field_values[idx].dn_z += grid.dn_z_update_coeff * h_curl;

    // Source injection
    let step_counter_usize = *step_counter as usize;
    let val_idx = step_counter_usize.min(source_vals.len() - 1);
    let enable_source = (idx == source.cell_idx as usize && step_counter_usize < source_vals.len()) as u32 as f32;
    field_values[idx].dn_z += source_vals[val_idx] * enable_source;

    // Update En field
    field_values[idx].en_z = update_coeffs.en_z_update_coeff * field_values[idx].dn_z;

    // Write back results
    // field_values[idx] = field_values[idx];

    if idx == 0 {
        *step_counter += 1;
    }
}

/// All vector field values within a cell

#[derive(Copy, Clone, Pod, Zeroable, Default)]
#[repr(C)]
pub struct FieldValues2 {
    /// H field (NOT normalized). Staggered by `cell_size/2.` in space and `dt/2.` in time.
    pub h: Vec2,
    /// Z component of normalized D field. Normalized such that `dn_z == eps_r * en_z`
    pub dn_z: f32,
    /// Z component of normalized E field. Normalized such that `en_z == e / impedance_0`
    pub en_z: f32,
}

#[derive(Copy, Clone, Pod, Zeroable, Default)]
#[repr(C)]
pub struct GridInfo2 {
    /// Number of cells for each axis (the dimensions of the grid in `cell_size` units.
    pub grid_dim: UVec2,
    pub cell_size: Vec2,
    /// Used for incrementing array index to access neighboring cells when computing curl.
    pub i_incr: UVec2,
    pub dn_z_update_coeff: f32,
    pub _padding: u32
}

#[derive(Copy, Clone, Pod, Zeroable, Default)]
#[repr(C)]
pub struct GridCell2 {
    /// H field (NOT normalized). Staggered by `cell_size/2.` in space and `dt/2.` in time.
    pub h: Vec2,
    /// Z component of normalized D field. Normalized such that `dn_z == eps_r * en_z`
    pub dn_z: f32,
    /// Z component of normalized E field. Normalized such that `en_z == e / impedance_0`
    pub en_z: f32,
    /// The index of the material this grid cell has
    pub material_i: u32,
    pub _padding: [u32; 3],
}

// TODO: rename to UpdateConsts2 probably
#[derive(Copy, Clone, Pod, Zeroable, Default)]
#[repr(C)]
pub struct MaterialConstants2 {
    pub h_update_coeff: Vec2,
    pub en_z_update_coeff: f32,
    pub _padding: [u32; 1]
}

#[derive(Copy, Clone, Pod, Zeroable, Default)]
#[repr(C)]
pub struct GpuSource2 {
    pub cell_idx: u32 // TODO: change to tf/sf later on
}

#[derive(Copy, Clone, Pod, Zeroable, Default)]
#[repr(C)]
pub struct PmlCoefficients2 {
    pub h_coeffs: [Vec2; 4],
    pub dn_z_coeffs: Vec4,
}

#[derive(Copy, Clone, Pod, Zeroable, Default)]
#[repr(C)]
pub struct PmlIntegrations {
    pub e_curl: Vec2,
    pub dn_z: f32,
    pub _padding: u32,
}