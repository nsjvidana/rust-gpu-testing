use bytemuck::{Pod, Zeroable};
use khal_std::glamx::{USizeVec2, USizeVec3, UVec2, UVec3, Vec2, Vec3Swizzles};
use khal_std::macros::{spirv, spirv_bindgen};
use crate::vector_to_flat_idx;

#[spirv_bindgen]
#[spirv(compute(threads(8, 8, 1)))]
pub fn fdtd2(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] cells: &mut [GridCell2],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] materials: &[MaterialConstants2],
    #[spirv(uniform, descriptor_set = 0, binding = 2)] grid: &GridInfo2,
) {
    let id3 = id.as_usizevec3();
    let id = id3.xy();
    let n_cells = grid.n_cells.as_usizevec2();
    let n_cells3 = USizeVec3::from((n_cells, 1));
    // There's probably no workgroup barriers needed for 2d fdtd so just return at extra invocations
    let cmp_i = id.cmpge(n_cells);
    if cmp_i.any() || id3.z > 0 { return; }

    let i = vector_to_flat_idx(id3, n_cells3);
    let mat = materials[cells[i].material_i as usize];
    let d = grid.cell_size;
    let i_incr = grid.i_incr.as_usizevec2();

    // Update H from En
    let not_h_boundary = id.cmplt(n_cells - 1);
    let en_1_indices = (i + i_incr).min(n_cells - 1);
    let en_z = cells[i].en_z;
    let en_x1_z = if not_h_boundary.x { cells[en_1_indices.x].en_z } else { 0. };
    let en_y1_z = if not_h_boundary.y { cells[en_1_indices.y].en_z } else { 0. };
    let en_curl_xy = Vec2::new(
        (en_y1_z - en_z) / d.y,
        -(en_x1_z - en_z) / d.x,
    );
    cells[i].h += mat.h_update_coeff * en_curl_xy;

    // Update Dn/En from H
    let not_dn_boundary = id.cmpgt(USizeVec2::ZERO);
    let h_1_indices = if i > 0 { USizeVec2::splat(i).wrapping_sub(i_incr) } else { USizeVec2::ZERO };
    let h = cells[i].h;
    let h_x1_y = if not_dn_boundary.x { cells[h_1_indices.x].h.y } else { 0. };
    let h_y1_x = if not_dn_boundary.y { cells[h_1_indices.y].h.x } else { 0. };
    let h_curl_z = (h.y - h_x1_y) / d.x - (h.x - h_y1_x) / d.y;
    cells[i].dn_z += grid.dn_z_update_coeff * h_curl_z;
    cells[i].en_z = mat.en_z_update_coeff * cells[i].dn_z;
}

#[derive(Copy, Clone, Pod, Zeroable, Default)]
#[repr(C)]
pub struct GridInfo2 {
    /// Number of cells for each axis (the dimensions of the grid in `cell_size` units.
    pub n_cells: UVec2,
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