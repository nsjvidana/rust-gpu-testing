use std::num::NonZeroU32;
use std::ops::Div;
use glam::{UVec3, Vec3};
use crate::prelude::*;
use khal::backend::{Backend, DispatchGrid, GpuBackend, GpuBackendError, GpuBuffer, GpuPass};
use khal::BufferUsages;
use rapier3d_meshloader::LoadedShape;
use shader_crate::{HFieldCompute, GridCell, GridInfo, MaterialConstants, PointCharge};

pub const PROTON_MASS: f32 = 1.6726219259552e-27;
pub const ELECTRON_MASS: f32 = 9.109383713928e-31;
pub const ELEMENTARY_CHARGE: f32 = 1.602176634e-19;

pub struct FDTDData {
    pub cells: Vec<GridCell>,
    pub material_constants: Vec<MaterialConstants>,
    pub grid_info: GridInfo,
    pub point_charges: Vec<PointCharge>,
}

impl FDTDData {
    pub fn new(grid_info: GridInfo) -> Result<Self> {
        let num_cells = grid_info.grid_dimensions.element_product();
        if num_cells == 0 {
            return Err(Error::BufferSizeZero)
        }
        Ok(Self {
            cells: vec![GridCell::default(); num_cells as usize],
            material_constants: vec![MaterialConstants::free_space(grid_info.dt)],
            grid_info,
            point_charges: vec![PointCharge::default()],
        })
    }
}

pub struct MaxwellEqsBuffers {
    pub cells: GpuBuffer<GridCell>,
    pub material_constants: GpuBuffer<MaterialConstants>,
    pub point_charges: GpuBuffer<PointCharge>,
    pub grid_info: GpuBuffer<GridInfo>,
}

pub fn create_buffers(
    backend: &GpuBackend,
    data: &FDTDData
) -> std::result::Result<MaxwellEqsBuffers, GpuBackendError> {
    Ok(
        MaxwellEqsBuffers {
            cells: backend.init_buffer(
                data.cells.as_slice(),
                BufferUsages::STORAGE | BufferUsages::COPY_SRC
            )?,
            material_constants: backend.init_buffer(
                data.material_constants.as_slice(),
                BufferUsages::STORAGE
            )?,
            point_charges: backend.init_buffer(
                data.point_charges.as_slice(),
                BufferUsages::STORAGE | BufferUsages::COPY_SRC
            )?,
            grid_info: backend.init_buffer(
                &[data.grid_info],
                BufferUsages::UNIFORM
            )?
        }
    )
}

pub fn encode_h_field_compute(
    kernel: &HFieldCompute,
    pass: &mut GpuPass,
    buffers: &mut MaxwellEqsBuffers,
    grid_info: &GridInfo,
) -> std::result::Result<(), GpuBackendError> {
    let workgroup_count = grid_info.grid_dimensions.map(|v| v.div_ceil(4)).to_array();
    kernel.call(
        pass,
        DispatchGrid::Grid(workgroup_count),
        &mut buffers.cells,
        &mut buffers.point_charges,
        &buffers.material_constants,
        &buffers.grid_info
    )
}

/// Compute grid cell size and index dimensions
///
/// This is a helper function, so there is no need to use this function if you know what cell size
/// you want, how many cells on each axis, etc.
///
/// `min_wavelength` & `min_feature_length` must not be zero.
pub fn compute_grid_dimensions(
    grid_dimensions: Vec3,
    min_wavelength: f32,
    min_feature_length: f32,
    cells_per_wavelength: NonZeroU32,
    cells_per_feature_length: NonZeroU32,
) -> (f32, UVec3) {
    let cell_size = (min_wavelength / cells_per_wavelength.get() as f32)
        .min(min_feature_length / cells_per_feature_length.get() as f32);
    (cell_size, grid_dimensions.div(cell_size).ceil().as_uvec3())
}

/// Calculate the size of the smallest feature in a mesh shape.
pub fn min_feature_length(mesh: LoadedShape) -> f32 {
    mesh.raw_mesh.faces.iter().map(|f| {
        let verts = f.map(|i| mesh.raw_mesh.vertices[i as usize]);
        let len0 = (Vec3::from_array(verts[0]) - Vec3::from_array(verts[1]))
            .length_squared();
        let len1 = (Vec3::from_array(verts[0]) - Vec3::from_array(verts[2]))
            .length_squared();
        let len2 = (Vec3::from_array(verts[1]) - Vec3::from_array(verts[2]))
            .length_squared();
        len0.max(len1).max(len2).sqrt()
    }).max_by(|a, b| a.total_cmp(b)).expect("Mesh cannot be empty")
}

/// Calculate the minimum wavelength that will be in the simulation
///
/// `max_freq` is the maximum frequency of a signal that will be injected into the simulation
/// `max_refractive_idx` is the largest refractive index that a material will have
pub fn min_wavelength(max_freq: f32, max_refractive_idx: f32) -> f32 {
    MaterialConstants::C_0 / (max_freq * max_refractive_idx)
}