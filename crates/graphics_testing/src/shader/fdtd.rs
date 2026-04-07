use crate::prelude::*;
use khal::backend::{Backend, DispatchGrid, GpuBackend, GpuBackendError, GpuBuffer, GpuPass};
use khal::BufferUsages;
use shader_crate::{EFieldCompute, GridCell, GridInfo, MaterialConstants, PointCharge};

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
            material_constants: vec![MaterialConstants::default()],
            grid_info,
            point_charges: Vec::new(),
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
                BufferUsages::STORAGE
            )?
        }
    )
}

pub fn encode_e_field_compute(
    kernel: &EFieldCompute,
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