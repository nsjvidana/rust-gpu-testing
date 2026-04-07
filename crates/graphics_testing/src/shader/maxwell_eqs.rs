use khal::backend::{Backend, DispatchGrid, GpuBackend, GpuBackendError, GpuBuffer, GpuPass};
use khal::BufferUsages;
use crate::prelude::*;
use shader_crate::{EFieldCompute, GridCell, GridInfo, PointCharge};

pub const PROTON_MASS: f32 = 1.6726219259552e-27;
pub const ELECTRON_MASS: f32 = 9.109383713928e-31;
pub const ELEMENTARY_CHARGE: f32 = 1.602176634e-19;

pub struct MaxwellEqsData {
    pub cells: Vec<GridCell>,
    pub grid_info: GridInfo,
    pub point_charges: Vec<PointCharge>,
}

impl MaxwellEqsData {
    pub fn new(grid_info: GridInfo, point_charges: Vec<PointCharge>) -> Result<Self> {
        let num_cells = grid_info.grid_dimensions.element_product();
        if num_cells == 0 {
            return Err(Error::BufferSizeZero)
        }
        Ok(Self {
            cells: vec![GridCell::default(); num_cells as usize],
            grid_info,
            point_charges,
        })
    }
}

pub struct MaxwellEqsBuffers {
    pub cells: GpuBuffer<GridCell>,
    pub point_charges: GpuBuffer<PointCharge>,
    pub grid_info: GpuBuffer<GridInfo>,
}

pub fn create_buffers(
    backend: &GpuBackend,
    data: &MaxwellEqsData
) -> std::result::Result<MaxwellEqsBuffers, GpuBackendError> {
    let cells = backend.init_buffer(
        data.cells.as_slice(),
        BufferUsages::STORAGE | BufferUsages::COPY_SRC
    )?;
    let pt_charges = backend.init_buffer(
        data.point_charges.as_slice(),
        BufferUsages::STORAGE | BufferUsages::COPY_SRC
    )?;
    let grid_info = backend.init_buffer(
        &[data.grid_info],
        BufferUsages::STORAGE
    )?;

    Ok(
        MaxwellEqsBuffers {
            cells,
            point_charges: pt_charges,
            grid_info
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
        &buffers.grid_info,
    )
}