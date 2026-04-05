use crate::prelude::*;
use crate::shader::{ComputeBuffer, ComputeShader};
use glam::Vec4;
use shader_crate::{GridCell, GridInfo, PointCharge};
use std::borrow::Cow;
use wgpu::{Device, Queue, SubmissionIndex};

pub const PROTON_MASS: f32 = 1.6726219259552e-27;
pub const ELECTRON_MASS: f32 = 9.109383713928e-31;
pub const ELEMENTARY_CHARGE: f32 = 1.602176634e-19;

/// Interface for interacting with the Maxwell's Equations shader.
pub struct MaxwellEqsCompute {
    /// The underlying compute shader.
    shader: ComputeShader,
    buffers: Option<MaxwellEqsBuffers>,
}

impl MaxwellEqsCompute {
    pub const ENTRY_POINT: Option<&'static str> = Some("e_field_compute");
    /// Creates an uninitialized [`MaxwellEqsCompute`] shader.
    pub fn new(device: &Device) -> Result<Self> {
        let wgpu::ShaderSource::SpirV(spirv) = &super::SHADER.source else {
            unreachable!()
        };
        let shader = ComputeShader::new(device, wgpu::ShaderModuleDescriptor {
            label: Self::ENTRY_POINT,
            source: wgpu::ShaderSource::SpirV(Cow::Borrowed(spirv))
        });
        Ok(Self {
            shader,
            buffers: None,
        })
    }

    /// Initializes the shader for running. Call this after setting the inputs to the shader
    /// (`grid_data`, `grid_info`, and `point_charges`).
    ///
    /// You can call this whenever you want, but preferably you call it only once just before running
    /// the shader for the first time. TODO: Use the write functions to edit the buffers
    pub fn initialize(&mut self, device: &Device, data: &MaxwellEqsData, grid_info: &GridInfo) -> Result<()> {
        let cells_buf = ComputeBuffer::create_init(
            device,
            bytemuck::cast_slice(data.cells.as_slice()),
            wgpu::BufferUsages::STORAGE,
            false,
            true
        );
        let pt_charges_buf = ComputeBuffer::create_init(
            device,
            bytemuck::cast_slice(data.point_charges.as_slice()),
            wgpu::BufferUsages::STORAGE,
            false,
            true
        );
        let grid_info_buf = ComputeBuffer::create_init(
            device,
            bytemuck::bytes_of(grid_info),
            wgpu::BufferUsages::UNIFORM,
            true,
            true
        );
        self.shader
            .bind_buffer_sequential(device, &cells_buf)
            .bind_buffer_sequential(device, &pt_charges_buf)
            .bind_buffer_sequential(device, &grid_info_buf);
        self.buffers = Some(MaxwellEqsBuffers {
            cells_buf,
            pt_charges_buf,
            grid_info_buf,
        });
        self.shader.initialize(device, Self::ENTRY_POINT, true)?;
        Ok(())
    }

    pub fn run(&mut self, device: &Device, queue: &Queue, grid_info: &GridInfo) -> Result<SubmissionIndex> {
        let workgroup_count = grid_info.grid_dimensions.map(|v| v.div_ceil(4))
            .to_array();
        self.shader.run_shader(device, queue, Self::ENTRY_POINT, workgroup_count, true)
    }

    pub fn read_buffers(&self) -> Result<MaxwellEqsData> {
        let Some(buffers) = &self.buffers else {
            return Err(Error::ShaderIsUninitialized)
        };

        let cells_view = buffers.cells_buf.read_slice()?;
        let pt_charges_view = buffers.pt_charges_buf.read_slice()?;
        let cells: Vec<GridCell> = bytemuck::cast_slice(&cells_view).to_vec();
        let point_charges: Vec<PointCharge> = bytemuck::cast_slice(&pt_charges_view).to_vec();
        Ok(MaxwellEqsData {
            cells,
            point_charges,
        })
    }
}

pub struct MaxwellEqsData {
    pub cells: Vec<GridCell>,
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
            point_charges,
        })
    }
}

pub struct MaxwellEqsBuffers {
    pub cells_buf: ComputeBuffer,
    pub pt_charges_buf: ComputeBuffer,
    pub grid_info_buf: ComputeBuffer,
}