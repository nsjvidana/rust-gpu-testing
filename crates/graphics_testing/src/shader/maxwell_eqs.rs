use std::borrow::Cow;
use bytemuck::{Pod, Zeroable};
use glam::{UVec4, Vec3, Vec4};
use wgpu::Device;
use shader_crate::{GridInfo, PointCharge};
use crate::shader::{ComputeBuffer, ComputeShader};
use crate::prelude::*;

const SHADER: wgpu::ShaderModuleDescriptor = wgpu::include_spirv!(env!("shader_crate.spv"));

pub struct MaxwellEqsCompute {
    /// The underlying compute shader.
    shader: ComputeShader,
    pub point_charges: Vec<PointCharge>,
    pub grid_data: GridData,
    pub grid_info: GridInfo,
    buffers: Option<MaxwellEqsBuffers>,
}

impl MaxwellEqsCompute {

    /// Creates an uninitialized [`MaxwellEqsCompute`] shader.
    pub fn new(device: &Device, grid_info: GridInfo) -> Result<Self> {
        let wgpu::ShaderSource::SpirV(spirv) = &SHADER.source else {
            unreachable!()
        };
        let shader = ComputeShader::new(device, wgpu::ShaderModuleDescriptor {
            label: Some("e_field_compute"),
            source: wgpu::ShaderSource::SpirV(Cow::Borrowed(spirv))
        });

        let num_cells = grid_info.grid_dimensions.element_product() as usize;
        if num_cells == 0 { return Err(Error::BufferSizeZero) };
        let grid_data = GridData {
            e_field: vec![Vec4::ZERO; num_cells],
        };
        Ok(Self {
            shader,
            point_charges: Vec::new(),
            grid_data,
            grid_info,
            buffers: None,
        })
    }

    /// Initializes the shader for running. Call this after setting the inputs to the shader
    /// (`grid_data`, `grid_info`, and `point_charges`).
    pub fn initialize(&mut self, device: &Device) -> Result<()> {
        if self.shader.is_initialized() && self.buffers.is_some() {
            return Ok(());
        }
        let e_field_buf = ComputeBuffer::create_init(
            device,
            bytemuck::cast_slice(self.grid_data.e_field.as_slice()),
            wgpu::BufferUsages::STORAGE,
            false,
            true
        );
        // TODO: let _b_field_buf = ;
        let pt_charges_buf = ComputeBuffer::create_init(
            device,
            bytemuck::cast_slice(self.point_charges.as_slice()),
            wgpu::BufferUsages::STORAGE,
            false,
            true
        );
        let grid_info_buf = ComputeBuffer::create_init(
            device,
            bytemuck::bytes_of(&self.grid_info),
            wgpu::BufferUsages::UNIFORM,
            true,
            true
        );
        self.shader
            .bind_buffer_sequential(device, &e_field_buf)
            .bind_buffer_sequential(device, &pt_charges_buf)
            .bind_buffer_sequential(device, &grid_info_buf);
        self.shader.initialize(device, Some("e_field_compute"), true)?;
        Ok(())
    }
}

pub struct MaxwellEqsBuffers {
    pub e_field_buf: ComputeBuffer,
    // TODO: pub b_field_buf: ComputeBuffer,
    pub pt_charges_buf: ComputeBuffer,
}

#[derive(Default)]
pub struct GridData {
    /// Electric field
    pub e_field: Vec<Vec4>,
    // TODO: magnetic field
}