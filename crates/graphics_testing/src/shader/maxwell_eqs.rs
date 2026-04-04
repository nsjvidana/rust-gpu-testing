use crate::prelude::*;
use crate::shader::{ComputeBuffer, ComputeShader};
use glam::Vec4;
use shader_crate::GridInfo;
use std::borrow::Cow;
use wgpu::Device;

pub struct MaxwellEqsCompute {
    /// The underlying compute shader.
    shader: ComputeShader,
    buffers: Option<MaxwellEqsBuffers>,
}

impl MaxwellEqsCompute {

    /// Creates an uninitialized [`MaxwellEqsCompute`] shader.
    pub fn new(device: &Device) -> Result<Self> {
        let wgpu::ShaderSource::SpirV(spirv) = &super::SHADER.source else {
            unreachable!()
        };
        let shader = ComputeShader::new(device, wgpu::ShaderModuleDescriptor {
            label: Some("e_field_compute"),
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
    pub fn initialize(&mut self, device: &Device, data: &MaxwellEqsData) -> Result<()> {
        let e_field_buf = ComputeBuffer::create_init(
            device,
            bytemuck::cast_slice(data.e_field.as_slice()),
            wgpu::BufferUsages::STORAGE,
            false,
            true
        );
        // TODO: let _b_field_buf = ;
        let pt_charges_buf = ComputeBuffer::create_init(
            device,
            bytemuck::cast_slice(data.point_charges.as_slice()),
            wgpu::BufferUsages::STORAGE,
            false,
            true
        );
        let grid_info_buf = ComputeBuffer::create_init(
            device,
            bytemuck::bytes_of(&data.grid_info),
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

pub struct MaxwellEqsData {
    pub e_field: Vec<Vec4>,
    // TODO: pub b_field: Vec<Vec4>,
    pub point_charges: Vec<Vec4>,
    pub grid_info: GridInfo,
}

pub struct MaxwellEqsBuffers {
    pub e_field_buf: ComputeBuffer,
    // TODO: pub b_field_buf: ComputeBuffer,
    pub pt_charges_buf: ComputeBuffer,
}