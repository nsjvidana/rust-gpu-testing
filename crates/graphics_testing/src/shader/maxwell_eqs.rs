use std::borrow::Cow;
use glam::{UVec4, Vec4};
use wgpu::Device;
use crate::shader::ComputeShader;

const SHADER: wgpu::ShaderModuleDescriptor = wgpu::include_spirv!(env!("shader_crate.spv"));

pub struct EFieldCompute {
    shader: ComputeShader,
}

impl EFieldCompute {
    pub fn new(device: &Device) -> Self {
        let wgpu::ShaderSource::SpirV(spirv) = &SHADER.source else {
            unreachable!()
        };
        Self {
            shader: ComputeShader::new(device, wgpu::ShaderModuleDescriptor {
                label: Some("e_field_compute"),
                source: wgpu::ShaderSource::SpirV(Cow::Borrowed(spirv))
            }),
        }
    }
}

pub struct GridData {
    pub e_field: Vec<Vec4>,
    pub b_field: Vec<Vec4>,
}