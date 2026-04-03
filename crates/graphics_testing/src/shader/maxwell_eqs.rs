use wgpu::Device;
use crate::shader::ComputeShader;

const SHADER: wgpu::ShaderModuleDescriptor = wgpu::include_spirv!(env!("shader_crate.spv"));

pub struct EFieldCompute {
    shader: ComputeShader,
}

impl EFieldCompute {
    pub fn new(device: &Device) -> Self {
        Self {
            shader: ComputeShader::new(device, SHADER),
        }
    }
}

pub struct Grid {

}