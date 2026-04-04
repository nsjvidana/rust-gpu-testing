use std::borrow::Cow;
use wgpu::{BufferUsages, SubmissionIndex};
use crate::prelude::*;
use crate::shader::{ComputeBuffer, ComputeShader};

pub struct DoubleMe {
    pub shader: ComputeShader,
    float_values_buf: Option<ComputeBuffer>
}

impl DoubleMe {
    const ENTRY_POINT: Option<&'static str> = Some("double_me");

    pub fn new(device: &wgpu::Device) -> Self {
        let wgpu::ShaderSource::SpirV(spirv) = &super::SHADER.source else {
            unreachable!()
        };
        let shader = ComputeShader::new(device, wgpu::ShaderModuleDescriptor {
            label: Self::ENTRY_POINT,
            source: wgpu::ShaderSource::SpirV(Cow::Borrowed(spirv))
        });

        Self {
            shader,
            float_values_buf: None
        }
    }

    pub fn initialize(&mut self, device: &wgpu::Device, float_values: &[f32], force: bool) -> Result<()> {
        if force && let Some(b) = self.float_values_buf.take() {
            b.destroy();
        } else if self.float_values_buf.is_some() {
            return Ok(());
        }

        let buf = ComputeBuffer::create_init(
            &device,
            bytemuck::cast_slice(&float_values),
            BufferUsages::STORAGE,
            false,
            true
        );

        self.shader.bind_buffer_sequential(device, &buf)
            .initialize(device, Self::ENTRY_POINT, force)?;
        self.float_values_buf = Some(buf);
        Ok(())
    }

    pub fn run(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, float_values_count: u32) -> Result<SubmissionIndex> {
        let workgroup_count = float_values_count.div_ceil(64);
        self.shader.run_shader(device, queue, Self::ENTRY_POINT, [workgroup_count, 1, 1], true)
    }

    pub fn read_buf(&self) -> Result<Vec<f32>> {
        if let Some(buf) = self.float_values_buf.as_ref() {
            let view = buf.read_slice()?;
            let output_floats: &[f32] = bytemuck::cast_slice(&view);
            return Ok(output_floats.to_vec())
        }
        Err(Error::ShaderIsUninitialized)
    }
}