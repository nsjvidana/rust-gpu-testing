mod maxwell_eqs;

use wgpu::{BindingType, Buffer, BufferAddress, BufferBindingType, BufferDescriptor, BufferUsages, BufferView, CommandEncoder, CommandEncoderDescriptor, Device, Limits, Queue, ShaderModule, ShaderModuleDescriptor, ShaderStages, SubmissionIndex};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use crate::prelude::*;

/// A bind group for buffer bindings
pub type ComputeBindGroup = Vec<Option<ComputeBuffer>>;

pub const SHADER: wgpu::ShaderModuleDescriptor = wgpu::include_spirv!(env!("shader_crate.spv"));

pub struct ComputeShader {
    pub name: String,
    pub module: ShaderModule,
    /// Bind groups for buffer bindings
    pub bind_groups: Vec<Option<ComputeBindGroup>>,
    /// Underlying wgpu data
    inner: Option<ComputeShaderInner>
}

impl ComputeShader {
    pub fn new(device: &Device, descriptor: ShaderModuleDescriptor) -> Self {
        Self {
            name: descriptor.label.unwrap_or("Unnamed Shader").into(),
            module: device.create_shader_module(descriptor),
            bind_groups: Vec::with_capacity(1),
            inner: Default::default()
        }
    }

    pub fn bind_buffer_sequential(&mut self, device: &Device, compute_buffer: &ComputeBuffer) -> &mut Self {
        let max_bindings = device.limits().max_bindings_per_bind_group as usize;

        let bind_group_id = self.bind_groups.iter_mut()
            .position(|bind_group| bind_group.as_ref().is_some_and(|v| v.len() < max_bindings))
            .unwrap_or_else(|| {
                // create new bind group if we cannot find one with enough space
                self.bind_groups.push(Some(ComputeBindGroup::new()));
                self.bind_groups.len() - 1
            });

        self.bind_groups[bind_group_id]
            .as_mut()
            .unwrap()
            .push(Some(compute_buffer.clone()));

        self
    }

    pub fn bind_buffer_at(
        &mut self,
        device: &Device,
        compute_buffer: &ComputeBuffer,
        bind_group_id: u32,
        binding: u32
    ) -> Result<()> {
        let Limits {
            max_bind_groups,
            max_bindings_per_bind_group: max_bindings,
            ..
        } = device.limits();
        if bind_group_id >= max_bind_groups ||
            binding >= max_bindings
        {
            return Err(Error::InvalidBinding {
                bind_group_id,
                binding,
                max_bindings,
                max_bind_groups,
            })
        }

        let bind_group_id = bind_group_id as usize;
        let binding = binding as usize;
        if self.bind_groups.len() <= bind_group_id {
            self.bind_groups.resize(bind_group_id + 1, None);
            self.bind_groups[bind_group_id] = Some(ComputeBindGroup::new());
        }
        if self.bind_groups[bind_group_id].is_none() {
            self.bind_groups[bind_group_id] = Some(ComputeBindGroup::new());
        }
        let bg = self.bind_groups.get_mut(bind_group_id).unwrap()
            .as_mut().unwrap();
        if bg.len() <= binding {
            bg.resize(binding + 1, None);
        }
        bg[binding] = Some(compute_buffer.clone());

        Ok(())
    }

    /// Submits the shader to the queue for running.
    pub fn run_shader(
        &mut self,
        device: &Device,
        queue: &Queue,
        entry_point: Option<&str>,
        workgroup_count: [u32; 3],
        download_buffers: bool
    ) -> Result<SubmissionIndex> {
        let mut encoder =
            device.create_command_encoder(&CommandEncoderDescriptor { label: None });
        self.encode_run_commands(
            device,
            &mut encoder,
            entry_point,
            workgroup_count,
            download_buffers
        )?;
        let cmd_buf = encoder.finish();
        Ok(queue.submit([cmd_buf]))
    }

    /// Encode commands in a [`CommandEncoder`] with commands for running the shader.
    ///
    /// Optionally include commands for downloading and mapping all downloadable buffers for reading
    /// by setting `download_buffers` to `true`.
    pub fn encode_run_commands(
        &mut self,
        device: &Device,
        encoder: &mut CommandEncoder,
        entry_point: Option<&str>,
        workgroup_count: [u32; 3],
        download_buffers: bool
    ) -> Result<()> {
        if !self.is_initialized() {
            self.generate_pipeline(device, entry_point, false)?;
        }
        let inner = self.inner.as_ref().unwrap();

        let mut compute_pass =
            encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        compute_pass.set_pipeline(inner.pipeline.as_ref().unwrap());
        for (bg_id, bg) in inner.bind_groups.iter().enumerate() {
            compute_pass.set_bind_group(bg_id as _, bg.as_ref(), &[]);
        }
        compute_pass.dispatch_workgroups(workgroup_count[0], workgroup_count[1], workgroup_count[2]);
        drop(compute_pass);

        // download bufs
        if download_buffers {
            for bg in self.bind_groups.iter() {
                let Some(bg) = bg else { continue; };
                for buf in bg.iter() {
                    let Some(buf) = buf else { continue; };
                    buf.encode_download_command(encoder)?;
                    buf.encode_map_read(encoder)?;
                }
            }
        }

        Ok(())
    }

    pub fn is_initialized(&self) -> bool {
        self.inner.as_ref().is_some_and(|v| v.pipeline.is_some())
    }

    /// Initializes the shader.
    pub fn initialize(&mut self, device: &Device, entry_point: Option<&str>, force: bool) -> Result<()> {
        self.generate_pipeline(device, entry_point, force)
    }

    /// Generates the underlying [`wgpu::BindGroup`]s, [`wgpu::ComputePipeline`], and their layouts.
    ///
    /// The `force` parameter recreates and overwrites ALL underlying [`wgpu::BindGroup`]s, [`wgpu::ComputePipeline`],
    ///     and their layouts ([`wgpu::Buffer`] are not destroyed).
    pub fn generate_pipeline(&mut self, device: &Device, entry_point: Option<&str>, force: bool) -> Result<()> {
        self.generate_bind_groups(device, force)?;
        let inner = self.inner.as_mut().unwrap();

        let bg_layouts_refs = inner.bind_group_layouts.iter()
            .map(|v| v.as_ref())
            .collect::<Vec<_>>();
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: bg_layouts_refs.as_slice(),
            immediate_size: 0
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            module: &self.module,
            entry_point: entry_point,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None
        });

        inner.pipeline_layout = Some(pipeline_layout);
        inner.pipeline = Some(pipeline);
        Ok(())
    }

    /// Generates the underlying [`wgpu::BindGroup`]s and their layouts.
    ///
    /// The `force` parameter deletes ALL underlying [`wgpu::BindGroup`]s, [`wgpu::ComputePipeline`], and their layouts,
    ///     and recreates [`wgpu::BindGroup`]s and their layouts ([`wgpu::Buffer`]s are not destroyed).
    pub fn generate_bind_groups(&mut self, device: &Device, force: bool) -> Result<()> {
        if force {
            self.inner = None;
        }

        // Return if bind groups are already initialized.
        if self.inner.as_ref().is_some_and(|v| v.bind_groups.len() != 0) {
            return Ok(());
        }

        let bg_layouts = self.bind_groups.iter()
            .map(|bg| {
                let Some(bg) = bg else {
                    return None
                };
                let entries = bg.iter()
                    .enumerate()
                    .filter(|(_, b)| b.is_some())
                    .map(|(binding, buf)| wgpu::BindGroupLayoutEntry {
                        binding: binding as _,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: buf.as_ref().unwrap().binding_type,
                            has_dynamic_offset: false,
                            min_binding_size: None
                        },
                        count: None,
                    })
                    .collect::<Vec<_>>();
                Some(device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: None,
                    entries: entries.as_slice(),
                }))
            })
            .collect::<Vec<_>>();
        let bind_groups = bg_layouts.iter()
            .zip(self.bind_groups.iter())
            .map(|(layout, bg)| {
                if layout.is_none() {
                    return None
                }

                let bg = bg.as_ref().unwrap();
                let entries = bg.iter()
                    .enumerate()
                    .filter(|(_, b)| b.is_some())
                    .map(|(binding, buf)| {
                        wgpu::BindGroupEntry {
                            binding: binding as u32,
                            resource: buf.as_ref().unwrap().buf.as_entire_binding(),
                        }
                    })
                    .collect::<Vec<_>>();
                Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: layout.as_ref().unwrap(),
                    entries: entries.as_slice(),
                }))
            })
            .collect::<Vec<_>>();

        self.inner = Some(ComputeShaderInner {
            bind_group_layouts: bg_layouts,
            bind_groups,
            ..Default::default()
        });
        Ok(())
    }
}

#[derive(Default)]
struct ComputeShaderInner {
    pub bind_group_layouts: Vec<Option<wgpu::BindGroupLayout>>,
    pub bind_groups: Vec<Option<wgpu::BindGroup>>,
    pub pipeline_layout: Option<wgpu::PipelineLayout>,
    pub pipeline: Option<wgpu::ComputePipeline>,
}

/// A buffer used by a [`ComputeShader`]
#[derive(Clone, PartialEq)]
pub struct ComputeBuffer {
    pub buf: Buffer,
    pub download_buf: Option<Buffer>,
    pub binding_type: BufferBindingType,
}

impl ComputeBuffer {
    pub fn create(
        device: &Device,
        size: BufferAddress,
        mut usage: BufferUsages,
        read_only: bool,
        with_download: bool
    ) -> Self {
        if with_download { usage |= BufferUsages::COPY_SRC }
        let buf = device.create_buffer(&BufferDescriptor {
            label: None,
            size,
            usage,
            mapped_at_creation: false
        });
        let download_buf = with_download.then(|| {
            device.create_buffer(&BufferDescriptor {
                label: None,
                size,
                usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
                mapped_at_creation: false
            })
        });
        let binding_type =
            if usage.contains(BufferUsages::STORAGE) { BufferBindingType::Storage { read_only } }
            else { BufferBindingType::Uniform };

        Self { buf, download_buf, binding_type }
    }

    pub fn create_init(
        device: &Device,
        data: &[u8],
        mut usage: BufferUsages,
        read_only: bool,
        with_download: bool
    ) -> Self {
        if with_download { usage |= BufferUsages::COPY_SRC }
        let buf = device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: data,
            usage,
        });
        let download_buf = with_download.then( || {
            device.create_buffer(&BufferDescriptor {
                label: None,
                size: buf.size(),
                usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
                mapped_at_creation: false
            })
        });
        let binding_type =
            if usage.contains(BufferUsages::STORAGE) { BufferBindingType::Storage { read_only } }
            else { BufferBindingType::Uniform };

        Self { buf, download_buf, binding_type }
    }

    pub fn encode_download_command(&self, encoder: &mut CommandEncoder) -> Result<()> {
        let download_buf = self.get_download_buf()?;
        encoder.copy_buffer_to_buffer(
            &self.buf,
            0,
            download_buf,
            0,
            self.buf.size()
        );

        Ok(())
    }

    /// Encodes a [`CommandEncoder::map_buffer_on_submit`] command spanning the entirety of
    /// this buffer's download buffer
    pub fn encode_map_read(&self, encoder: &mut CommandEncoder) -> Result<()> {
        let download_buf = self.get_download_buf()?;
        encoder.map_buffer_on_submit(
            download_buf,
            wgpu::MapMode::Read,
            ..,
            |_| {}
        );

        Ok(())
    }

    pub fn read_slice(&self) -> Result<BufferView> {
        let download_buf = self.get_download_buf()?;
        Ok(download_buf.get_mapped_range(..))
    }

    pub fn get_download_buf(&self) -> Result<&Buffer> {
        self.download_buf.as_ref().ok_or(Error::BufferCannotBeDownloaded)
    }

    pub fn destroy(mut self) {
        self.buf.destroy();
        if let Some(b) = self.download_buf.take() {
            b.destroy();
        }
    }
}