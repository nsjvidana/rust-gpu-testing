use std::num::NonZeroU64;
use wgpu::util::DeviceExt;

fn main() {
    const SHADER: wgpu::ShaderModuleDescriptor = wgpu::include_spirv!(env!("shader_crate.spv"));

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());

    // Create wgpu objects
    let (adapter, device, queue) = pollster::block_on(get_adapter_device_queue(&instance));

    let float_values = std::iter::repeat_with(|| rand::random_range(-100f32..100.0))
        .take(64)
        .collect::<Vec<f32>>();
    let buffers = make_buffers(&device, bytemuck::cast_slice(float_values.as_slice()));

    let (shader_module, pipeline, bind_group) = make_compute_pipeline::<f32>(
        &device,
        SHADER,
        &buffers,
    );

    let workgroup_count = float_values.len().div_ceil(64);
    let output_slice = run_shader(
        &device,
        &queue,
        &pipeline,
        &bind_group,
        &buffers,
        workgroup_count as u32
    );
    let output: &[f32] = bytemuck::cast_slice(&output_slice);

    println!("before:{float_values:?} \nafter: {output:?}");
}

async fn get_adapter_device_queue(
    instance: &wgpu::Instance
) -> (wgpu::Adapter, wgpu::Device, wgpu::Queue) {
    let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .expect("Couldn't find a compatible graphics adapter on this computer");

    if !adapter
        .get_downlevel_capabilities()
        .flags
        .contains(wgpu::DownlevelFlags::COMPUTE_SHADERS) {
        panic!("Graphics adapter doesn't support compute shaders")
    }

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor::default())
        .await
        .expect("Couldn't find compatible graphics device on this computer");
    (adapter, device, queue)
}

fn make_buffers(device: &wgpu::Device, compute_data: &[u8]) -> ComputeBuffers {
    let input_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("compute data buf"),
        contents: compute_data,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let download_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("float values download buf"),
        size: input_buf.size(),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    ComputeBuffers {
        data_buf: input_buf,
        // output_buf,
        download_buf,
    }
}

fn make_compute_pipeline<Data>(
    device: &wgpu::Device,
    shader_src: wgpu::ShaderModuleDescriptor,
    buffers: &ComputeBuffers
) -> (wgpu::ShaderModule, wgpu::ComputePipeline, wgpu::BindGroup) {
    let module = device.create_shader_module(shader_src);

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility:  wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: Some(NonZeroU64::new(4).unwrap())
                },
                count: None
            },
        ]
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buffers.data_buf.as_entire_binding(),
            },
        ]
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[Some(&bind_group_layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some("double_me"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None
    });

    (module, pipeline, bind_group)
}

fn run_shader(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &wgpu::ComputePipeline,
    bind_group: &wgpu::BindGroup,
    buffers: &ComputeBuffers,
    workgroup_count: u32
) -> wgpu::BufferView {
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    let mut compute_pass =
        encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
    compute_pass.set_pipeline(&pipeline);
    compute_pass.set_bind_group(0, bind_group, &[]);
    compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
    drop(compute_pass);

    encoder.copy_buffer_to_buffer(
        &buffers.data_buf,
        0,
        &buffers.download_buf,
        0,
        buffers.data_buf.size(),
    );

    encoder.map_buffer_on_submit(
        &buffers.download_buf,
        wgpu::MapMode::Read,
        ..,
        |_| {}
    );

    let cmd_buf = encoder.finish();
    queue.submit([cmd_buf]);
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();

    let data = buffers.download_buf.get_mapped_range(..);
    data
}

struct ComputeBuffers {
    pub data_buf: wgpu::Buffer,
    pub download_buf: wgpu::Buffer,
}
