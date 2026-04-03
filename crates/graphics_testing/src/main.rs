pub mod shader;
pub mod error;
mod prelude;

use wgpu::{BufferUsages};
use wgpu::util::DeviceExt;

fn main() {
    const SHADER: wgpu::ShaderModuleDescriptor = wgpu::include_spirv!(env!("shader_crate.spv"));

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());

    // Create wgpu objects
    let (adapter, device, queue) = pollster::block_on(get_adapter_device_queue(&instance));

    let float_values = std::iter::repeat_with(|| rand::random_range(-100f32..100.0))
        .take(64)
        .collect::<Vec<f32>>();

    let buf = shader::ComputeBuffer::create_init(
        &device,
        bytemuck::cast_slice(&float_values),
        BufferUsages::STORAGE,
        false,
        true
    );
    let mut shader = shader::ComputeShader::new(&device, SHADER);
    shader
        .bind_buffer_sequential(&device, &buf);

    let workgroup_count = float_values.len().div_ceil(64) as u32;
    let submission_idx = shader.run_shader(
        &device,
        &queue,
        Some("double_me"),
        [workgroup_count, 1, 1],
        true
    ).unwrap();
    device.poll(wgpu::PollType::Wait {
        submission_index: Some(submission_idx),
        timeout: None
    }).unwrap();

    let output_slice = buf.read_slice().unwrap();
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