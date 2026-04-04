pub mod shader;
pub mod error;
mod prelude;

use crate::shader::double_me::DoubleMe;

fn main() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());

    // Create wgpu objects
    let (_adapter, device, queue) = pollster::block_on(get_adapter_device_queue(&instance));

    let float_values = std::iter::repeat_with(|| rand::random_range(-100f32..100.0))
        .take(64)
        .collect::<Vec<f32>>();

    let mut double_me = DoubleMe::new(&device);
    double_me.initialize(&device, float_values.as_slice(), false).unwrap();
    device.poll(wgpu::PollType::Wait {
        submission_index: Some(
            double_me.run(&device, &queue, float_values.len() as _).unwrap()
        ),
        timeout: None
    }).unwrap();
    let output = double_me.read_buf().unwrap();

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