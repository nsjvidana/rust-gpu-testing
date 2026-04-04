pub mod shader;
pub mod error;
mod prelude;

use glam::{UVec3, Vec3};
use rand::{RngExt, SeedableRng};
use shader_crate::{GridInfo, PointCharge};
use crate::shader::double_me::DoubleMe;
use crate::shader::maxwell_eqs::{MaxwellEqsCompute, MaxwellEqsData, ELECTRON_MASS, ELEMENTARY_CHARGE, PROTON_MASS};

fn main() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());

    // Create wgpu objects
    let (_adapter, device, queue) = pollster::block_on(get_adapter_device_queue(&instance));

    let grid_info = GridInfo::new(Vec3::ZERO, UVec3::splat(8), 1., );
    let mut rng = rand::rngs::StdRng::seed_from_u64(1234567);
    let point_charges = std::iter::repeat_with(|| {
        let charge_bool = rng.random_bool(0.5);
        let particle_mass = if charge_bool { PROTON_MASS } else { ELECTRON_MASS };
        let particle_count = rng.random_range(1..5) as f32;
        let charge_sign = if charge_bool { 1. } else { -1. };
        PointCharge::new(
            ELEMENTARY_CHARGE * particle_count * charge_sign,
            Vec3::new(
                rng.random_range(0.1..7.),
                rng.random_range(0.1..7.),
                rng.random_range(0.1..7.)
            ),
            particle_mass * particle_count,
        )
    }).take(5).collect::<Vec<_>>();
    let input_data = MaxwellEqsData::new(grid_info, point_charges).unwrap();
    let mut shader = MaxwellEqsCompute::new(&device).unwrap();
    shader.initialize(
        &device,
        &input_data,
        &grid_info
    ).unwrap();
    let submission = shader.run(&device, &queue, &grid_info).unwrap();
    shader::wait(&device, submission).unwrap();

    let output_data = shader.read_buffers().unwrap();

    println!("E field before: {:?}", input_data.e_field);
    println!("E field after: {:?}", output_data.e_field);
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