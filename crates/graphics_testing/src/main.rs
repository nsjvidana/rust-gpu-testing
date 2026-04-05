pub mod shader;
pub mod error;
mod prelude;
mod util;

use glam::{USizeVec3, UVec3, Vec3};
use kiss3d::prelude::*;
use rand::{RngExt, SeedableRng};
use shader_crate::{GridInfo, PointCharge};
use crate::shader::maxwell_eqs::{MaxwellEqsCompute, MaxwellEqsData, ELECTRON_MASS, ELEMENTARY_CHARGE, PROTON_MASS};
use crate::util::{arrow_polyline, bb_polyline, flat_idx_to_vector};

#[kiss3d::main]
async fn main() {
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
        PointCharge {
            q: ELEMENTARY_CHARGE * particle_count * charge_sign,
            position: Vec3::new(
                rng.random_range(0.1..7.),
                rng.random_range(0.1..7.),
                rng.random_range(0.1..7.)
            ),
            mass: particle_mass * particle_count,
            ..Default::default()
        }
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

    let mut window = Window::new("Compute Shader Testing").await;
    let mut camera = OrbitCamera3d::default();
    let mut scene = SceneNode3d::empty();
    scene.add_light(Light::point(100.))
        .set_position(Vec3::new(10., 10., 10.));
    // Draw data
    let bb_extents = grid_info.grid_dimensions.as_vec3() * grid_info.cell_size;
    let bb_poly_line = bb_polyline(bb_extents, grid_info.position);
    let arrows = output_data.e_field.iter()
        .enumerate()
        .map(|(i, v)| {
            let i = i as u32;
            let v = Vec3::new(v.x, v.y, v.z).normalize() * grid_info.cell_size / 2.;
            let cell_position = flat_idx_to_vector(i, grid_info.grid_dimensions)
                .as_vec3() * grid_info.cell_size;
            arrow_polyline(cell_position, v)
        }).collect::<Vec<_>>();
    for pt in output_data.point_charges.iter().map(|v| v.position) {
        scene.add_sphere(0.2)
            .set_position(pt)
            .set_color(RED);
    }
    // Render the window
    while window.render_3d(&mut scene, &mut camera).await {
        window.draw_polyline(&bb_poly_line);
        arrows.iter().for_each(|a| window.draw_polyline(a))
    }
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