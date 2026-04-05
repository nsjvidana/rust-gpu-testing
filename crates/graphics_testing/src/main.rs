pub mod shader;
pub mod error;
mod prelude;

use glam::{USizeVec3, UVec3, Vec3};
use kiss3d::prelude::*;
use rand::{RngExt, SeedableRng};
use shader_crate::{GridInfo, PointCharge};
use crate::shader::maxwell_eqs::{MaxwellEqsCompute, MaxwellEqsData, ELECTRON_MASS, ELEMENTARY_CHARGE, PROTON_MASS};

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

    let mut window = Window::new("Compute Shader Testing").await;
    let mut camera = OrbitCamera3d::default();
    let mut scene = SceneNode3d::empty();
    scene.add_light(Light::point(100.))
        .set_position(Vec3::new(10., 10., 10.));
    // Draw data
    let grid_dim = grid_info.grid_dimensions;
    let bb_poly_line = {
        let bb_extents = grid_dim.as_vec3() * grid_info.cell_size;
        let grid_pos = grid_info.position;
        let v0 = grid_pos;
        let v0x = grid_pos + Vec3::new(bb_extents.x, 0., 0.);
        let v0xy = grid_pos + Vec3::new(bb_extents.x, bb_extents.y, 0.);
        let v0y = grid_pos + Vec3::new(0., bb_extents.y, 0.);
        let v1 = v0 + Vec3::new(0., 0., bb_extents.z);
        let v1x = v0x + Vec3::new(0., 0., bb_extents.z);
        let v1xy = v0xy + Vec3::new(0., 0., bb_extents.z);
        let v1y = v0y + Vec3::new(0., 0., bb_extents.z);
        Polyline3d::new(vec![
            v0, v0x, v0xy, v0y, v0,
            v1, v1x, v1xy, v1y, v1,
            v1x, v0x, v0xy, v1xy,
            v1y, v0y
        ]).with_color(WHITE)
    };
    let arrows = output_data.e_field.iter()
        .enumerate()
        .map(|(i, v)| {
            let i = i as u32;
            let v = Vec3::new(v.x, v.y, v.z).normalize() * grid_info.cell_size / 2.;
            let cell_position = UVec3::new(
                i % grid_dim.x,
                (i / grid_dim.x) % grid_dim.y,
                i / (grid_dim.x * grid_dim.y),
            ).as_vec3() * grid_info.cell_size;

            let u = -v;
            let mut axis = u.cross(Vec3::X).normalize_or_zero();
                if axis.length_squared() == 0. {
                    axis = u.cross(Vec3::Y).normalize();
                }
            let arrow_head = u.rotate_axis(axis, -std::f32::consts::FRAC_PI_4)
                .normalize() * grid_info.cell_size / 6.;

            Polyline3d::new(vec![
                cell_position,
                cell_position + v,
                cell_position + v + arrow_head
            ])
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