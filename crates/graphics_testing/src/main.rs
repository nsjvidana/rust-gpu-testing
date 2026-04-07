pub mod shader;
pub mod error;
mod prelude;
mod util;

use glam::{USizeVec3, UVec3, Vec3};
use include_dir::{include_dir, Dir};
use khal::backend::{Backend, DispatchGrid, Encoder, GpuBackend, GpuBackendError, WebGpu};
use khal::{BufferUsages, Shader};
use kiss3d::prelude::*;
use rand::{RngExt, SeedableRng};
use shader_crate::{EFieldCompute, GridInfo, PointCharge};
use crate::shader::maxwell_eqs::{MaxwellEqsBuffers, MaxwellEqsData, ELECTRON_MASS, ELEMENTARY_CHARGE, PROTON_MASS};
use crate::util::{arrow_polyline, bb_polyline, flat_idx_to_vector};

static SPIRV_DIR: Dir<'static> = include_dir!("$CARGO_MANIFEST_DIR/shaders-spirv");

#[kiss3d::main]
async fn main() {
    let webgpu = WebGpu::default().await.unwrap();
    let backend = GpuBackend::WebGpu(webgpu);

    // Generating input data for the shader
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

    let output_data = compute_e_field(&backend, &input_data).await
        .unwrap();
    render_result(output_data).await;
}

#[derive(Shader)]
pub struct GpuKernels {
    pub e_field_compute: EFieldCompute
}

pub async fn compute_e_field(
    backend: &GpuBackend,
    input_data: &MaxwellEqsData,
) -> Result<MaxwellEqsData, GpuBackendError> {
    let mut buffers = shader::maxwell_eqs::create_buffers(backend, input_data)?;

    let kernels = GpuKernels::from_backend(&backend).unwrap();
    let mut encoder = backend.begin_encoding();
    let mut pass = encoder.begin_pass("e_field_compute", None);
    shader::maxwell_eqs::encode_e_field_compute(
        &kernels.e_field_compute,
        &mut pass,
        &mut buffers,
        &input_data.grid_info
    )?;
    drop(pass);
    backend.submit(encoder)?;

    let cells = backend.slow_read_vec(&buffers.cells).await?;
    let point_charges = backend.slow_read_vec(&buffers.point_charges).await?;

    Ok(
        MaxwellEqsData {
            cells,
            point_charges,
            grid_info: input_data.grid_info,
        }
    )
}

pub async fn render_result(output_data: MaxwellEqsData) {
    let grid_info = &output_data.grid_info;
    let mut window = Window::new("Compute Shader Testing").await;
    let mut camera = OrbitCamera3d::default();
    let mut scene = SceneNode3d::empty();
    scene.add_light(Light::point(100.))
        .set_position(Vec3::new(10., 10., 10.));
    // Draw data
    let bb_extents = grid_info.grid_dimensions.as_vec3() * grid_info.cell_size;
    let bb_poly_line = bb_polyline(bb_extents, grid_info.position);
    let arrows = output_data.cells.iter()
        .enumerate()
        .map(|(i, cell)| {
            let i = i as u32;
            let v = cell.e.normalize() * grid_info.cell_size / 2.;
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