pub mod shader;
pub mod error;
mod prelude;
mod util;

use std::ops::Mul;
use crate::shader::fdtd::{FdtdData, GaussianPulse, ELECTRON_MASS, ELEMENTARY_CHARGE, PROTON_MASS};
use crate::util::{arrow_polyline, bb_polyline, flat_idx_to_vector};
use glam::{UVec3, Vec3};
use include_dir::{include_dir, Dir};
use khal::backend::{Backend, Encoder, GpuBackend, GpuBackendError, WebGpu};
use khal::Shader;
use kiss3d::prelude::*;
use rand::{RngExt, SeedableRng};
use shader_crate::{FdtdDirichlet, GridCell, GridInfo, PointCharge};

static SPIRV_DIR: Dir<'static> = include_dir!("$CARGO_MANIFEST_DIR/shaders-spirv");

#[kiss3d::main]
async fn main() {
    let webgpu = WebGpu::default().await.unwrap();
    let backend = GpuBackend::WebGpu(webgpu);

    // Generating input data for the shader
    let mut grid_info = GridInfo::new(Vec3::ZERO, UVec3::splat(8), Vec3::splat(1.));
    let pulse = GaussianPulse::from_max_frequency(1e6, 1., Vec3::ONE, &grid_info);
        grid_info.adjust_dt_from_gaussian_pulse(pulse.half_duration, 10);
    let mut rng = rand::rngs::StdRng::seed_from_u64(1234567);
    let mut input_data = FdtdData::new(grid_info).unwrap();
        input_data.sources.push(
            pulse.construct_source(100, grid_info.dt, grid_info.dt*5.)
        );

    let output_data = compute_h_field(&backend, &input_data).await
        .unwrap();
    render_result(output_data, &input_data).await;
}

#[derive(Shader)]
pub struct GpuKernels {
    pub h_field_compute: FdtdDirichlet
}

pub async fn compute_h_field(
    backend: &GpuBackend,
    input_data: &FdtdData,
) -> Result<FdtdOutput, GpuBackendError> {
    let mut buffers = shader::fdtd::create_buffers(backend, input_data)?;

    let kernels = GpuKernels::from_backend(&backend)?;
    let mut encoder = backend.begin_encoding();
    let mut pass = encoder.begin_pass("e_field_compute", None);
    shader::fdtd::encode_h_field_compute(
        &kernels.h_field_compute,
        &mut pass,
        &mut buffers,
        &input_data.grid_info
    )?;
    drop(pass);
    backend.submit(encoder)?;

    let cells = backend.slow_read_vec(&buffers.cells).await?;
    let point_charges = backend.slow_read_vec(&buffers.point_charges).await?;

    Ok(
        FdtdOutput {
            cells,
            point_charges,
        }
    )
}

pub struct FdtdOutput {
    pub cells: Vec<GridCell>,
    pub point_charges: Vec<PointCharge>,
}

pub async fn render_result(output: FdtdOutput, input: &FdtdData) {
    let grid_info = &input.grid_info;
    let mut window = Window::new("Compute Shader Testing").await;
    let mut camera = OrbitCamera3d::default();
    let mut scene = SceneNode3d::empty();
    scene.add_light(Light::point(100.))
        .set_position(Vec3::new(10., 10., 10.));
    // Draw data
    let bb_extents = grid_info.grid_dimensions.as_vec3() * grid_info.cell_size;
    let bb_poly_line = bb_polyline(bb_extents, grid_info.position);
    let arrows = output.cells.iter()
        .enumerate()
        .map(|(i, cell)| {
            let i = i as u32;
            let v = cell.e.normalize() * grid_info.cell_size / 2.;
            let cell_position = flat_idx_to_vector(i, grid_info.grid_dimensions)
                .as_vec3() * grid_info.cell_size;
            arrow_polyline(cell_position, v)
        }).collect::<Vec<_>>();

    for pt in output.point_charges.iter().map(|v| v.position) {
        scene.add_sphere(0.2)
            .set_position(pt)
            .set_color(BLUE);
    }
    for src in input.sources.iter() {
        let pos = flat_idx_to_vector(src.cell_idx, grid_info.grid_dimensions).as_vec3()
            .mul(grid_info.cell_size);
        scene.add_sphere(0.1)
            .set_position(pos)
            .set_color(RED);
    }
    // Render the window
    while window.render_3d(&mut scene, &mut camera).await {
        window.draw_polyline(&bb_poly_line);
        arrows.iter().for_each(|a| window.draw_polyline(a))
    }
}