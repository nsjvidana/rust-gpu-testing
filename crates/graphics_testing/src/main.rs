pub mod shader;
pub mod error;
mod prelude;
mod util;

use crate::shader::fdtd::{FDTDData, ELECTRON_MASS, ELEMENTARY_CHARGE, PROTON_MASS};
use crate::util::{arrow_polyline, bb_polyline, flat_idx_to_vector};
use glam::{UVec3, Vec3};
use include_dir::{include_dir, Dir};
use khal::backend::{Backend, Encoder, GpuBackend, GpuBackendError, WebGpu};
use khal::Shader;
use kiss3d::prelude::*;
use rand::{RngExt, SeedableRng};
use shader_crate::{HFieldCompute, GridInfo, PointCharge};

static SPIRV_DIR: Dir<'static> = include_dir!("$CARGO_MANIFEST_DIR/shaders-spirv");

#[kiss3d::main]
async fn main() {
    let webgpu = WebGpu::default().await.unwrap();
    let backend = GpuBackend::WebGpu(webgpu);

    // Generating input data for the shader
    let grid_info = GridInfo::new(Vec3::ZERO, UVec3::splat(8), 1., GridInfo::DEFAULT_DT);
    let mut rng = rand::rngs::StdRng::seed_from_u64(1234567);
    let input_data = FDTDData::new(grid_info).unwrap();
    let output_data = compute_h_field(&backend, &input_data).await
        .unwrap();
    render_result(output_data).await;
}

#[derive(Shader)]
pub struct GpuKernels {
    pub h_field_compute: HFieldCompute
}

pub async fn compute_h_field(
    backend: &GpuBackend,
    input_data: &FDTDData,
) -> Result<FDTDData, GpuBackendError> {
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
        FDTDData {
            cells,
            material_constants: input_data.material_constants.clone(),
            point_charges,
            grid_info: input_data.grid_info,
        }
    )
}

pub async fn render_result(output_data: FDTDData) {
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