pub mod shader;
pub mod error;
mod prelude;
mod util;

use std::ops::Mul;
use crate::shader::fdtd::{FdtdData, GaussianPulse, MaxwellEqsBuffers, ELECTRON_MASS, ELEMENTARY_CHARGE, PROTON_MASS};
use crate::util::{arrow_polyline, bb_polyline, flat_idx_to_vector};
use glam::{UVec3, Vec3};
use include_dir::{include_dir, Dir};
use khal::backend::{Backend, Buffer, DispatchGrid, Encoder, GpuBackend, GpuBackendError, GpuBuffer, WebGpu};
use khal::{AsGpuSlice, BufferUsages, Shader};
use kiss3d::prelude::*;
use rand::{RngExt, SeedableRng};
use shader_crate::{vector_to_flat_idx, FdtdDirichlet, GridCell, GridInfo, PointCharge};

static SPIRV_DIR: Dir<'static> = include_dir!("$CARGO_MANIFEST_DIR/shaders-spirv");

#[kiss3d::main]
async fn main() {
    let webgpu = WebGpu::default().await.unwrap();
    let backend = GpuBackend::WebGpu(webgpu);

    // Generating input data for the shader
    let grid_lengths = Vec3::splat(8.);
    let cell_size = 0.5;
    let index_dims = (grid_lengths / cell_size).ceil().as_uvec3();
    let mut grid_info = GridInfo::new(Vec3::ZERO, index_dims, Vec3::splat(cell_size));
    let pulse = GaussianPulse::from_max_frequency(1e6, 1., Vec3::Z, &grid_info);
        grid_info.adjust_dt_from_gaussian_pulse(pulse.half_duration, 10);
    let mut rng = rand::rngs::StdRng::seed_from_u64(1234567);
    let mut input_data = FdtdData::new(grid_info).unwrap();
        input_data.sources[0] = pulse.construct_source(100, grid_info.dt, 0.);

    println!("{:?}", grid_info);
    println!("{}", input_data.material_constants[0].hn_update_coeff_inv);

    main_render_loop(&backend, &input_data).await.unwrap();
}

#[derive(Shader)]
pub struct GpuKernels {
    pub h_field_compute: FdtdDirichlet
}

pub async fn main_render_loop(
    backend: &GpuBackend,
    input: &FdtdData
) -> Result<(), GpuBackendError> {
    let grid_info = &input.grid_info;
    let mut window = Window::new("Compute Shader Testing").await;
    let mut camera = OrbitCamera3d::default();
    let mut scene = SceneNode3d::empty();
    scene.add_light(Light::point(100.))
        .set_position(Vec3::new(10., 10., 10.));
    // Draw data
    let bb_extents = grid_info.grid_dimensions.as_vec3() * grid_info.cell_size;
    let bb_poly_line = bb_polyline(bb_extents, grid_info.position);

    let mut arrow_polylines = Vec::with_capacity(grid_info.grid_dimensions.element_product() as usize);
    for i in 0..grid_info.grid_dimensions.element_product() {
        let v = Vec3::NEG_Z * grid_info.cell_size / 2.;
        let mut polyline = arrow_polyline(Vec3::ZERO, v);
        let cell_position = flat_idx_to_vector(i, grid_info.grid_dimensions)
            .as_vec3() * grid_info.cell_size;
        polyline.transform = Pose3::from_translation(cell_position);
        arrow_polylines.push(polyline);
    }
    for src in input.sources.iter() {
        let pos = flat_idx_to_vector(src.cell_idx, grid_info.grid_dimensions).as_vec3()
            .mul(grid_info.cell_size);
        scene.add_sphere(0.1)
            .set_position(pos)
            .set_color(RED);
    }

    let mut buffers = shader::fdtd::create_buffers(backend, input)?;
    let mut cells_read_buf = backend.init_buffer(
        vec![GridCell::default(); buffers.cells.len()].as_slice(),
        BufferUsages::COPY_DST | BufferUsages::MAP_READ
    )?;
    // Render the window
    while window.render_3d(&mut scene, &mut camera).await {

        if window.get_key(Key::T) == Action::Press {
            backend.synchronize()?;
            let mut out = vec![GridCell::default(); buffers.cells.len()];
            backend.read_buffer(&cells_read_buf, out.as_mut_slice()).await?;

            submit_simulation(
                &backend,
                &mut buffers,
                &mut cells_read_buf,
                grid_info
            )?;

            // Update arrow polylines
            for (arrow, dir) in arrow_polylines.iter_mut()
                .zip(out.iter().map(|c| c.e.normalize() * grid_info.cell_size / 2.))
            {
                let pos = arrow.transform.translation;
                arrow.transform = Pose3::look_at_rh(Vec3::ZERO, dir, Vec3::Y)
                    .append_translation(pos);
            }
            println!("{}", out[vector_to_flat_idx(UVec3::Z * 3, grid_info.grid_dimensions) as usize].e);
        }

        // Draw polylines
        window.draw_polyline(&bb_poly_line);
        arrow_polylines.iter().for_each(|a| window.draw_polyline(a))
    }

    Ok(())
}

pub fn submit_simulation(
    backend: &GpuBackend,
    buffers: &mut MaxwellEqsBuffers,
    cells_read_buf: &mut GpuBuffer<GridCell>,
    grid_info: &GridInfo
) -> Result<(), GpuBackendError> {
    let kernels = GpuKernels::from_backend(&backend)?;
    let mut encoder = backend.begin_encoding();
    let mut pass = encoder.begin_pass("e_field_compute", None);
    let workgroup_count = grid_info.grid_dimensions.map(|v| v.div_ceil(4)).to_array();
    kernels.h_field_compute.call(
        &mut pass,
        DispatchGrid::Grid(workgroup_count),
        &mut buffers.cells,
        &buffers.material_constants,
        &buffers.source_values,
        &mut buffers.sources,
        &buffers.grid_info
    )?;
    drop(pass);
    encoder.copy_buffer_to_buffer(
        &buffers.cells, 0,
        cells_read_buf, 0,
        buffers.cells.len()
    )?;
    backend.submit(encoder)?;
    Ok(())
}