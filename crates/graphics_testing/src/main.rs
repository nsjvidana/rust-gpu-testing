pub mod shader;
pub mod error;
mod prelude;
mod util;

use std::ops::{Div, Mul};
use crate::shader::fdtd::{FdtdData, GaussianPulse, MaxwellEqsBuffers, ELECTRON_MASS, ELEMENTARY_CHARGE, PROTON_MASS};
use crate::util::{arrow_polyline, bb_polyline, flat_idx_to_vector};
use glam::{UVec3, Vec3};
use include_dir::{include_dir, Dir};
use khal::backend::{Backend, Buffer, DispatchGrid, Encoder, GpuBackend, GpuBackendError, GpuBuffer, WebGpu};
use khal::{AsGpuSlice, BufferUsages, Shader};
use kiss3d::prelude::*;
use rand::{RngExt, SeedableRng};
use shader_crate::{vector_to_flat_idx, FdtdDirichlet, GridCell, GridInfo, MaterialConstants, PointCharge};

static SPIRV_DIR: Dir<'static> = include_dir!("$CARGO_MANIFEST_DIR/shaders-spirv");

#[kiss3d::main]
async fn main() {
    let webgpu = WebGpu::default().await.unwrap();
    let backend = GpuBackend::WebGpu(webgpu);

    // Generating input data for the shader
    let pulse_max_freq = 10e6;

    let grid_dimensions = Vec3::splat(30.);
    let mut cell_size = Vec3::splat(f32::MAX); // gets minimized greatly by then next line
    GridInfo::cell_size_for_frequency(&mut cell_size, pulse_max_freq, 1., 20);
    // TODO: adjust for min feature len of a device
    GridInfo::cube_cell_size(&mut cell_size);

    let mut dt = f32::MAX;
    GridInfo::courant_stability_condition(&mut dt, cell_size, 1.);

    let mut grid_info = GridInfo::new(grid_dimensions, cell_size, dt);
    println!("{grid_info:?}");

    let pulse = GaussianPulse::from_max_frequency(pulse_max_freq, 1., grid_info.cell_size * 2., &grid_info);
    GridInfo::adjust_dt_for_gaussian_pulse(&mut grid_info.dt, pulse.half_duration, 20);

    let mut input_data = FdtdData::new(grid_info).unwrap();
        input_data.sources.push(pulse.construct_source(100, grid_info.dt, 0.));
    input_data.prepare_for_simulation().unwrap();

    main_render_loop(&backend, &input_data).await.unwrap();
}

#[derive(Shader)]
pub struct GpuKernels {
    pub fdtd_dirichlet: FdtdDirichlet
}

pub async fn main_render_loop(
    backend: &GpuBackend,
    input: &FdtdData
) -> Result<(), GpuBackendError> {
    let grid_info = &input.grid_info;
    let mut window = Window::new("Compute Shader Testing").await;
    let mut camera = OrbitCamera3d::default();
    camera.look_at(
        grid_info.idx_dimensions.as_vec3() * grid_info.cell_size * 2.,
        grid_info.idx_dimensions.as_vec3() * grid_info.cell_size / 2.
    );
    let mut scene = SceneNode3d::empty();
    scene.add_light(Light::point(1000.))
        .set_position(grid_info.idx_dimensions.as_vec3() * grid_info.cell_size);

    // Draw data
    let bb_extents = grid_info.idx_dimensions.as_vec3() * grid_info.cell_size;
    let bb_poly_line = bb_polyline(bb_extents, grid_info.position);
    let mut arrow_polylines = Vec::with_capacity(grid_info.idx_dimensions.element_product() as usize);
        let v = Vec3::NEG_Z * grid_info.cell_size / 2.;
        let mut red_arrow = arrow_polyline(Vec3::ZERO, v).with_color(RED);
        for i in 0..grid_info.idx_dimensions.element_product() {
            let cell_position = flat_idx_to_vector(i, grid_info.idx_dimensions)
                .as_vec3() * grid_info.cell_size;
            let mut polyline = red_arrow.clone();
            polyline.transform = Pose3::from_translation(cell_position);
            arrow_polylines.push(polyline);
        }
    for src in input.sources.iter() {
        let pos = flat_idx_to_vector(src.cell_idx, grid_info.idx_dimensions).as_vec3()
            .mul(grid_info.cell_size);
        scene.add_sphere(2.)
            .translate(pos)
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
            let mut cells_out = vec![GridCell::default(); buffers.cells.len()];
            backend.read_buffer(&cells_read_buf, cells_out.as_mut_slice()).await?;

            // Update arrow polylines
            let lengths = cells_out.iter().map(|c| c.e.length_squared())
                .collect::<Vec<_>>();
            let max_len = lengths.iter()
                .max_by(|a, b| a.total_cmp(b))
                .unwrap();
            for i in 0..cells_out.len() {
                let arrow = &mut arrow_polylines[i];
                let e = cells_out[i].e;

                // if e.length_squared() != 0. {
                //     println!("{}", e);
                // }

                let pos = arrow.transform.translation;
                let relative_len = e.length_squared() / max_len;
                let dir = e.normalize_or(Vec3::Z);
                arrow.transform = Pose3::look_at_rh(Vec3::ZERO, dir, Vec3::Y)
                    .append_translation(pos);
                arrow.color = RED.with_alpha(relative_len);
            }
            // println!("-------------");

            submit_simulation(
                &backend,
                &mut buffers,
                &mut cells_read_buf,
                grid_info
            )?;
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
    let workgroup_count = grid_info.idx_dimensions.map(|v| v.div_ceil(4)).to_array();
    kernels.fdtd_dirichlet.call(
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
        &buffers.cells,
        0,
        cells_read_buf,
        0,
        buffers.cells.len()
    )?;
    backend.submit(encoder)?;
    Ok(())
}