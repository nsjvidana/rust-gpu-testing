use glam::{UVec3, Vec3, Vec4, Vec4Swizzles};
use khal::backend::{Backend, Buffer, DispatchGrid, Encoder, GpuBackend, GpuBackendError, GpuBuffer};
use khal::{BufferUsages, Shader};
use kiss3d::camera::OrbitCamera3d;
use kiss3d::event::{Action, Key};
use kiss3d::light::Light;
use kiss3d::prelude::{Polyline3d, Pose3, SceneNode3d, Window, RED};
use shader_crate::fdtd_3d::{Fdtd3d, GpuSource, GridCell, GridInfo, MaterialConstants};
use shader_crate::{flat_idx_to_vector, vector_to_flat_idx};
use crate::util::{arrow_polyline, bb_polyline};

#[derive(Shader)]
struct GpuKernels {
    pub fdtd_3d: Fdtd3d
}

pub async fn run_fdtd_3d(backend: &GpuBackend) {
    let pulse_freq = 1e6;

    let mut grid_info = GridInfo::max_values(Vec3::ZERO);
    grid_info.min_wavelength(pulse_freq, 1., 20);
    grid_info.courant_stability_condition(1., 6.);
    grid_info.set_dimensions(grid_info.cell_size * 30.);

    let pulse = GaussianPulse::from_max_frequency(pulse_freq, 1., grid_info.dimensions.xyz()/2., 100);
    grid_info.account_for_pulse(pulse.tau, 10);

    println!("{grid_info:?}");

    let cell_count = grid_info.idx_dims.xyz().element_product() as usize;
    let mut data = Fdtd3dData {
        cells: vec![GridCell::default(); cell_count],
        materials: vec![MaterialConstants::new_linear(1., 1., grid_info.dt)],
        grid_info: grid_info.clone(),
        ..Default::default()
    };
    pulse.add_source(&mut data.sources, &mut data.source_vals, &grid_info);

    main_render_loop(backend, data).await.unwrap();
}

async fn main_render_loop(backend: &GpuBackend, data: Fdtd3dData) -> Result<(), GpuBackendError> {
    let grid_info = &data.grid_info;
    let idx_dims = grid_info.idx_dims.xyz();

    let mut window = Window::new("Compute Shader Testing").await;
    let mut camera = OrbitCamera3d::default();
    camera.look_at(
        idx_dims.as_vec3() * grid_info.cell_size * 2.,
        idx_dims.as_vec3() * grid_info.cell_size / 2.
    );
    let mut scene = SceneNode3d::empty();
    scene.add_light(Light::point(1000.))
        .set_position(idx_dims.as_vec3() * grid_info.cell_size * 2.);

    let kernels = GpuKernels::from_backend(&backend)?;
    let mut buffers = data.create_buffers(backend)?;

    let num_cells = idx_dims.element_product() as usize;
    let mut cells_out = vec![GridCell::default(); num_cells];
    let bb_polyline = bb_polyline(idx_dims.as_vec3()*grid_info.cell_size, Vec3::ZERO);
    let mut arrow = arrow_polyline(Vec3::ZERO, Vec3::NEG_Z * grid_info.cell_size.min_element() / 2.);
        arrow.color = RED;
    let mut arrows = std::iter::repeat(arrow)
        .take(num_cells)
        .collect::<Vec<_>>();
    for (i, arrow) in arrows.iter_mut().enumerate() {
        let pos = flat_idx_to_vector(i as _, idx_dims).as_vec3() * grid_info.cell_size;
        arrow.transform = Pose3::look_at_rh(Vec3::ZERO, Vec3::NEG_Z, Vec3::Y)
            .append_translation(pos);
    }
    for src in data.sources.iter() {
        let pos = flat_idx_to_vector(src.cell_idx, idx_dims)
            .as_vec3() * grid_info.cell_size;
        scene.add_sphere(grid_info.cell_size.min_element() / 3.)
            .translate(pos)
            .set_color(RED);
    }

    // Main render loop
    let mut l = 0.;
    while window.render_3d(&mut scene, &mut camera).await {
        let update_sim = window.get_key(Key::T) == Action::Press;
        if update_sim {
            backend.synchronize()?;
            backend.read_buffer(&buffers.cells_read, &mut cells_out).await?;

            submit_simulation(
                backend,
                &kernels,
                &mut buffers,
                grid_info
            )?;
        }

        let max_len = cells_out.iter()
            .map(|c| c.e.length_squared())
            .max_by(|a, b| a.total_cmp(b))
            .unwrap();
        if max_len > l {
            println!("max E magnitude: {}", max_len);
            l = max_len;
        }
        for (i, c) in cells_out.iter().enumerate() {
            let e = c.e;
            let relative_len = e.length_squared() / max_len;
            let pos = flat_idx_to_vector(i as _, idx_dims).as_vec3() * grid_info.cell_size;
            let dir = e.normalize_or(Vec3::NEG_Z) * grid_info.cell_size / 2.;

            window.draw_line(pos, pos + dir, RED.with_alpha((relative_len*1e4).min(1.)), 2.0, false);
        }

        window.draw_polyline(&bb_polyline);
    }
    Ok(())
}

fn submit_simulation(
    backend: &GpuBackend,
    kernels: &GpuKernels,
    buffers: &mut Fdtd3dBuffers,
    grid_info: &GridInfo
) -> Result<(), GpuBackendError> {
    let mut encoder = backend.begin_encoding();
    let mut pass = encoder.begin_pass("", None);
    let dispatch_grid = [
        grid_info.idx_dims.x.div_ceil(4),
        grid_info.idx_dims.y.div_ceil(4),
        grid_info.idx_dims.z.div_ceil(4),
    ];
    kernels.fdtd_3d.call(
        &mut pass,
        DispatchGrid::Grid(dispatch_grid),
        &mut buffers.cells,
        &mut buffers.materials,
        &buffers.source_vals,
        &mut buffers.sources,
        &buffers.grid_info
    )?;
    drop(pass);
    encoder.copy_buffer_to_buffer(
        &buffers.cells,
        0,
        &mut buffers.cells_read,
        0,
        buffers.cells.len()
    )?;
    backend.submit(encoder)
}

#[derive(Default)]
pub struct Fdtd3dData {
    pub cells: Vec<GridCell>,
    pub materials: Vec<MaterialConstants>,
    pub source_vals: Vec<Vec4>,
    pub sources: Vec<GpuSource>,
    pub grid_info: GridInfo
}

impl Fdtd3dData {
    pub fn create_buffers(&self, backend: &GpuBackend) -> Result<Fdtd3dBuffers, GpuBackendError> {
        Ok(Fdtd3dBuffers {
            cells: backend.init_buffer(
                self.cells.as_slice(),
                BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            )?,
            materials: backend.init_buffer(
                self.materials.as_slice(),
                BufferUsages::STORAGE,
            )?,
            source_vals: backend.init_buffer(
                self.source_vals.as_slice(),
                BufferUsages::STORAGE,
            )?,
            sources: backend.init_buffer(
                self.sources.as_slice(),
                BufferUsages::STORAGE,
            )?,
            grid_info: backend.init_buffer(
                &[self.grid_info],
                BufferUsages::UNIFORM,
            )?,
            cells_read: backend.init_buffer(
                self.cells.as_slice(),
                BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            )?,
        })
    }
}

pub struct Fdtd3dBuffers {
    pub cells: GpuBuffer<GridCell>,
    pub materials: GpuBuffer<MaterialConstants>,
    pub source_vals: GpuBuffer<Vec4>,
    pub sources: GpuBuffer<GpuSource>,
    pub grid_info: GpuBuffer<GridInfo>,

    pub cells_read: GpuBuffer<GridCell>,
}

pub struct GaussianPulse {
    pub amplitude: f32,
    pub tau: f32,
    pub t_0: f32,
    /// Location on Z axis
    pub location: Vec3,
    pub resolution: u32,
}

impl GaussianPulse {
    /// `resolution` is the number of data points of this source. It can't be zero.
    ///
    /// # Simulation Stability
    /// **HIGHLY** recommended to use [`GridInfo1D::account_for_pulse`] when using a [`GaussianPulse`].
    pub fn from_max_frequency(max_frequency: f32, amplitude: f32, at_point: Vec3, resolution: u32) -> Self {
        debug_assert_ne!(resolution, 0);
        let tau = 0.5 / max_frequency;
        Self {
            amplitude,
            tau,
            t_0: 6. * tau,
            location: at_point,
            resolution
        }
    }

    pub fn add_source(
        &self,
        sources: &mut Vec<GpuSource>,
        source_values: &mut Vec<Vec4>,
        grid_info: &GridInfo
    ) {
        let mut vals = vec![Vec4::ZERO; self.resolution as usize];
        let mut t = 0.;
        for i in 0..self.resolution {
            t += grid_info.dt;
            let g = core::f32::consts::E.powf(
                -((t - self.t_0) / self.tau).powi(2)
            );
            vals[i as usize] = Vec4::new(g, g, g, 0.);
        }

        let start_idx = source_values.len() as u32;
        source_values.extend_from_slice(&vals);
        let end_idx = source_values.len() as u32 - 1;
        let cell_idx_v = (self.location / grid_info.cell_size).round().as_uvec3();
        let cell_idx = vector_to_flat_idx(cell_idx_v, grid_info.idx_dims.xyz());
        sources.push(GpuSource {
            start_idx,
            end_idx,
            curr_idx: 0,
            cell_idx,
        });
    }
}