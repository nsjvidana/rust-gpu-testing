use glam::Vec3;
use khal::backend::{Backend, Buffer, DispatchGrid, Encoder, GpuBackend, GpuBackendError, GpuBuffer};
use khal::{BufferUsages, Shader};
use kiss3d::camera::OrbitCamera3d;
use kiss3d::event::{Action, Key};
use kiss3d::light::Light;
use kiss3d::prelude::{Polyline3d, Pose3, SceneNode3d, Window, RED};
use shader_crate::fdtd_1d::{Fdtd1d, GpuSource1D, GridCell1D, GridInfo1D, MaterialConstants1D};

#[derive(Shader)]
struct GpuKernels {
    pub fdtd_1d: Fdtd1d
}

pub async fn run_fdtd_1d(backend: &GpuBackend) {
    let pulse_freq = 1e6;

    let mut grid_info = GridInfo1D::max_values(0.);
    grid_info.min_wavelength(pulse_freq, 1., 20);
    grid_info.courant_stability_condition(1., 2.);
    grid_info.set_dimensions(grid_info.cell_size * 30.);

    let pulse = GaussianPulse1D::from_max_frequency(pulse_freq, 1., grid_info.dimensions/2., 100);
    grid_info.account_for_pulse(pulse.tau, 10);

    println!("{grid_info:?}");

    let mut data = Fdtd1dData {
        cells: vec![GridCell1D::default(); grid_info.num_cells as usize],
        materials: vec![MaterialConstants1D::new(1., 1., grid_info.dt)],
        grid_info: grid_info.clone(),
        ..Default::default()
    };
    pulse.add_source(&mut data.sources, &mut data.source_vals, &grid_info);

    main_render_loop(backend, data).await.unwrap();
}

async fn main_render_loop(backend: &GpuBackend, data: Fdtd1dData) -> Result<(), GpuBackendError> {
    let grid_info = &data.grid_info;
    let mut window = Window::new("Compute Shader Testing").await;
    let mut camera = OrbitCamera3d::default();
    camera.look_at(
        Vec3::new(1., 1., grid_info.dimensions),
        Vec3::ZERO
    );
    let mut scene = SceneNode3d::empty();
    scene.add_light(Light::point(1000.))
        .set_position(Vec3::new(1., 1., 0.));

    let kernels = GpuKernels::from_backend(&backend)?;
    let mut buffers = data.create_buffers(backend)?;

    let mut cells_out = vec![GridCell1D::default(); grid_info.num_cells as usize];
    let axis_line = Polyline3d::new(vec![Vec3::ZERO, Vec3::Z * grid_info.dimensions]);
    for src in data.sources.iter() {
        let pos = Vec3::Z * src.cell_idx as f32 * grid_info.cell_size;
        scene.add_sphere(grid_info.cell_size / 5.)
            .translate(pos)
            .set_color(RED);
    }

    let mut abs_max_val = 0.;
    while window.render_3d(&mut scene, &mut camera).await {
        if window.get_key(Key::T) == Action::Press {
            backend.synchronize()?;
            backend.read_buffer(&buffers.cells_read, &mut cells_out).await?;

            submit_simulation(
                backend,
                &kernels,
                &mut buffers,
                grid_info
            )?;
        }

        let max_val = cells_out.iter()
            .map(|c| c.e_y.abs())
            .max_by(|a, b| a.total_cmp(b))
            .unwrap();
        if max_val > abs_max_val {
            println!("max E magn: {max_val}");
            abs_max_val = max_val;
        }
        for (i, c) in cells_out.iter().enumerate() {
            let relative_len = c.e_y / max_val;
            let pos = Vec3::Z * i as f32 * grid_info.cell_size;
            let dir = Vec3::Y * relative_len * grid_info.cell_size;

            window.draw_line(pos, pos + dir, RED, 2.0, false);
        }

        window.draw_polyline(&axis_line);
    }
    Ok(())
}

fn submit_simulation(
    backend: &GpuBackend,
    kernels: &GpuKernels,
    buffers: &mut Fdtd1dBuffers,
    grid_info: &GridInfo1D
) -> Result<(), GpuBackendError> {
    let mut encoder = backend.begin_encoding();
    let mut pass = encoder.begin_pass("", None);
    kernels.fdtd_1d.call(
        &mut pass,
        DispatchGrid::Grid([grid_info.num_cells.div_ceil(64), 1, 1]),
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
pub struct Fdtd1dData {
    pub cells: Vec<GridCell1D>,
    pub materials: Vec<MaterialConstants1D>,
    pub source_vals: Vec<f32>,
    pub sources: Vec<GpuSource1D>,
    pub grid_info: GridInfo1D
}

impl Fdtd1dData {
    pub fn create_buffers(&self, backend: &GpuBackend) -> Result<Fdtd1dBuffers, GpuBackendError> {
        Ok(Fdtd1dBuffers {
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

pub struct Fdtd1dBuffers {
    pub cells: GpuBuffer<GridCell1D>,
    pub materials: GpuBuffer<MaterialConstants1D>,
    pub source_vals: GpuBuffer<f32>,
    pub sources: GpuBuffer<GpuSource1D>,
    pub grid_info: GpuBuffer<GridInfo1D>,

    pub cells_read: GpuBuffer<GridCell1D>,
}

pub struct GaussianPulse1D {
    pub amplitude: f32,
    pub tau: f32,
    pub t_0: f32,
    /// Location on Z axis
    pub location: f32,
    pub resolution: u32,
}

impl GaussianPulse1D {
    /// `resolution` is the number of data points of this source. It can't be zero.
    ///
    /// # Simulation Stability
    /// **HIGHLY** recommended to use [`GridInfo1D::account_for_pulse`] when using a [`GaussianPulse1D`].
    pub fn from_max_frequency(max_frequency: f32, amplitude: f32, at_point: f32, resolution: u32) -> Self {
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
        sources: &mut Vec<GpuSource1D>,
        source_values: &mut Vec<f32>,
        grid_info: &GridInfo1D
    ) {
        let mut vals = vec![0.; self.resolution as usize];
        let mut t = 0.;
        for i in 0..self.resolution {
            t += grid_info.dt;
            let g = core::f32::consts::E.powf(
                -((t - self.t_0) / self.tau).powi(2)
            );
            vals[i as usize] = g;
        }

        let start_idx = source_values.len() as u32;
        source_values.extend_from_slice(&vals);
        let end_idx = source_values.len() as u32 - 1;
        let cell_idx = (self.location / grid_info.cell_size).round() as u32;
        sources.push(GpuSource1D {
            start_idx,
            end_idx,
            curr_idx: 0,
            cell_idx,
        });
    }
}