use crate::util::{CreateGpuBuffer, CreateGpuBufferReadable, GpuBufferReadable};
use glam::Vec3;
use khal::backend::{Backend, Buffer, DispatchGrid, Encoder, GpuBackend, GpuBackendError, GpuBuffer};
use khal::Shader;
use kiss3d::camera::OrbitCamera3d;
use kiss3d::event::{Action, Key};
use kiss3d::light::Light;
use kiss3d::prelude::{Color, Polyline3d, SceneNode3d, Window, GREEN, RED};
use shader_crate::fdtd_1d::{Fdtd1d, GpuSource1D, GridCell1D, GridInfo1D, MaterialConstants1D, PerfectBoundaryData};

#[derive(Shader)]
struct GpuKernels {
    pub fdtd_1d: Fdtd1d
}

pub async fn run_fdtd_1d(backend: &GpuBackend) {
    let max_pulse_freq = 10e6;
    let simulation_dimensions_z =  800.;

    let eps_r_mat = 2.0_f32;
    let mu_r_mat = 0.5_f32;
    let n_mat = (eps_r_mat * mu_r_mat).sqrt();

    let n_max = n_mat.max(1.);
    let n_min = n_mat.min(1.);

    let mut grid_info = GridInfo1D::max_values(simulation_dimensions_z);
    grid_info
        .min_wavelength(max_pulse_freq, n_max, 20);
    grid_info
        .set_step_count(2);

    let pulse = GaussianPulse1D::from_max_frequency(
        max_pulse_freq,
        1.,
        grid_info.dimensions/5.,
        &mut grid_info,
        20
    );
    println!("{:?}", pulse);

    grid_info.set_cfl_perfect_boundary(1.);
    println!("{grid_info:?}");

    let obj = ObjectInfo1D {
        width: grid_info.dimensions / 10.,
        position: grid_info.dimensions / 2.,
        material_constants: MaterialConstants1D::new(eps_r_mat, mu_r_mat, grid_info.dt),
        color: GREEN
    };

    let mut data = Fdtd1dData {
        cells: vec![GridCell1D::default(); grid_info.num_cells as usize],
        materials: vec![
            MaterialConstants1D::new(1., 1., grid_info.dt),
            obj.material_constants,
        ],
        grid_info: grid_info.clone(),
        ..Default::default()
    };
    data.add_object(obj);
    pulse.add_source(&mut data.sources, &mut data.source_vals, &grid_info);
    let max_src_val = data.source_vals.iter()
        .map(|v| v.abs())
        .max_by(|a, b| a.total_cmp(b))
        .unwrap();
    println!("max src val: {}", max_src_val);

    main_render_loop(backend, data, max_src_val).await.unwrap();
}

async fn main_render_loop(
    backend: &GpuBackend,
    mut data: Fdtd1dData,
    max_src_val: f32,
) -> Result<(), GpuBackendError> {
    let grid_info = &data.grid_info;
    let mut window = Window::new("Compute Shader Testing").await;
    let mut camera = OrbitCamera3d::new_with_frustum(
        core::f32::consts::PI / 4.0,
        0.1,
        grid_info.dimensions * 3.,
        Vec3::new(-grid_info.dimensions, 0., grid_info.dimensions / 2.),
        Vec3::new(0., 0., grid_info.dimensions / 2.)
    );
    let mut scene = SceneNode3d::empty();
    scene.add_light(Light::point(1000.))
        .set_position(Vec3::new(1., 1., 0.));

    let kernels = GpuKernels::from_backend(&backend)?;
    let mut buffers = data.create_buffers(backend)?;

    let axis_line = Polyline3d::new(vec![Vec3::ZERO, Vec3::Z * grid_info.dimensions]);
    for src in data.sources.iter() {
        let pos = Vec3::Z * src.cell_idx as f32 * grid_info.cell_size;
        scene.add_sphere(grid_info.cell_size / 5.)
            .translate(pos)
            .set_color(RED);
    }

    let mut abs_max_val = 0.;
    let mut cells_out = vec![GridCell1D::default(); grid_info.num_cells as usize];
    let mut boundary_out = vec![PerfectBoundaryData::default()];
    let mut prev_action = Action::Release;
    while window.render_3d(&mut scene, &mut camera).await {
        let curr_action = window.get_key(Key::T);
        if window.get_key(Key::LControl) != Action::Press {
            prev_action = Action::Release;
        }
        if curr_action != prev_action && curr_action == Action::Press {
            backend.synchronize()?;
            buffers.cells.read(backend, &mut cells_out).await?;
            buffers.perfect_boundary_data.read(backend, &mut boundary_out).await?;

            submit_simulation(
                backend,
                &kernels,
                &mut buffers,
                grid_info
            )?;
        }
        prev_action = curr_action;

        let max_val = cells_out.iter()
            .map(|c| c.e_y.abs())
            .max_by(|a, b| a.total_cmp(b))
            .unwrap();
        if max_val > abs_max_val {
            println!("max E magn: {max_val}");
            abs_max_val = max_val;
        }
        let mut prev = {
            let c = &cells_out[0];
            let relative_len = c.e_y / max_src_val;
            let pos = Vec3::ZERO;
            let dir = Vec3::new(0., relative_len * grid_info.dimensions / 10., 0.);
            pos + dir
        };
        for (i, c) in cells_out.iter().enumerate().skip(1) {
            let relative_len = c.e_y / max_src_val;
            let pos = Vec3::Z * i as f32 * grid_info.cell_size;
            let dir = Vec3::new(0., relative_len * grid_info.dimensions / 10., 0.);
            let curr = pos + dir;
            window.draw_line(prev, curr, RED, 2.0, false);
            prev = curr;
        }

        let half_line = Vec3::new(0., grid_info.dimensions / 10., 0.);
        for obj in data.objects.iter() {
            let (start_idx, idx_width) = data.get_obj_indices(obj);
            let start = Vec3::new(0., 0., start_idx as f32 * grid_info.cell_size);
            let end = (start_idx + idx_width).min(data.cells.len()-1) as f32 * grid_info.cell_size;
                let end = Vec3::new(0., 0., end);
            window.draw_line(start + half_line, start - half_line, obj.color, 2.0, false);
            window.draw_line(end + half_line, end - half_line, obj.color, 2.0, false);
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
    for _ in 0..grid_info.step_count {
        kernels.fdtd_1d.call(
            &mut pass,
            DispatchGrid::Grid([grid_info.num_cells.div_ceil(64), 1, 1]),
            &mut buffers.cells.buffer,
            &mut buffers.materials,
            &buffers.source_vals,
            &mut buffers.sources,
            &mut buffers.perfect_boundary_data.buffer,
            &buffers.grid_info
        )?;
    }
    drop(pass);
    buffers.cells.encode_copy_cmd(&mut encoder)?;
    buffers.perfect_boundary_data.encode_copy_cmd(&mut encoder)?;
    backend.submit(encoder)
}

#[derive(Default)]
pub struct Fdtd1dData {
    pub cells: Vec<GridCell1D>,
    pub materials: Vec<MaterialConstants1D>,
    pub source_vals: Vec<f32>,
    pub sources: Vec<GpuSource1D>,
    pub grid_info: GridInfo1D,

    pub objects: Vec<ObjectInfo1D>,
}

impl Fdtd1dData {
    pub fn create_buffers(&self, backend: &GpuBackend) -> Result<Fdtd1dBuffers, GpuBackendError> {
        Ok(Fdtd1dBuffers {
            cells: self.cells.create_gpu_buffer_readable(backend)?,
            materials: self.materials.create_gpu_buffer(backend)?,
            source_vals: self.source_vals.create_gpu_buffer(backend)?,
            sources: self.sources.create_gpu_buffer(backend)?,
            perfect_boundary_data: PerfectBoundaryData::default()
                .create_gpu_buffer_readable(backend)?,
            grid_info: self.grid_info.create_gpu_uniform(backend)?,
        })
    }

    pub fn add_object(&mut self, obj: ObjectInfo1D) -> &mut Self {
        let mat_idx = self.materials.len() as u32;
        self.materials.push(obj.material_constants);

        let (start, idx_width) = self.get_obj_indices(&obj);
        for cell in self.cells.iter_mut()
            .skip(start)
            .take(idx_width)
        {
            cell.material_idx = mat_idx;
        }
        self.objects.push(obj);
        self
    }

    /// Returns starting cell index and the width of the object in grid cells: `(start_idx, idx_width)`
    pub fn get_obj_indices(&self, obj: &ObjectInfo1D) -> (usize, usize) {
        let center_idx = (obj.position / self.grid_info.cell_size) as usize;
        let idx_width = (obj.width / self.grid_info.cell_size).ceil() as usize;
        let start = (center_idx - idx_width.div_ceil(2)).max(0);
        (start, idx_width)
    }
}

#[derive(Default)]
pub struct ObjectInfo1D {
    pub width: f32,
    pub position: f32,
    pub material_constants: MaterialConstants1D,
    pub color: Color
}

pub struct Fdtd1dBuffers {
    pub cells: GpuBufferReadable<GridCell1D>,
    pub materials: GpuBuffer<MaterialConstants1D>,
    pub source_vals: GpuBuffer<f32>,
    pub sources: GpuBuffer<GpuSource1D>,
    pub perfect_boundary_data: GpuBufferReadable<PerfectBoundaryData>,
    pub grid_info: GpuBuffer<GridInfo1D>,
}

#[derive(Debug)]
pub struct GaussianPulse1D {
    pub amplitude: f32,
    pub tau: f32,
    pub t_0: f32,
    /// Location on Z axis
    pub location: f32,
    pub resolution: u32,
}

impl GaussianPulse1D {
    /// Create a Gaussian Pulse that has a maximum frequency of `max_frequency`
    ///
    /// # Simulation Stability
    /// **HIGHLY** recommended to use [`GridInfo1D::account_for_pulse`] when using a [`GaussianPulse1D`].
    /// Have `cells_resolution >= 10` for better results
    pub fn from_max_frequency(
        max_frequency: f32,
        amplitude: f32,
        at_point: f32,
        grid_info: &mut GridInfo1D,
        cells_resolution: u32,
    ) -> Self {
        let tau = 0.5 / max_frequency;

        grid_info.dt = grid_info.dt.min(tau / cells_resolution as f32);

        let approx_pulse_duration = 12. * tau;
        let resolution = (approx_pulse_duration / grid_info.dt).ceil() as u32;
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
            vals[i as usize] = g * self.amplitude;
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