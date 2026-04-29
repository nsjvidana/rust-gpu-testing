use std::ops::Range;
use crate::util::{CreateGpuBuffer, CreateGpuBufferReadable, GpuBufferReadable};
use glam::Vec3;
use khal::backend::{Backend, Buffer, DispatchGrid, Encoder, GpuBackend, GpuBackendError, GpuBuffer};
use khal::Shader;
use kiss3d::camera::OrbitCamera3d;
use kiss3d::event::{Action, Key};
use kiss3d::light::Light;
use kiss3d::prelude::{Color, Polyline3d, SceneNode3d, Window, GREEN, RED, WHITE};
use shader_crate::fdtd_1d::{Fdtd1d, GpuSource1D, GridCell1D, GridInfo1D, MaterialConstants1D, PerfectBoundaryData};

#[derive(Shader)]
struct GpuKernels {
    pub fdtd_1d: Fdtd1d
}

pub async fn run_fdtd_1d(backend: &GpuBackend) {
    let max_pulse_freq = 1e9;
    let simulation_dimensions_z =  800.;

    let eps_r_mat = 6.0_f32;
    let mu_r_mat = 2.0_f32;
    let n_mat = (eps_r_mat * mu_r_mat).sqrt();

    let n_max = n_mat.max(1.);
    let n_min = n_mat.min(1.);

    let stability_values = StabilityValues {
        spacer_region_cells: 10,
        ..Default::default()
    };

    let obj = ObjectInfo1D {
        width: 0.3048, // 1ft wide
        position: 0.,
        material: ElectricMaterial::new(eps_r_mat, mu_r_mat),
        color: GREEN
    };

    let mut _dummy_grid_info = GridInfo1D::max_values(0.); // for now just use a dummy grid info
    let source = GaussianPulse1D::from_max_frequency(
        max_pulse_freq,
        1.,
        0., // dummy position value for now
        &mut _dummy_grid_info,
        20
    );

    let mut data = Fdtd1dData::new();
    data
        .add_object(obj)
        .add_source(source);

    data.prepare_for_gpu(&stability_values, None);
    data.grid_info.set_step_count(1);
    println!("{:?}", data.grid_info);

    let max_src_val = data.source_vals.iter()
        .map(|v| v.abs())
        .max_by(|a, b| a.total_cmp(b))
        .unwrap();

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

    for src in data.sources_gpu.iter() {
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
        for (obj, cell_range) in data.objects.iter().zip(data.object_cell_indices.iter()) {
            let start = Vec3::new(0., 0., cell_range.start as f32 * grid_info.cell_size);
            let end = Vec3::new(0., 0., cell_range.end as f32 * grid_info.cell_size);
            window.draw_line(start + half_line, start - half_line, obj.color, 2.0, false);
            window.draw_line(end + half_line, end - half_line, obj.color, 2.0, false);
        }

        window.draw_line(Vec3::ZERO, Vec3::Z * grid_info.dimensions, WHITE, 2.0, false);
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
    pub sources: Vec<GaussianPulse1D>,
    pub source_vals: Vec<f32>,
    pub sources_gpu: Vec<GpuSource1D>,
    pub grid_info: GridInfo1D,

    pub objects: Vec<ObjectInfo1D>,
    /// The range of cells each object takes
    pub object_cell_indices: Vec<Range<usize>>,
}

impl Fdtd1dData {
    pub fn new() -> Self {
        Self {
            grid_info: GridInfo1D::max_values(0.),
            ..Default::default()
        }
    }

    pub fn prepare_for_gpu(
        &mut self,
        stability: &StabilityValues,
        custom_default_material: Option<ElectricMaterial>
    ) -> &mut Self {
        // Temporary materials vec since no stable dt exists
        let mut materials = vec![];
        let default_mat = custom_default_material.unwrap_or(ElectricMaterial::FREE_SPACE);
        materials.push(default_mat);
        let mut obj_material_idxs = vec![0; self.objects.len()];
        for (i, obj) in self.objects.iter().enumerate() {
            let mat = obj.material;
            let mat_idx = materials.iter()
                .position(|m| m.eq(&mat))
                .unwrap_or_else(|| {
                    let mat_idx = materials.len();
                    materials.push(mat);
                    mat_idx
                });
            obj_material_idxs[i] = mat_idx;
        }

        self.enforce_stability_conditions(&stability, materials[0].n);
        let dt = self.grid_info.dt;
        let dz = self.grid_info.cell_size;

        // Recompute material constants with stable dt
        self.materials.resize(materials.len(), MaterialConstants1D::default());
        for (mat_consts, mat) in self.materials.iter_mut().zip(materials) {
            *mat_consts = MaterialConstants1D::new(mat.eps_r, mat.mu_r, dt);
        }

        // Update grid dimensions & initialize cells
        // Object positions & widths are considered relative to each other. They don't consider
        // things like spacer regions.
        let min_pos = self.objects.iter()
            .map(|o| o.position - o.width/2.)
            .chain(self.sources.iter().map(|s| s.location))
            .min_by(|a, b| a.total_cmp(b))
            .unwrap_or(0.);
        let max_pos = self.objects.iter()
            .map(|o| o.position + o.width/2.)
            .max_by(|a, b| a.total_cmp(b))
            .unwrap_or(dz);
        let dimensions = dz * 2. + // Transmittance/Reflectance cells & source cells
            dz * stability.spacer_region_cells as f32 * 2. + // Both spacer regions
            dz * ((max_pos - min_pos)/dz).abs().ceil(); // Account for objects & sources
        self.grid_info.set_dimensions(dimensions);
        self.cells.resize(self.grid_info.num_cells as usize, GridCell1D::default());

        // Set grid cell material indices
        let offset = stability.spacer_region_cells + 2;
        self.object_cell_indices = vec![0..0; self.objects.len()];
        for (obj_i, obj) in self.objects.iter().enumerate() {
            let obj_pos = obj.position - min_pos; // localize object positions
            let pos_idx = (obj_pos / dz).round() as usize;
            let cells_width = (obj.width / dz).round() as usize;
            let width_half1 = cells_width / 2;
            let width_half2 = cells_width.div_ceil(2);
            let middle = offset + pos_idx;

            let start = middle - width_half1;
            let end = middle + width_half2;
            let cells_range = start..end;
            for i in cells_range.clone() {
                self.cells[i].material_idx = obj_material_idxs[obj_i] as u32;
            }
            self.object_cell_indices[obj_i] = cells_range;
        }

        // Prepare sources
        let mut new_sources = Vec::with_capacity(self.sources.len());
        let mut new_src_vals = Vec::with_capacity(self.source_vals.len());
        for src in self.sources.iter() {
            let vals = src.compute_source_values(self);

            let start_idx = new_src_vals.len() as u32;
            new_src_vals.extend_from_slice(&vals);
            let end_idx = new_src_vals.len() as u32 - 1;
            let cell_idx = 2; // Sources are always located in the first spacer region's 1st cell

            new_sources.push(GpuSource1D {
                start_idx,
                end_idx,
                curr_idx: 0,
                cell_idx,
            });
        }
        self.sources_gpu = new_sources;
        self.source_vals = new_src_vals;

        self
    }

    pub fn min_wavelength(&mut self, f_max: f32, cells_per_wavelength: usize) -> &mut Self {
        let n_max = self.materials.iter()
            .map(|m| m.n)
            .max_by(|a, b| a.total_cmp(b))
            .unwrap_or(1.);
        let min_wavelen = MaterialConstants1D::C_0 / (f_max * n_max);
        self.grid_info.cell_size = self.grid_info.cell_size
            .min(min_wavelen / cells_per_wavelength as f32);
        self.update_cell_count()
    }

    /// Sets up `dt` for simulating with a perfect boundary condition.
    ///
    /// Guarantees that the fastest wave in the simulation travels 1 grid cell in exactly
    /// two timesteps.
    pub fn set_cfl_perfect_boundary(&mut self, n_boundary:f32) -> &mut Self {
        self.grid_info.dt = self.compute_cfl_upper_bound(n_boundary, 2.);
        self
    }

    pub fn compute_cfl_upper_bound(&self, n_min: f32, safety_margin: f32) -> f32 {
        (n_min * self.grid_info.cell_size) / (safety_margin * MaterialConstants1D::C_0)
    }

    pub fn enforce_stability_conditions(&mut self, stability: &StabilityValues, n_boundary: f32) -> &mut Self {
        let f_max = self.sources.iter()
            .map(|g| 0.5 / g.tau)
            .max_by(|a, b| a.total_cmp(b))
            .expect("There must be at least one source!");
        self.min_wavelength(f_max, stability.cells_per_wavelength);
        self.set_cfl_perfect_boundary(n_boundary)
    }

    pub fn set_dimensions(&mut self, dimensions: f32) -> &mut Self {
        self.grid_info.dimensions = dimensions;
        self.update_cell_count()
    }

    pub fn update_cell_count(&mut self) -> &mut Self {
        self.grid_info.num_cells = (self.grid_info.dimensions / self.grid_info.cell_size).ceil() as u32;
        self
    }

    pub fn create_buffers(&self, backend: &GpuBackend) -> Result<Fdtd1dBuffers, GpuBackendError> {
        Ok(Fdtd1dBuffers {
            cells: self.cells.create_gpu_buffer_readable(backend)?,
            materials: self.materials.create_gpu_buffer(backend)?,
            source_vals: self.source_vals.create_gpu_buffer(backend)?,
            sources: self.sources_gpu.create_gpu_buffer(backend)?,
            perfect_boundary_data: PerfectBoundaryData::default()
                .create_gpu_buffer_readable(backend)?,
            grid_info: self.grid_info.create_gpu_uniform(backend)?,
        })
    }

    pub fn add_object(&mut self, obj: ObjectInfo1D) -> &mut Self {
        self.objects.push(obj);
        self
    }

    pub fn add_source(&mut self, src: GaussianPulse1D) -> &mut Self {
        self.sources.push(src);
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

pub struct StabilityValues {
    /// The number of cells that should be within the smallest wavelength in the simulation.
    ///
    /// Usually `cells_per_wavelength >= 10` gives good stability. Default is `20`.
    pub cells_per_wavelength: usize,
    /// Number of "empty" grid cells on either side of the object. Usually `10` cells is enough.
    pub spacer_region_cells: usize,
}

impl Default for StabilityValues {
    fn default() -> Self {
        Self {
            cells_per_wavelength: 20,
            spacer_region_cells: 10
        }
    }
}

#[derive(Default)]
pub struct ObjectInfo1D {
    pub width: f32,
    pub position: f32,
    pub material: ElectricMaterial,
    pub color: Color,
}

#[derive(Copy, Clone, PartialEq)]
pub struct ElectricMaterial {
    pub eps_r: f32,
    pub mu_r: f32,
    pub n: f32,
}

impl ElectricMaterial {
    pub const FREE_SPACE: Self = Self {
        eps_r: 1.,
        mu_r: 1.,
        n: 1.,
    };

    pub fn new(eps_r: f32, mu_r: f32) -> Self {
        Self {
            eps_r,
            mu_r,
            n: (eps_r*mu_r).sqrt()
        }
    }
}

impl Default for ElectricMaterial {
    fn default() -> Self { Self::FREE_SPACE }
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

    pub fn compute_source_values(&self, sim_data: &Fdtd1dData) -> Vec<f32> {
        let approx_pulse_duration = 12. * self.tau;
        let num_vals = (approx_pulse_duration / sim_data.grid_info.dt).ceil() as u32;
        let mut vals = vec![0.; num_vals as usize];

        let mut t = 0.;
        for i in 0..vals.len() {
            t += sim_data.grid_info.dt;
            let g = core::f32::consts::E.powf(
                -((t - self.t_0) / self.tau).powi(2)
            );
            vals[i] = g * self.amplitude;
        }

        vals
    }
}