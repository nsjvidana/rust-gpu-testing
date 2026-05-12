use crate::prelude::GpuResult;
use crate::util::{bb_polyline, CreateGpuBuffer, CreateGpuBufferReadable, GpuBufferReadable};
use glam::{USizeVec3, UVec2, UVec3, Vec2};
use khal::backend::{Backend, DispatchGrid, Encoder, GpuBackend, GpuBuffer};
use khal::Shader;
use kiss3d::egui;
use kiss3d::prelude::*;
use shader_crate::fdtd2::{Fdtd2, GpuSource2, GridCell2, GridInfo2, MaterialConstants2, SoftSource2};
use shader_crate::{flat_idx_to_vector, vector_to_flat_idx};

#[derive(Shader)]
struct GpuKernels2 {
    fdtd2: Fdtd2,
    soft_source2: SoftSource2,
}

pub async fn run_fdtd2(backend: &GpuBackend) -> GpuResult<()> {
    let gpu_kernels = GpuKernels2::from_backend(backend)?;
    let mut data = FdtdData2::new();

    let pulse_freq = 1e6;
    let pulse_amplitude = 1.;
    let pulse = GaussianPulse2::from_max_frequency(pulse_freq, pulse_amplitude);

    data.min_wavelength(pulse_freq, 20)
        .cfl_condition(3.);
    data.grid.n_cells = UVec2::new(30, 30);
    data.grid.cells.resize(data.grid.n_cells.element_product() as usize, GridCell2::default());
    data.set_source(pulse, 10);

    println!("dt: {:?}", data.dt);
    println!("cell_size: {:?}", data.grid.cell_size);
    let mut runner = data.create_gpu(1, backend)?;

    // Set up window
    let mut window = Window::new("FDTD 2D").await;
    let grid_dims = data.grid.cell_size * data.grid.n_cells.as_vec2();
    let z_far = (data.grid.n_cells.as_vec2() * data.grid.cell_size).max_element() * 10.;
    let mut camera = OrbitCamera3d::new_with_frustum(
        core::f32::consts::PI / 4.0, data.grid.cell_size.min_element(), z_far,
        Vec3::splat(grid_dims.max_element()),
        Vec3::from((grid_dims / 2., 0.))
    );
        camera.set_up_axis_dir(Vec3::Z);
    let mut scene = SceneNode3d::empty();
    scene
        .add_light(Light::point(100.0))
        .set_position(Vec3::new(0.0, 2.0, -2.0));
    let mut render_data = RenderData2::new(&data, 0.01);
    let mut max_en_magnitude = 0.;
    // Main render loop
    while window.render_3d(&mut scene, &mut camera).await {
        if window.get_key(Key::T) == Action::Press {
            backend.synchronize()?;
            runner.cells.read(backend, &mut data.grid.cells).await?;
            runner.submit_step(&gpu_kernels, backend)?;

            let curr_max_en_mag = data.grid.cells.iter()
                .map(|c| c.en_z)
                .max_by(|a, b| a.total_cmp(b))
                .unwrap();
            if curr_max_en_mag > max_en_magnitude {
                println!("New max En magnitude: {}", curr_max_en_mag);
                max_en_magnitude = curr_max_en_mag;
            }
        }
        
        render_data.render_simulation(&mut window, &data, max_en_magnitude);
        render_data.egui_window(&mut window);
    }

    Ok(())
}

pub struct RenderData2 {
    pub cell_positions: Vec<Vec3>,
    pub en_color: Color,
    pub grid_bb: Polyline3d,
    pub alpha_threshold: f32,
}

impl RenderData2 {
    pub fn new(data: &FdtdData2, alpha_threshold: f32) -> Self {
        let grid = &data.grid;
        let en_color = RED;
        let n_cells3 = USizeVec3::from((grid.n_cells.as_usizevec2(), 1));
        let cell_size3 = Vec3::from((grid.cell_size, 0.));

        let grid_bb = bb_polyline(
            Vec3::from(cell_size3 * n_cells3.as_vec3()),
            Vec3::ZERO
        );

        Self {
            cell_positions: (0..grid.cells.len())
                .map(|i| {
                    let i3 = flat_idx_to_vector!(i, n_cells3, USizeVec3);
                    i3.as_vec3() * cell_size3
                })
                .collect(),
            grid_bb,
            en_color,
            alpha_threshold
        }
    }

    pub fn render_simulation(
        &mut self,
        window: &mut Window,
        data: &FdtdData2,
        max_en_value: f32,
    ) {
        let cell_diagonal_len = data.grid.cell_size.length();
        for (c, pos) in data.grid.cells.iter()
            .zip(self.cell_positions.iter())
        {
            // Drawing En field
            let alpha = c.en_z.abs() / max_en_value;
            if alpha > self.alpha_threshold {
                let color = self.en_color.with_alpha(alpha);
                let line_len = alpha * cell_diagonal_len * c.en_z.signum();
                window.draw_line(*pos, pos + Vec3::new(0., 0., line_len), color, 2., false);
            }
        }

        window.draw_polyline(&self.grid_bb);
    }

    pub fn egui_window(&mut self, window: &mut Window) {
        window.draw_ui(|ctx| {
            egui::Window::new("Import mesh")
                .show(ctx, |ui| self.ui(ui));
        });
    }

    pub fn ui(&mut self, ui: &mut egui::Ui) {
        ui.label("Hi");
    }
}

pub struct FdtdData2 {
    pub dt: f32,
    pub grid: FdtdGrid2,
    pub materials: Vec<ElectricMaterial2>,
    pub source: GaussianPulse2,
}

impl FdtdData2 {
    pub fn new() -> Self {
        Self {
            dt: f32::MAX,
            grid: FdtdGrid2::new(),
            materials: vec![],
            source: GaussianPulse2::default()
        }
    }

    pub fn min_wavelength(&mut self, f_max: f32, cells_per_wavelength: usize) -> &mut Self {
        let n_max = self.materials.iter()
            .max_by(|m1, m2| m1.n.total_cmp(&m2.n))
            .map(|m| m.n)
            .unwrap_or(1.);
        let min_wavelen = ElectricMaterial2::C_0 / (f_max * n_max);
        self.grid.cell_size = self.grid.cell_size
            .min(Vec2::splat(min_wavelen / cells_per_wavelength as f32));
        self
    }

    /// Enforce the Courant–Friedrichs–Lewy stability condition
    pub fn cfl_condition(&mut self, safety_margin: f32) -> &mut Self {
        let denom_term_2 = self.grid.cell_size
            .recip()
            .powf(2.)
            .element_sum()
            .sqrt();
        let safety_margin = safety_margin.max(1.);
        self.dt = self.dt.min(
            1. / (ElectricMaterial2::C_0 * denom_term_2 * safety_margin)
        );
        self
    }

    /// Computes the absolute maximum frequency the simulation can resolve
    /// according to the Nyquist-Shannon Sampling Theorem
    pub fn compute_max_frequency(&mut self) -> f32 {
        0.5 / self.dt
    }

    /// Minimum number of steps needed to resolve frequency down to a resolution of `df`.
    /// The smaller `df` is, the less "blurred" the simulated frequency response is.
    ///
    /// Useful for getting proper Fourier Transform results
    pub fn steps_for_df(&self, df: f32) -> u32 {
        (1. / (self.dt * df)).round() as u32
    }

    pub fn prepare_materials(&mut self) -> &mut Self {
        if self.materials.is_empty() {
            self.materials.push(ElectricMaterial2::FREE_SPACE);
        }
        // TODO: include object materials
        self
    }

    /// Set the `GaussianPulse2` source for this simulation.
    ///
    /// `resolution` should be at least `10` to `20` for better results
    pub fn set_source(&mut self, pulse: GaussianPulse2, resolution: u32) -> &mut Self {
        self.dt = self.dt.min(pulse.tau / resolution as f32);
        self.source = pulse;
        self
    }

    pub fn create_gpu(&mut self, steps_per_submission: usize, backend: &GpuBackend) -> GpuResult<GpuFdtd2> {
        let n_cells3 = UVec3::from((self.grid.n_cells, 1));
        self.prepare_materials();

        let step_counter = 0;

        let gpu_fdtd = GpuFdtd2 {
            cells: self.grid.cells.create_gpu_buffer_readable(backend)?,
            grid_info: GridInfo2 {
                n_cells: self.grid.n_cells,
                cell_size: self.grid.cell_size,
                i_incr: UVec2::new(
                    vector_to_flat_idx!(UVec3::X, n_cells3),
                    vector_to_flat_idx!(UVec3::Y, n_cells3),
                ),
                dn_z_update_coeff: ElectricMaterial2::C_0 * self.dt,
                _padding: 0
            }.create_gpu_uniform(backend)?,
            materials: self.materials.iter()
                .map(|m| m.to_gpu(self.dt))
                .collect::<Vec<_>>()
                .create_gpu_buffer(backend)?,
            source: GpuSource2 {
                cell_idx: vector_to_flat_idx!(n_cells3 / 2, n_cells3),
            }.create_gpu_buffer(backend)?,
            source_vals: self.source.compute_source_values(self.dt)
                .create_gpu_buffer(backend)?,
            step_counter: step_counter.create_gpu_buffer(backend)?,

            dispatch_grid: n_cells3.map(|v| v.div_ceil(8)).to_array(),
            steps_per_submission,
        };

        Ok(gpu_fdtd)
    }
}

pub struct GpuFdtd2 {
    pub grid_info: GpuBuffer<GridInfo2>,
    pub cells: GpuBufferReadable<GridCell2>,
    pub materials: GpuBuffer<MaterialConstants2>,
    pub source: GpuBuffer<GpuSource2>,
    pub source_vals: GpuBuffer<f32>,
    pub step_counter: GpuBuffer<u32>,

    pub dispatch_grid: [u32; 3],
    pub steps_per_submission: usize,
}

impl GpuFdtd2 {
    pub fn submit_step(&mut self, gpu_kernels: &GpuKernels2, backend: &GpuBackend) -> GpuResult<()> {
        let mut encoder = backend.begin_encoding();

        let mut pass = encoder.begin_pass("fdtd2", None);
        for _ in 0..self.steps_per_submission {
            gpu_kernels.fdtd2.call(
                &mut pass,
                DispatchGrid::Grid(self.dispatch_grid),
                &mut self.cells.buffer,
                &self.materials,
                &self.grid_info
            )?;
            gpu_kernels.soft_source2.call(
                &mut pass,
                DispatchGrid::Grid([1, 1, 1]),
                &mut self.cells.buffer,
                &self.source,
                &self.source_vals,
                &mut self.step_counter
            )?;
        }
        drop(pass);

        self.cells.encode_copy_cmd(&mut encoder)?;

        backend.submit(encoder)?;
        Ok(())
    }
}

pub struct FdtdGrid2 {
    pub cells: Vec<GridCell2>,
    pub cell_size: Vec2,
    /// Number of cells for each axis (the dimensions of the grid in `cell_size` units).
    pub n_cells: UVec2,
}

impl FdtdGrid2 {
    pub fn new() -> Self {
        Self {
            cells: vec![],
            cell_size: Vec2::MAX,
            n_cells: UVec2::new(0, 0),
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct ElectricMaterial2 {
    /// Relative Magnetic Permeability (X & Y component of tensor diagonal)
    pub mu_r: Vec2,
    /// Relative Electric Permittivity (Z component of tensor diagonal)
    pub eps_r_z: f32,
    /// Refractive Index
    /// Impedance
    pub n: f32,
    pub impedance: f32,
}

impl ElectricMaterial2 {
    /// Speed of EM wave in free space
    pub const C_0: f32 = 299792458.0;
    pub const EPS_0: f32 = 8.8541878188e-12;
    pub const MU_0: f32 = 1.25663706127e-6;
    pub const IMPEDANCE_0: f32 = 376.730313412;
    pub const FREE_SPACE: Self = Self {
        eps_r_z: 1.,
        mu_r: Vec2::ONE,
        n: 1.,
        impedance: Self::IMPEDANCE_0,
    };

    pub fn new_linear(eps_r_z: f32, mu_r: f32) -> Self {
        Self {
            mu_r: Vec2::splat(mu_r),
            eps_r_z,
            n: (eps_r_z * mu_r).sqrt(),
            impedance: f32::sqrt((Self::MU_0 * mu_r) / (Self::EPS_0 * eps_r_z)),
        }
    }

    pub fn to_gpu(self, dt: f32) -> MaterialConstants2 {
        let c_0_dt = Self::C_0 * dt;
        MaterialConstants2 {
            h_update_coeff: Vec2::new(
                -c_0_dt / self.mu_r.x,
                -c_0_dt / self.mu_r.y,
            ),
            en_z_update_coeff: 1. / self.eps_r_z,
            ..Default::default()
        }
    }
}

#[derive(Debug)]
pub struct GaussianPulse2 {
    pub amplitude: f32,
    pub tau: f32,
    pub t_0: f32,
}

impl GaussianPulse2 {
    /// Create a Gaussian Pulse that has a maximum frequency of `max_frequency`
    pub fn from_max_frequency(
        max_frequency: f32,
        amplitude: f32,
    ) -> Self {
        let tau = core::f32::consts::FRAC_1_PI / max_frequency;

        Self {
            amplitude,
            tau,
            t_0: 6. * tau,
        }
    }

    pub fn compute_source_values(&self, dt: f32) -> Vec<f32> {
        let approx_pulse_duration = 12. * self.tau;
        let num_vals = (approx_pulse_duration / dt).ceil() as u32;
        let mut vals = vec![0.; num_vals as usize];

        let mut t = 0.;
        for i in 0..vals.len() {
            t += dt;
            let g = core::f32::consts::E.powf(
                -((t - self.t_0) / self.tau).powi(2)
            );
            vals[i] = g * self.amplitude;
        }

        vals
    }
}

impl Default for GaussianPulse2 {
    fn default() -> Self {
        Self {
            amplitude: 0.,
            tau: 1e-6,
            t_0: 0.,
        }
    }
}