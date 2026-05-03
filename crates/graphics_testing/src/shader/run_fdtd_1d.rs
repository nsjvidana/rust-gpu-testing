use crate::util::{CreateGpuBuffer, CreateGpuBufferReadable, GpuBufferReadable};
use egui_plot::{Legend, Line, Plot, PlotPoint, PlotPoints};
use glam::Vec3;
use khal::backend::{Backend, Buffer, DispatchGrid, Encoder, GpuBackend, GpuBackendError, GpuBuffer};
use khal::Shader;
use kiss3d::camera::OrbitCamera3d;
use kiss3d::egui;
use kiss3d::event::{Action, Key};
use kiss3d::light::Light;
use kiss3d::prelude::{Color, SceneNode3d, Window, GREEN, RED, WHITE};
use shader_crate::fdtd_1d::{ComputeFftKernels1d, Fdtd1d, Fft1d, FftDataGPU, FinishFft1d, GpuSource1D, GridCell1D, GridInfo1D, MaterialConstants1D, PerfectBoundaryData};
use shader_crate::GpuComplexPolar;
use std::ops::{Range, RangeInclusive};

#[derive(Shader)]
struct GpuKernels {
    pub fdtd_1d: Fdtd1d,
    pub compute_fft_kernels: ComputeFftKernels1d,
    pub fft: Fft1d,
    pub finish_fft: FinishFft1d
}

pub async fn run_fdtd_1d(backend: &GpuBackend) {
    let max_pulse_freq = 1e9;

    let obj_width = 0.3048; // 1ft wide
    let eps_r_mat = 6.0_f32;
    let mu_r_mat = 2.0_f32;

    let stability_values = StabilityValues::default();

    let obj = ObjectInfo1D {
        width: obj_width,
        position: 0.,
        material: ElectricMaterial::new(eps_r_mat, mu_r_mat),
        color: GREEN
    };

    let source = GaussianPulse1D::from_max_frequency(max_pulse_freq, 1.);

    let mut data = Fdtd1dData::new();
    data
        .add_object(obj)
        .set_source(source);

    data.prepare_for_gpu(&stability_values, None);

    let mut f_res = data.estimate_max_timesteps(None);
        if f_res % 2 == 0 { f_res -= 1; }
    let f_max = data.compute_max_frequency();
    data.enable_ffts((-f_max)..=f_max, f_res);

    data.set_step_count(1);
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

    let gpu_kernels = GpuKernels::from_backend(&backend)?;
    let mut buffers = data.create_buffers(backend)?;

    let src = &data.source_gpu;
    let pos = Vec3::Z * src.cell_idx as f32 * grid_info.cell_size;
    scene.add_sphere(grid_info.cell_size / 5.)
        .translate(pos)
        .set_color(RED);

    // Compute FFT kernels before running simulation
    if let Some(fft_bufs) = &mut buffers.fft_buffers {
        let mut encoder = backend.begin_encoding();
        let mut pass = encoder.begin_pass("", None);
        let dispatch_count = fft_bufs.fft_kernels.len().div_ceil(64) as u32;
        gpu_kernels.compute_fft_kernels.call(
            &mut pass,
            DispatchGrid::Grid([dispatch_count, 1, 1]),
            &mut fft_bufs.fft_kernels,
            &mut fft_bufs.fft_data,
            &buffers.grid_info
        )?;
        drop(pass);
        backend.submit(encoder)?;
    }

    let mut abs_max_val = 0.;
    let mut cells_out = vec![GridCell1D::default(); grid_info.num_cells as usize];
    let mut boundary_out = vec![PerfectBoundaryData::default()];
    let mut fft_out = data.fft_data.as_ref().map(|fft| Ffts::new(fft.resolution as _));
    let mut prev_action = Action::Release;
    let mut fft_plot = fft_out.as_ref().map(|f| FftPlot::new(&f));
    while window.render_3d(&mut scene, &mut camera).await {
        let curr_action = window.get_key(Key::T);
        if window.get_key(Key::LControl) != Action::Press {
            prev_action = Action::Release;
        }
        if curr_action != prev_action && curr_action == Action::Press {
            backend.synchronize()?;
            buffers.cells.read(backend, &mut cells_out).await?;
            buffers.perfect_boundary_data.read(backend, &mut boundary_out).await?;
            if let Some(fft_out) = fft_out.as_mut() {
                let bufs = buffers.fft_buffers.as_ref().unwrap();
                bufs.reflectance_fft.read(backend, &mut fft_out.reflectance).await?;
                bufs.transmittance_fft.read(backend, &mut fft_out.transmittance).await?;
                bufs.source_fft.read(backend, &mut fft_out.source).await?;
            }

            submit_simulation(
                backend,
                &gpu_kernels,
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

        if let Some(fft_out) = &fft_out {
            window.draw_ui(|ctx| {
                egui::Window::new("Reflectance and Transmittance").show(ctx, |ui| {
                    fft_plot.as_mut().unwrap()
                        .fft_ui(fft_out, data.fft_data.as_ref().unwrap(), ui);
                });
            });
        }
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

    // Compute pass
    let mut pass = encoder.begin_pass("", None);
    for _ in 0..grid_info.steps_per_call {
        kernels.fdtd_1d.call(
            &mut pass,
            DispatchGrid::Grid([grid_info.num_cells.div_ceil(64), 1, 1]),
            &mut buffers.cells.buffer,
            &mut buffers.materials,
            &buffers.source_vals,
            &mut buffers.source,
            &mut buffers.perfect_boundary_data.buffer,
            &mut buffers.timestep_counter,
            &buffers.grid_info
        )?;

        if let Some(FftBuffers {
            fft_kernels,
            reflectance_fft,
            transmittance_fft,
            source_fft,
            ..
        }) = &mut buffers.fft_buffers {
            let dispatch_count = fft_kernels.len().div_ceil(64) as u32;
            kernels.fft.call(
                &mut pass,
                DispatchGrid::Grid([dispatch_count, 1, 1]),
                &buffers.cells.buffer,
                &mut reflectance_fft.buffer,
                &mut transmittance_fft.buffer,
                &mut source_fft.buffer,
                &buffers.source,
                fft_kernels,
                &buffers.timestep_counter
            )?;
            kernels.finish_fft.call(
                &mut pass,
                DispatchGrid::Grid([dispatch_count, 1, 1]),
                &mut reflectance_fft.buffer,
                &mut transmittance_fft.buffer,
                &mut source_fft.buffer,
                &buffers.grid_info
            )?;
        }
    }
    drop(pass);

    buffers.cells.encode_copy_cmd(&mut encoder)?;
    buffers.perfect_boundary_data.encode_copy_cmd(&mut encoder)?;
    if let Some(fft) = &mut buffers.fft_buffers {
        fft.reflectance_fft.encode_copy_cmd(&mut encoder)?;
        fft.transmittance_fft.encode_copy_cmd(&mut encoder)?;
        fft.source_fft.encode_copy_cmd(&mut encoder)?;
    }

    backend.submit(encoder)
}

pub struct FftPlot {
    pub reflectance: Vec<PlotPoint>,
    pub transmittance: Vec<PlotPoint>,
    pub prev_pointer_coords: Option<PlotPoint>,
}

impl FftPlot {
    fn new(ffts: &Ffts) -> Self {
        let vals_count = ffts.reflectance.len();
        let fft_vals = vec![PlotPoint::new(0, 0); vals_count];
        Self {
            transmittance: fft_vals.clone(),
            reflectance: fft_vals,
            prev_pointer_coords: None,
        }
    }
    fn fft_ui(&mut self, ffts: &Ffts, fft_data: &FftData, ui: &mut egui::Ui) {
        let f_start = *fft_data.frequency_range.start() as f64;
        let f_end = *fft_data.frequency_range.end() as f64;
        let f_incr = fft_data.f_increment as f64;
        let coords_txt = self.prev_pointer_coords
            .map(|p| format!("x: {}, y: {}", p.x, p.y))
            .unwrap_or("None".to_string());
        ui.label(format!("Pointer Coords: {coords_txt}"));
        Plot::new("FFTs")
            .legend(Legend::default())
            .show(ui, |plot| {
                plot.set_plot_bounds_x(f_start..=f_end);
                self.prev_pointer_coords = plot.pointer_coordinate();

                for (i, fft) in ffts.reflectance.iter().enumerate() {
                    let src = ffts.source[i].r + (ffts.source[i].r == 0.) as u32 as f32;
                    let a = (fft.r / src).powi(2) as f64;
                    self.reflectance[i] = PlotPoint::new(f_start + f_incr * i as f64, a);
                }
                for (i, fft) in ffts.transmittance.iter().enumerate() {
                    let src = ffts.source[i].r + (ffts.source[i].r == 0.) as u32 as f32;
                    let a = (fft.r / src).powi(2) as f64;
                    self.transmittance[i] = PlotPoint::new(f_start + f_incr * i as f64, a);
                }
                plot.line(Line::new("Reflectance", PlotPoints::Borrowed(&self.reflectance)));
                plot.line(Line::new("Transmittance", PlotPoints::Borrowed(&self.transmittance)));
            });
    }
}

pub struct Ffts {
    pub reflectance: Vec<GpuComplexPolar>,
    pub transmittance: Vec<GpuComplexPolar>,
    pub source: Vec<GpuComplexPolar>,
}

impl Ffts {
    pub fn new(resolution: usize) -> Self {
        Self {
            reflectance: vec![GpuComplexPolar::default(); resolution],
            transmittance: vec![GpuComplexPolar::default(); resolution],
            source: vec![GpuComplexPolar::default(); resolution],
        }
    }
}

#[derive(Default)]
pub struct Fdtd1dData {
    pub cells: Vec<GridCell1D>,
    pub materials: Vec<MaterialConstants1D>,
    pub source: GaussianPulse1D,
    pub source_vals: Vec<f32>,
    pub source_gpu: GpuSource1D,
    pub grid_info: GridInfo1D,
    /// If this has a value, FFTs are enabled.
    pub fft_data: Option<FftData>,

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

    /// Call this to enable reflectance and transmittance FFTs.
    ///
    /// The maximum number of timesteps must be known to call this function.
    pub fn enable_ffts(&mut self, frequency_range: RangeInclusive<f32>, resolution: u32) -> &mut Self {
        self.fft_data = Some(FftData::new(frequency_range, resolution));
        self
    }

    /// Estimates the number of timesteps the simulation needs to be considered "finished."
    /// Does NOT consider the resonance of objects in the simulations.
    ///
    /// You can use this function's output as FFT resolution
    pub fn estimate_max_timesteps(&mut self, custom_default_material: Option<ElectricMaterial>) -> u32 {
        let default_mat = custom_default_material.unwrap_or(ElectricMaterial::FREE_SPACE);
        let mut n_max = self.objects.iter()
            .map(|o| o.material.n)
            .max_by(|a, b| a.total_cmp(b))
            .unwrap_or(1.);
        n_max = n_max.max(default_mat.n);

        let max_src_duration = self.source.tau * 12.;
        // time it takes to the slowest wave to propagate across the grid (a worst-case scenario)
        let t_prop = n_max/MaterialConstants1D::C_0 * self.grid_info.num_cells as f32;
        ((max_src_duration + t_prop) / self.grid_info.dt).ceil() as u32
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
            .min_by(|a, b| a.total_cmp(b))
            .unwrap_or(0.);
        let max_pos = self.objects.iter()
            .map(|o| o.position + o.width/2.)
            .max_by(|a, b| a.total_cmp(b))
            .unwrap_or(dz);
        let dimensions = dz * 2. + // Transmittance/Reflectance cells & source cells
            dz * stability.spacer_region_cells as f32 * 2. + // Both spacer regions
            dz * ((max_pos - min_pos)/dz).abs().ceil(); // Account for objects & sources
        self.set_dimensions(dimensions);
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

        // Prepare source
        self.source_vals.clear();
        let src = &self.source;
        let vals = src.compute_source_values(self);
        let start_idx = self.source_vals.len() as u32;
        self.source_vals.extend_from_slice(&vals);
        let end_idx = self.source_vals.len() as u32 - 1;
        let cell_idx = 2; // Sources are always located in the first spacer region's 1st cell
        self.source_gpu = GpuSource1D {
            start_idx,
            end_idx,
            curr_idx: 0,
            cell_idx,
        };

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

    pub fn min_feature_length(&mut self, min_feature_length: f32, cells_per_min_len: u32) -> &mut Self {
        self.grid_info.cell_size = self.grid_info.cell_size.min(min_feature_length / cells_per_min_len as f32);
        self.update_cell_count()
    }

    pub fn snap_to_critical_dim(&mut self, critical_dim: f32) -> &mut Self {
        let cells_per_crit_dim = (critical_dim / self.grid_info.cell_size).ceil();
        self.grid_info.cell_size = critical_dim / cells_per_crit_dim;
        self.update_cell_count()
    }

    /// Sets up `dt` for simulating with a Perfect Absorbing Boundary.
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
        let f_max = self.compute_max_frequency();
        self.min_wavelength(f_max, stability.cells_per_wavelength)
            .set_cfl_perfect_boundary(n_boundary);
        self
    }

    pub fn set_dimensions(&mut self, dimensions: f32) -> &mut Self {
        self.grid_info.dimensions = dimensions;
        self.update_cell_count()
    }

    /// Set the amount of `dt` time steps per shader dispatch.
    pub fn set_step_count(&mut self, step_count: u32) -> &mut Self {
        self.grid_info.steps_per_call = step_count;
        self
    }

    pub fn update_cell_count(&mut self) -> &mut Self {
        self.grid_info.num_cells = (self.grid_info.dimensions / self.grid_info.cell_size).ceil() as u32;
        self
    }

    pub fn create_buffers(&self, backend: &GpuBackend) -> Result<Fdtd1dBuffers, GpuBackendError> {
        let mut fft_buffers = None;
        if let Some(fft) = &self.fft_data {
            let kernel_count = fft.resolution as usize;
            let init_fft_kernels = vec![GpuComplexPolar::default(); kernel_count];
            let init_fft_data = vec![GpuComplexPolar::default(); kernel_count];

            let f_start = *fft.frequency_range.start();
            fft_buffers = Some(
                FftBuffers {
                    fft_data: FftDataGPU {
                        f_start,
                        f_increment: fft.f_increment
                    }.create_gpu_uniform(backend)?,
                    fft_kernels: init_fft_kernels.create_gpu_buffer(backend)?,
                    reflectance_fft: init_fft_data.create_gpu_buffer_readable(backend)?,
                    transmittance_fft: init_fft_data.create_gpu_buffer_readable(backend)?,
                    source_fft: init_fft_data.create_gpu_buffer_readable(backend)?,
                }
            )
        }
        let timestep_counter = 0;

        Ok(Fdtd1dBuffers {
            cells: self.cells.create_gpu_buffer_readable(backend)?,
            materials: self.materials.create_gpu_buffer(backend)?,
            source_vals: self.source_vals.create_gpu_buffer(backend)?,
            source: self.source_gpu.create_gpu_buffer(backend)?,
            perfect_boundary_data: PerfectBoundaryData::default()
                .create_gpu_buffer_readable(backend)?,
            timestep_counter: timestep_counter.create_gpu_buffer(backend)?,
            grid_info: self.grid_info.create_gpu_uniform(backend)?,
            fft_buffers
        })
    }

    pub fn add_object(&mut self, obj: ObjectInfo1D) -> &mut Self {
        self.objects.push(obj);
        self
    }

    pub fn set_source(&mut self, src: GaussianPulse1D) -> &mut Self {
        self.source = src;
        self
    }

    /// Returns starting cell index and the width of the object in grid cells: `(start_idx, idx_width)`
    pub fn get_obj_indices(&self, obj: &ObjectInfo1D) -> (usize, usize) {
        let center_idx = (obj.position / self.grid_info.cell_size) as usize;
        let idx_width = (obj.width / self.grid_info.cell_size).ceil() as usize;
        let start = (center_idx - idx_width.div_ceil(2)).max(0);
        (start, idx_width)
    }

    pub fn compute_max_frequency(&self) -> f32 {
        0.5 / self.source.tau
    }
}

pub struct FftData {
    pub frequency_range: RangeInclusive<f32>,
    pub resolution: u32,
    pub f_increment: f32
}

impl FftData {
    pub fn new(frequency_range: RangeInclusive<f32>, resolution: u32) -> Self {
        Self {
            f_increment: Self::compute_f_increment(&frequency_range, resolution),
            frequency_range,
            resolution,
        }
    }

    pub fn compute_f_increment(f_range: &RangeInclusive<f32>, resolution: u32) -> f32 {
        (f_range.end() - f_range.start()) / (resolution as f32 - 1.)
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
    pub source: GpuBuffer<GpuSource1D>,
    pub perfect_boundary_data: GpuBufferReadable<PerfectBoundaryData>,
    pub grid_info: GpuBuffer<GridInfo1D>,
    pub timestep_counter: GpuBuffer<u32>,

    pub fft_buffers: Option<FftBuffers>
}

pub struct FftBuffers {
    pub fft_data: GpuBuffer<FftDataGPU>,
    pub fft_kernels: GpuBuffer<GpuComplexPolar>,
    pub reflectance_fft: GpuBufferReadable<GpuComplexPolar>,
    pub transmittance_fft: GpuBufferReadable<GpuComplexPolar>,
    pub source_fft: GpuBufferReadable<GpuComplexPolar>,
}

#[derive(Debug)]
pub struct GaussianPulse1D {
    pub amplitude: f32,
    pub tau: f32,
    pub t_0: f32,
}

impl GaussianPulse1D {
    /// Create a Gaussian Pulse that has a maximum frequency of `max_frequency`
    pub fn from_max_frequency(
        max_frequency: f32,
        amplitude: f32,
    ) -> Self {
        let tau = 0.5 / max_frequency;

        Self {
            amplitude,
            tau,
            t_0: 6. * tau,
        }
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

impl Default for GaussianPulse1D {
    fn default() -> Self {
        Self {
            amplitude: 0.,
            tau: 1.,
            t_0: 0.,
        }
    }
}