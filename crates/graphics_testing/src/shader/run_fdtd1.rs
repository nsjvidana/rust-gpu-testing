use crate::prelude::GpuResult;
use crate::util::{CreateGpuBuffer, CreateGpuBufferReadable, GpuBufferReadable};
use egui_plot::{Legend, Line, Plot, PlotPoint, PlotPoints};
use khal::backend::{Backend, Buffer, DispatchGrid, Encoder, GpuBackend, GpuBackendError, GpuBuffer};
use khal::Shader;
use kiss3d::camera::OrbitCamera3d;
use kiss3d::egui;
use kiss3d::event::{Action, Key};
use kiss3d::light::Light;
use kiss3d::prelude::{Color, SceneNode3d, Window, GREEN, RED, WHITE};
use rayon::prelude::*;
use shader_crate::fdtd1::{Dft1, DftInfo1, Fdtd1, FinishDft1, GpuSource1, GridCell1, GridInfo1, MaterialConstants1, PerfectBoundaryData, PrecomputeDftKernels1};
use shader_crate::GpuComplexPolar;
use std::ops::{Range, RangeInclusive};
use kiss3d::glamx::Vec3;

#[derive(Shader)]
struct GpuKernels {
    pub fdtd_1d: Fdtd1,
    pub precompute_dft_kernels1d: PrecomputeDftKernels1,
    pub dft1d: Dft1,
    pub finish_dft1d: FinishDft1,
}

pub async fn run_fdtd_1d(backend: &GpuBackend) {

    // let obj_width = 0.3048; // 1ft wide
    // let eps_r_mat = 12.0_f32;
    // let mu_r_mat = 1.0_f32;

    // let obj = ObjectInfo1D {
    //     width: obj_width,
    //     position: 0.,
    //     material: ElectricMaterial::new(eps_r_mat, mu_r_mat),
    //     color: GREEN
    // };

    let wavelength_0 = 980e-9;
    let si_o2_mat = ElectricMaterial::new(1.5, 1.);
    let si_n_mat = ElectricMaterial::new(2.0, 1.);
    let si_o2_width = wavelength_0 / (4. * si_o2_mat.n);
    let si_n_width = wavelength_0 / (4. * si_n_mat.n);

    let widths = [si_o2_width, si_n_width];
    let mats = [si_o2_mat, si_n_mat];
    let mut layers = vec![ObjectInfo1 { color: GREEN, ..Default::default() }; 30];
    let mut curr_pos = 0.;
    let mut prev_half_width = 0.;
    for (i, layer) in layers.iter_mut().enumerate() {
        let mat_i = i % 2;
        let width = widths[mat_i];
        let half_width = width/2.;
        curr_pos += prev_half_width + half_width;
        layer.material = mats[mat_i];
        layer.position = curr_pos;
        layer.width = width;

        prev_half_width = half_width;
    }

    let max_pulse_freq = MaterialConstants1::C_0 / wavelength_0 * 1.5;
    println!("Target Frequency: {}", MaterialConstants1::C_0 / wavelength_0);

    let stability_values = StabilityValues::default();

    let source = GaussianPulse1::from_max_frequency(max_pulse_freq, 1.);

    let mut data = Fdtd1Data::new();
    data.set_source(source);
    for layer in layers.iter().cloned() {
        data.add_object(layer);
    }

    data.prepare_for_gpu(&stability_values, None);

    // let mut f_res = data.estimate_max_timesteps(None);
    data.enable_dfts((0.0)..=(max_pulse_freq), 5000);

    data.set_step_count(2);
    println!("{:?}", data.grid_info);

    let max_src_val = data.source_vals.iter()
        .map(|v| v.abs())
        .max_by(|a, b| a.total_cmp(b))
        .unwrap();

    main_render_loop(backend, data, max_src_val).await.unwrap();
}

async fn main_render_loop(
    backend: &GpuBackend,
    mut data: Fdtd1Data,
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

    // Compute DFT kernels before running simulation
    if let Some(dft) = &mut data.dft {
        buffers.dft_buffers = Some(dft.create_buffers(backend)?);
        let dft_bufs = buffers.dft_buffers.as_mut().unwrap();
        let mut encoder = backend.begin_encoding();
        let mut pass = encoder.begin_pass("precompute_dft_kernels1d", None);
        gpu_kernels.precompute_dft_kernels1d.call(
            &mut pass,
            DispatchGrid::Grid(dft_bufs.dispatch_grid),
            &mut dft_bufs.kernels,
            &buffers.grid_info,
            &dft_bufs.dft_info
        )?;
        drop(pass);
        backend.submit(encoder)?;
    }

    let mut abs_max_val = 0.;
    let mut cells_out = vec![GridCell1::default(); grid_info.num_cells as usize];
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
            if let Some(dft_bufs) = &buffers.dft_buffers {
                let dft = data.dft.as_mut().unwrap();
                dft_bufs.reflectance.read(backend, &mut dft.reflectance).await?;
                dft_bufs.transmittance.read(backend, &mut dft.transmittance).await?;
                dft_bufs.source.read(backend, &mut dft.source).await?;
                dft.update_dft_plots();
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

        if let Some(dft) = &mut data.dft {
            window.draw_ui(|ctx| dft.plot_dft(ctx));
        }
    }
    Ok(())
}

fn submit_simulation(
    backend: &GpuBackend,
    gpu_kernels: &GpuKernels,
    buffers: &mut Fdtd1Buffers,
    grid_info: &GridInfo1
) -> Result<(), GpuBackendError> {
    let mut encoder = backend.begin_encoding();

    // Compute pass
    let mut pass = encoder.begin_pass("", None);
    for _ in 0..grid_info.steps_per_call {
        gpu_kernels.fdtd_1d.call(
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

        if let Some(dft_bufs) = &mut buffers.dft_buffers {
            gpu_kernels.dft1d.call(
                &mut pass,
                DispatchGrid::Grid(dft_bufs.dispatch_grid),
                &dft_bufs.kernels,
                &mut dft_bufs.reflectance.buffer,
                &mut dft_bufs.transmittance.buffer,
                &mut dft_bufs.source.buffer,
                &buffers.source_vals,
                &buffers.cells.buffer,
                &buffers.timestep_counter
            )?;
            // TODO: call finish_dft_1d when reaching a "max iterations" value
        }
    }
    drop(pass);

    buffers.cells.encode_copy_cmd(&mut encoder)?;
    buffers.perfect_boundary_data.encode_copy_cmd(&mut encoder)?;
    if let Some(dft_bufs) = &mut buffers.dft_buffers {
        dft_bufs.reflectance.encode_copy_cmd(&mut encoder)?;
        dft_bufs.transmittance.encode_copy_cmd(&mut encoder)?;
        dft_bufs.source.encode_copy_cmd(&mut encoder)?;
    }

    backend.submit(encoder)
}

pub struct Dft {
    pub freq_range: RangeInclusive<f32>,
    pub f_increment: f32,
    pub reflectance: Vec<GpuComplexPolar>,
    pub transmittance: Vec<GpuComplexPolar>,
    pub source: Vec<GpuComplexPolar>,
    pub plot: DftPlot,
}

impl Dft {
    pub fn new(frequency_range: RangeInclusive<f32>, resolution: usize) -> Self {
        let start = *frequency_range.start() as f64;
        let end = *frequency_range.end() as f64;
        let f_increment = (end - start) / (resolution as f64 - 1.);

        let plot_points = (0..resolution)
            .map(|i| PlotPoint::new(start + f_increment * i as f64, 0.))
            .collect::<Vec<_>>();

        Self {
            freq_range: frequency_range,
            f_increment: f_increment as f32,
            reflectance: vec![GpuComplexPolar::default(); resolution],
            transmittance: vec![GpuComplexPolar::default(); resolution],
            source: vec![GpuComplexPolar::default(); resolution],
            plot: DftPlot {
                reflectance: plot_points.clone(),
                transmittance: plot_points.clone(),
                sum: plot_points,
                prev_pointer_pos: None
            },
        }
    }

    pub fn plot_dft(&mut self, egui_ctx: &egui::Context) {
        egui::Window::new("Reflectance & Transmittance DFTs").show(egui_ctx, |ui| {
            if let Some(pos) = self.plot.prev_pointer_pos {
                ui.label(format!("Pointer coords: ({}, {})", pos.x, pos.y));
            }
            Plot::new("DFT")
                .legend(Legend::default())
                .show(ui, |plot_ui| {
                    self.plot.prev_pointer_pos = plot_ui.pointer_coordinate();
                    plot_ui.line(Line::new("Reflectance", PlotPoints::Borrowed(&self.plot.reflectance)));
                    plot_ui.line(Line::new("Transmittance", PlotPoints::Borrowed(&self.plot.transmittance)));
                    plot_ui.line(Line::new("Reflectance + Transmittance", PlotPoints::Borrowed(&self.plot.sum)));
                })
        });
    }

    /// Prepare & normalize DFT plots. Called when DFTs have changed
    pub fn update_dft_plots(&mut self) {
        self.plot.reflectance.par_iter_mut()
            .zip(self.plot.transmittance.par_iter_mut())
            .zip(self.plot.sum.par_iter_mut())
            .map(|((r,t), s)| (r, t, s))
            .enumerate()
            .for_each(|(i, (refl, trans, sum))| {
                let src = self.source[i].r as f64;
                refl.y = (self.reflectance[i].r as f64 / src).powi(2);
                trans.y = (self.transmittance[i].r as f64 / src).powi(2);
                sum.y = refl.y + trans.y;
            });
    }

    pub fn create_buffers(&self, backend: &GpuBackend) -> GpuResult<DftBuffers> {
        let kernels = vec![GpuComplexPolar::default(); self.reflectance.len()];
        Ok(
            DftBuffers {
                reflectance: self.reflectance.create_gpu_buffer_readable(backend)?,
                transmittance: self.transmittance.create_gpu_buffer_readable(backend)?,
                source: self.source.create_gpu_buffer_readable(backend)?,
                kernels: kernels.create_gpu_buffer(backend)?,
                dft_info: DftInfo1 {
                    f_start: *self.freq_range.start(),
                    f_increment: self.f_increment
                }.create_gpu_uniform(backend)?,
                dispatch_grid: [kernels.len().div_ceil(64) as u32, 1, 1]
            }
        )
    }
}

pub struct DftPlot {
    pub reflectance: Vec<PlotPoint>,
    pub transmittance: Vec<PlotPoint>,
    pub sum: Vec<PlotPoint>,
    pub prev_pointer_pos: Option<PlotPoint>,
}

pub struct DftBuffers {
    pub reflectance: GpuBufferReadable<GpuComplexPolar>,
    pub transmittance: GpuBufferReadable<GpuComplexPolar>,
    pub source: GpuBufferReadable<GpuComplexPolar>,
    pub kernels: GpuBuffer<GpuComplexPolar>,
    pub dft_info: GpuBuffer<DftInfo1>,
    pub dispatch_grid: [u32; 3],
}

#[derive(Default)]
pub struct Fdtd1Data {
    pub cells: Vec<GridCell1>,
    pub materials: Vec<MaterialConstants1>,
    pub source: GaussianPulse1,
    pub source_vals: Vec<f32>,
    pub source_gpu: GpuSource1,
    pub grid_info: GridInfo1,

    pub dft: Option<Dft>,

    pub objects: Vec<ObjectInfo1>,
    /// The range of cells each object takes
    pub object_cell_indices: Vec<Range<usize>>,
}


impl Fdtd1Data {
    pub fn new() -> Self {
        Self {
            grid_info: GridInfo1::max_values(0.),
            ..Default::default()
        }
    }

    pub fn enable_dfts(&mut self, frequency_range: RangeInclusive<f32>, resolution: u32) -> &mut Self {
        let resolution = resolution + (resolution % 2 == 0) as u32; // resolution must be odd
        self.dft = Some(Dft::new(frequency_range, resolution as usize));
        self
    }

    /// Estimates the number of timesteps the simulation needs to be considered "finished."
    /// Does NOT consider the resonance of objects in the simulations.
    ///
    /// You can use this function's output as DFT resolution
    pub fn estimate_max_timesteps(&mut self, custom_default_material: Option<ElectricMaterial>) -> u32 {
        let default_mat = custom_default_material.unwrap_or(ElectricMaterial::FREE_SPACE);
        let mut n_max = self.objects.iter()
            .map(|o| o.material.n)
            .max_by(|a, b| a.total_cmp(b))
            .unwrap_or(1.);
        n_max = n_max.max(default_mat.n);

        let max_src_duration = self.source.tau * 12.;
        // time it takes to the slowest wave to propagate across the grid (a worst-case scenario)
        let t_prop = n_max/ MaterialConstants1::C_0 * self.grid_info.num_cells as f32;
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
        self.materials.resize(materials.len(), MaterialConstants1::default());
        for (mat_consts, mat) in self.materials.iter_mut().zip(materials) {
            *mat_consts = MaterialConstants1::new(mat.eps_r, mat.mu_r, dt);
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
        self.cells.resize(self.grid_info.num_cells as usize, GridCell1::default());

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
        self.source_gpu = GpuSource1 {
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
        let min_wavelen = MaterialConstants1::C_0 / (f_max * n_max);
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
        (n_min * self.grid_info.cell_size) / (safety_margin * MaterialConstants1::C_0)
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

    pub fn create_buffers(&self, backend: &GpuBackend) -> Result<Fdtd1Buffers, GpuBackendError> {
        let timestep_counter = 0;
        let mut dft_buffers = None;
        if let Some(dft) = &self.dft {
            dft_buffers = Some(dft.create_buffers(backend)?);
        }

        Ok(Fdtd1Buffers {
            cells: self.cells.create_gpu_buffer_readable(backend)?,
            materials: self.materials.create_gpu_buffer(backend)?,
            source_vals: self.source_vals.create_gpu_buffer(backend)?,
            source: self.source_gpu.create_gpu_buffer(backend)?,
            perfect_boundary_data: PerfectBoundaryData::default()
                .create_gpu_buffer_readable(backend)?,
            grid_info: self.grid_info.create_gpu_uniform(backend)?,
            dft_buffers,
            timestep_counter: timestep_counter.create_gpu_buffer(backend)?,
        })
    }

    pub fn add_object(&mut self, obj: ObjectInfo1) -> &mut Self {
        self.objects.push(obj);
        self
    }

    pub fn set_source(&mut self, src: GaussianPulse1) -> &mut Self {
        self.source = src;
        self
    }

    /// Returns starting cell index and the width of the object in grid cells: `(start_idx, idx_width)`
    pub fn get_obj_indices(&self, obj: &ObjectInfo1) -> (usize, usize) {
        let center_idx = (obj.position / self.grid_info.cell_size) as usize;
        let idx_width = (obj.width / self.grid_info.cell_size).ceil() as usize;
        let start = (center_idx - idx_width.div_ceil(2)).max(0);
        (start, idx_width)
    }

    pub fn compute_max_frequency(&self) -> f32 {
        0.5 / self.source.tau
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

#[derive(Clone, Default)]
pub struct ObjectInfo1 {
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

pub struct Fdtd1Buffers {
    pub cells: GpuBufferReadable<GridCell1>,
    pub materials: GpuBuffer<MaterialConstants1>,
    pub source_vals: GpuBuffer<f32>,
    pub source: GpuBuffer<GpuSource1>,
    pub perfect_boundary_data: GpuBufferReadable<PerfectBoundaryData>,
    pub grid_info: GpuBuffer<GridInfo1>,
    pub dft_buffers: Option<DftBuffers>,
    pub timestep_counter: GpuBuffer<u32>,
}

#[derive(Debug)]
pub struct GaussianPulse1 {
    pub amplitude: f32,
    pub tau: f32,
    pub t_0: f32,
}

impl GaussianPulse1 {
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

    pub fn compute_source_values(&self, sim_data: &Fdtd1Data) -> Vec<f32> {
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

impl Default for GaussianPulse1 {
    fn default() -> Self {
        Self {
            amplitude: 0.,
            tau: 1.,
            t_0: 0.,
        }
    }
}