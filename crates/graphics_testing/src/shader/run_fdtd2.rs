use crate::prelude::GpuResult;
use crate::util::{arrow_polyline, bb_polyline, CreateGpuBuffer, CreateGpuBufferReadable, GpuBufferReadable};
use glam::{USizeVec3, UVec2, Vec2};
use khal::backend::{Backend, DispatchGrid, Encoder, GpuBackend, GpuBuffer};
use khal::Shader;
use kiss3d::prelude::*;
use shader_crate::fdtd2::{Fdtd2, GridCell2, GridInfo2, MaterialConstants2};
use shader_crate::{flat_idx_to_vector, vector_to_flat_idx};

#[derive(Shader)]
struct GpuKernels {
    fdtd2: Fdtd2
}

pub async fn run_fdtd2(backend: &GpuBackend) -> GpuResult<()> {
    let gpu_kernels = GpuKernels::from_backend(backend)?;
    let mut data = FdtdData2::new();

    let pulse_freq = 1e6;
    let pulse_amplitude = 1.;

    data.min_wavelength(pulse_freq, 10)
        .cfl_condition();
    data.grid.n_cells = UVec2::new(20, 10);
    data.grid.cells.resize(data.grid.n_cells.element_product() as usize, GridCell2::default());

    // TODO: remove this test value
    data.grid.cells[30].en_z = pulse_amplitude;

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
    let mut render_data = RenderData2::new(&data, pulse_amplitude, 0.01);
    // Main render loop
    while window.render_3d(&mut scene, &mut camera).await {
        if window.get_key(Key::T) == Action::Press {
            backend.synchronize()?;
            runner.cells.read(backend, &mut data.grid.cells).await?;
            runner.submit_step(&gpu_kernels.fdtd2, backend)?;
        }
        
        render_data.render_simulation(&mut window, &data);
    }
    runner.submit_step(&gpu_kernels.fdtd2, backend)?;

    Ok(())
}

pub struct RenderData2 {
    pub en_arrows: Vec<(Vec3, Polyline3d)>,
    pub en_color: Color,
    pub grid_bb: Polyline3d,
    pub max_en_val: f32,
    pub alpha_threshold: f32,
}

impl RenderData2 {
    pub fn new(data: &FdtdData2, max_src_val: f32, alpha_threshold: f32) -> Self {
        let grid = &data.grid;
        let en_color = RED;
        let en_arrow = arrow_polyline(Vec3::ZERO, Vec3::Z * grid.cell_size.length())
            .with_color(en_color);
        let n_cells3 = USizeVec3::from((grid.n_cells.as_usizevec2(), 1));
        let cell_size3 = Vec3::from((grid.cell_size, 0.));

        let grid_bb = bb_polyline(
            Vec3::from(cell_size3 * n_cells3.as_vec3()),
            Vec3::ZERO
        );

        Self {
            en_arrows: (0..grid.cells.len())
                .map(|i| {
                    let i3 = flat_idx_to_vector(i, n_cells3);
                    let c_pos = i3.as_vec3() * cell_size3;
                    let polyline = en_arrow.clone()
                        .with_transform(Pose3::from_translation(c_pos))
                        .with_color(en_color);
                    (c_pos, polyline)
                })
                .collect(),
            max_en_val: max_src_val,
            grid_bb,
            en_color,
            alpha_threshold
        }
    }

    pub fn render_simulation(&mut self, window: &mut Window, data: &FdtdData2) {
        for (c, (_, arrow)) in data.grid.cells.iter()
            .zip(self.en_arrows.iter_mut())
        {
            let alpha = c.en_z / self.max_en_val;
            arrow.color = self.en_color.with_alpha(alpha);
            if alpha > self.alpha_threshold {
                window.draw_polyline(arrow);
            }
        }

        window.draw_polyline(&self.grid_bb);
    }
}

pub struct FdtdData2 {
    pub dt: f32,
    pub grid: FdtdGrid2,
    pub materials: Vec<ElectricMaterial2>,
}

impl FdtdData2 {
    pub fn new() -> Self {
        Self {
            dt: f32::MAX,
            grid: FdtdGrid2::new(),
            materials: vec![],
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
    pub fn cfl_condition(&mut self) -> &mut Self {
        let denom_term_2 = self.grid.cell_size
            .recip()
            .powf(2.)
            .element_sum()
            .sqrt();
        self.dt = self.dt.min(
            1. / (ElectricMaterial2::C_0 * denom_term_2)
        );
        self
    }

    pub fn prepare_materials(&mut self) -> &mut Self {
        if self.materials.is_empty() {
            self.materials.push(ElectricMaterial2::FREE_SPACE);
        }
        // TODO: include object materials
        self
    }

    pub fn create_gpu(&mut self, steps_per_submission: usize, backend: &GpuBackend) -> GpuResult<GpuFdtd2> {
        let n_cells3 = USizeVec3::from((self.grid.n_cells.as_usizevec2(), 1));
        self.prepare_materials();

        let gpu_fdtd = GpuFdtd2 {
            cells: self.grid.cells.create_gpu_buffer_readable(backend)?,
            grid: GridInfo2 {
                n_cells: self.grid.n_cells,
                cell_size: self.grid.cell_size,
                i_incr: UVec2::new(
                    vector_to_flat_idx(USizeVec3::X, n_cells3) as u32,
                    vector_to_flat_idx(USizeVec3::Y, n_cells3) as u32
                ),
                dn_z_update_coeff: ElectricMaterial2::C_0 * self.dt,
                _padding: 0
            }.create_gpu_uniform(backend)?,
            materials: self.materials.iter()
                .map(|m| m.to_gpu(self.dt))
                .collect::<Vec<_>>()
                .create_gpu_buffer(backend)?,
            dispatch_grid: n_cells3.map(|v| v.div_ceil(8)).as_uvec3().to_array(),
            steps_per_submission,
        };

        Ok(gpu_fdtd)
    }
}

pub struct GpuFdtd2 {
    pub grid: GpuBuffer<GridInfo2>,
    pub cells: GpuBufferReadable<GridCell2>,
    pub materials: GpuBuffer<MaterialConstants2>,
    pub dispatch_grid: [u32; 3],
    pub steps_per_submission: usize,
}

impl GpuFdtd2 {
    pub fn submit_step(&mut self, gpu_kernel: &Fdtd2, backend: &GpuBackend) -> GpuResult<()> {
        let mut encoder = backend.begin_encoding();

        let mut pass = encoder.begin_pass("fdtd2", None);
        for _ in 0..self.steps_per_submission {
            gpu_kernel.call(
                &mut pass,
                DispatchGrid::Grid(self.dispatch_grid),
                &mut self.cells.buffer,
                &self.materials,
                &self.grid
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
        eps_r_z: Self::EPS_0,
        mu_r: Vec2::splat(Self::MU_0),
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
                c_0_dt / self.mu_r.y,
            ),
            en_z_update_coeff: 1. / self.eps_r_z,
            ..Default::default()
        }
    }
}