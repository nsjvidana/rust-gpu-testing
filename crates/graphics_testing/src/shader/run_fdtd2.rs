use glam::{USizeVec3, UVec2, Vec2};
use khal::backend::{Backend, DispatchGrid, Encoder, GpuBackend, GpuBuffer};
use khal::Shader;
use shader_crate::fdtd2::{Fdtd2, GridCell2, GridInfo2, MaterialConstants2};
use shader_crate::vector_to_flat_idx;
use crate::prelude::GpuResult;
use crate::util::{CreateGpuBuffer, CreateGpuBufferReadable, GpuBufferReadable};

#[derive(Shader)]
struct GpuKernels {
    fdtd2: Fdtd2
}

pub async fn run_fdtd2(backend: &GpuBackend) {
    
}

pub struct FdtdData2 {
    pub dt: f32,
    pub grid: FdtdGrid2,
    pub materials: Vec<ElectricMaterial2>,
}

impl FdtdData2 {
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

    pub fn create_gpu(&self, steps_per_submission: usize, backend: &GpuBackend) -> GpuResult<GpuFdtd2> {
        let n_cells3 = USizeVec3::from((self.grid.n_cells.as_usizevec2(), 0));
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
            }.create_gpu_uniform(backend)?,
            materials: self.materials.iter()
                .map(|m| m.to_gpu(self.dt))
                .collect::<Vec<_>>()
                .create_gpu_buffer(backend)?,
            dispatch_grid: n_cells3.map(|v| v.div_ceil(8)).as_uvec3().to_array(),
            steps_per_submission
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

#[derive(Copy, Clone, Debug)]
pub struct ElectricMaterial2 {
    /// Relative Magnetic Permeability (X & Y component of tensor diagonal)
    pub mu_r: Vec2,
    /// Relative Electric Permittivity (Z component of tensor diagonal)
    pub eps_r_z: f32,
    /// Refractive Index
    pub n: f32,
    /// Impedance
    pub impedance: f32,
}

impl ElectricMaterial2 {
    /// Speed of EM wave in free space
    pub const C_0: f32 = 299792458.0;

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