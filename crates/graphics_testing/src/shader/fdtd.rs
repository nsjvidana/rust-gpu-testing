use std::num::NonZeroU32;
use std::ops::Div;
use glam::{UVec3, Vec3, Vec4};
use crate::prelude::*;
use khal::backend::{Backend, DispatchGrid, GpuBackend, GpuBackendError, GpuBuffer, GpuPass};
use khal::BufferUsages;
use khal::re_exports::bytemuck::{Pod, Zeroable};
use rapier3d_meshloader::LoadedShape;
use shader_crate::{vector_to_flat_idx, FdtdDirichlet, GpuSource, GridCell, GridInfo, MaterialConstants, PointCharge};

pub const PROTON_MASS: f32 = 1.6726219259552e-27;
pub const ELECTRON_MASS: f32 = 9.109383713928e-31;
pub const ELEMENTARY_CHARGE: f32 = 1.602176634e-19;

pub struct FdtdData {
    pub cells: Vec<GridCell>,
    pub material_constants: Vec<MaterialConstants>,
    pub grid_info: GridInfo,
    pub sources: Vec<Source>,
    pub point_charges: Vec<PointCharge>,
}

impl FdtdData {
    pub fn new(grid_info: GridInfo) -> Result<Self> {
        let num_cells = grid_info.grid_dimensions.element_product();
        if num_cells == 0 {
            return Err(Error::BufferSizeZero)
        }
        Ok(Self {
            cells: vec![GridCell::default(); num_cells as usize],
            material_constants: vec![MaterialConstants::free_space(grid_info.dt)],
            grid_info,
            sources: vec![GaussianPulse::default().construct_source(1, 1., 0.)],
            point_charges: vec![PointCharge::default()],
        })
    }
}

pub struct MaxwellEqsBuffers {
    pub cells: GpuBuffer<GridCell>,
    pub material_constants: GpuBuffer<MaterialConstants>,
    pub source_values: GpuBuffer<Vec4>,
    pub sources: GpuBuffer<GpuSource>,
    pub grid_info: GpuBuffer<GridInfo>,
    pub point_charges: GpuBuffer<PointCharge>,
}

/// A source signal (currently for E field only)
pub struct Source {
    /// The values that will be injected into the simulation
    pub values: Vec<Vec4>,
    /// The index of the cell at which the source will be injected.
    pub cell_idx: u32,
    /// The timestep when the source must be enabled
    pub begin_timestep: u32
}

/// A Gaussian pulse at a specific point in space
///
/// Computed as `amplitude * E ^ (-((t - t_offset)/half_duration)^2)`
///
/// If you are using a Gaussian pulse in your simulation, it is recommended to use the [`GridInfo`]
#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct GaussianPulse {
    /// Offset when the significant part of the pulse appears
    pub t_offset: f32,
    /// Controls duration of pulse. Cannot be 0.
    pub half_duration: f32,
    pub amplitude: f32,
    pub cell_idx: u32,
}

impl GaussianPulse {
    pub fn from_max_frequency(max_frequency: f32, amplitude: f32, at_point: Vec3, grid: &GridInfo) -> Self {
        let half_duration = core::f32::consts::FRAC_1_PI / max_frequency; // 1.0 / (π * max_frequency)
        let t_offset = 6. * half_duration;
        let cell_idx_vector = (at_point / grid.cell_size).floor().as_uvec3();
        let cell_idx = vector_to_flat_idx(cell_idx_vector, grid.grid_dimensions);

        Self {
            t_offset,
            half_duration,
            amplitude,
            cell_idx,
        }
    }

    /// Construct a [`Source`] struct containing all values that will be injected into the simulation
    /// over time.
    pub fn construct_source(&self, num_timesteps: usize, dt: f32, begin_timestep: f32) -> Source {
        let mut t = 0.;
        let mut values = vec![Vec4::ZERO; num_timesteps];
        for i in 0..num_timesteps {
            t += dt;
            let g = self.amplitude * core::f32::consts::E.powf(
                -((t - self.t_offset)/self.half_duration).powi(2)
            );
            values[i] = Vec4::splat(g);
        }
        Source {
            values,
            cell_idx: self.cell_idx,
            begin_timestep: (begin_timestep / dt).round() as u32,
        }
    }
}

impl Default for GaussianPulse {
    fn default() -> Self {
        Self {
            t_offset: 0.,
            half_duration: 1.,
            amplitude: 0.,
            cell_idx: 0,
        }
    }
}


// TODO: move this into impl FdtdData
pub fn create_buffers(
    backend: &GpuBackend,
    data: &FdtdData
) -> std::result::Result<MaxwellEqsBuffers, GpuBackendError> {
    let num_vals = data.sources.iter()
        .map(|v| v.values.len())
        .sum::<usize>();
    let mut all_source_values = Vec::with_capacity(num_vals);
    let mut sources = Vec::with_capacity(data.sources.len());
    for src in data.sources.iter() {
        let start_idx = all_source_values.len() as u32;
        all_source_values.extend_from_slice(src.values.as_slice());
        let end_idx = start_idx + src.values.len() as u32;
        sources.push(GpuSource::new(
            start_idx,
            end_idx,
            src.cell_idx,
            src.begin_timestep
        ))
    }

    Ok(
        MaxwellEqsBuffers {
            cells: backend.init_buffer(
                data.cells.as_slice(),
                BufferUsages::STORAGE | BufferUsages::COPY_SRC
            )?,
            material_constants: backend.init_buffer(
                data.material_constants.as_slice(),
                BufferUsages::STORAGE
            )?,
            grid_info: backend.init_buffer(
                &[data.grid_info],
                BufferUsages::UNIFORM
            )?,
            source_values: backend.init_buffer(
                all_source_values.as_slice(),
                BufferUsages::STORAGE
            )?,
            sources: backend.init_buffer(
                sources.as_slice(),
                BufferUsages::STORAGE | BufferUsages::COPY_SRC
            )?,
            point_charges: backend.init_buffer(
                data.point_charges.as_slice(),
                BufferUsages::STORAGE | BufferUsages::COPY_SRC
            )?,
        }
    )
}

/// Compute grid cell size and index dimensions
///
/// This is a helper function, so there is no need to use this function if you know what cell size
/// you want, how many cells on each axis, etc.
///
/// `min_wavelength` & `min_feature_length` must not be zero.
pub fn compute_grid_dimensions(
    grid_dimensions: Vec3,
    min_wavelength: f32,
    min_feature_length: f32,
    cells_per_wavelength: NonZeroU32,
    cells_per_feature_length: NonZeroU32,
) -> (Vec3, UVec3) {
    let cell_size = (min_wavelength / cells_per_wavelength.get() as f32)
        .min(min_feature_length / cells_per_feature_length.get() as f32);
    (Vec3::splat(cell_size), grid_dimensions.div(cell_size).ceil().as_uvec3())
}

/// Calculate the size of the smallest feature in a mesh shape.
pub fn min_feature_length(mesh: LoadedShape) -> f32 {
    mesh.raw_mesh.faces.iter().map(|f| {
        let verts = f.map(|i| mesh.raw_mesh.vertices[i as usize]);
        let len0 = (Vec3::from_array(verts[0]) - Vec3::from_array(verts[1]))
            .length_squared();
        let len1 = (Vec3::from_array(verts[0]) - Vec3::from_array(verts[2]))
            .length_squared();
        let len2 = (Vec3::from_array(verts[1]) - Vec3::from_array(verts[2]))
            .length_squared();
        len0.max(len1).max(len2).sqrt()
    }).max_by(|a, b| a.total_cmp(b)).expect("Mesh cannot be empty")
}

/// Calculate the minimum wavelength that will be in the simulation
///
/// `max_freq` is the maximum frequency of a signal that will be injected into the simulation
/// `max_refractive_idx` is the largest refractive index that a material will have
pub fn min_wavelength(max_freq: f32, max_refractive_idx: f32) -> f32 {
    MaterialConstants::C_0 / (max_freq * max_refractive_idx)
}