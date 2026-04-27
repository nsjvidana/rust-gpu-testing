use glam::{UVec3, Vec3};
use khal::backend::{Backend, Buffer, DeviceValue, Encoder, GpuBackend, GpuBuffer, GpuEncoder, MaybeSendSync};
use khal::BufferUsages;
use khal::re_exports::bytemuck::{AnyBitPattern, NoUninit};
use kiss3d::prelude::Polyline3d;
use crate::prelude::*;

pub struct GpuBufferReadable<T: DeviceValue + NoUninit + AnyBitPattern> {
    pub buffer: GpuBuffer<T>,
    pub readback: GpuBuffer<T>,
}

impl<T: DeviceValue + NoUninit + AnyBitPattern> GpuBufferReadable<T> {
    pub fn encode_copy_cmd(&mut self, encoder: &mut GpuEncoder) -> GpuResult<()> {
        encoder.copy_buffer_to_buffer(
            &self.buffer,
            0,
            &mut self.readback,
            0,
            self.buffer.len()
        )
    }

    pub async fn read(&self, backend: &GpuBackend, out: &mut [T]) -> GpuResult<()> {
        backend.read_buffer(&self.readback, out).await
    }
}

pub trait CreateGpuBuffer<T: DeviceValue + NoUninit> {
    fn create_gpu_buffer(&self, backend: &GpuBackend) -> GpuResult<GpuBuffer<T>>;
    fn create_gpu_uniform(&self, backend: &GpuBackend) -> GpuResult<GpuBuffer<T>> {
        panic!("Unimplemented or cannot be a uniform!")
    }
}

impl<T: DeviceValue + NoUninit> CreateGpuBuffer<T> for Vec<T> {
    fn create_gpu_buffer(&self, backend: &GpuBackend) -> GpuResult<GpuBuffer<T>> {
        backend.init_buffer(
            self.as_slice(),
            BufferUsages::STORAGE,
        )
    }
}

impl<T: DeviceValue + NoUninit> CreateGpuBuffer<T> for T {
    fn create_gpu_buffer(&self, backend: &GpuBackend) -> GpuResult<GpuBuffer<T>> {
        backend.init_buffer(
            &[*self],
            BufferUsages::STORAGE,
        )
    }
    fn create_gpu_uniform(&self, backend: &GpuBackend) -> GpuResult<GpuBuffer<T>> {
        backend.init_buffer(
            &[*self],
            BufferUsages::UNIFORM,
        )
    }
}

pub trait CreateGpuBufferReadable<T: DeviceValue + NoUninit + AnyBitPattern> {
    fn create_gpu_buffer_readable(&self, backend: &GpuBackend) -> GpuResult<GpuBufferReadable<T>>;
}

impl<T: DeviceValue + NoUninit + AnyBitPattern> CreateGpuBufferReadable<T> for Vec<T> {
    fn create_gpu_buffer_readable(&self, backend: &GpuBackend) -> GpuResult<GpuBufferReadable<T>> {
        let buffer = backend.init_buffer(
            self.as_slice(),
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        )?;
        let readback = backend.uninit_buffer(
            buffer.len(),
            BufferUsages::COPY_DST | BufferUsages::MAP_READ
        )?;
        Ok(
            GpuBufferReadable { buffer, readback }
        )
    }
}

impl<T: DeviceValue + NoUninit + AnyBitPattern> CreateGpuBufferReadable<T> for T {
    fn create_gpu_buffer_readable(&self, backend: &GpuBackend) -> GpuResult<GpuBufferReadable<T>> {
        let buffer = backend.init_buffer(
            &[*self],
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        )?;
        let readback = backend.uninit_buffer(
            buffer.len(),
            BufferUsages::COPY_DST | BufferUsages::MAP_READ
        )?;
        Ok(
            GpuBufferReadable { buffer, readback }
        )
    }
}

/// Constructs a [`kiss3d::prelude::Polyline3d`] that draws a bounding box with the extents `extents`
/// and "origin vertex" at `position`
pub fn bb_polyline(extents: Vec3, position: Vec3) -> Polyline3d {
    let v0 = position;
    let v0x = position + Vec3::new(extents.x, 0., 0.);
    let v0xy = position + Vec3::new(extents.x, extents.y, 0.);
    let v0y = position + Vec3::new(0., extents.y, 0.);
    let v1 = v0 + Vec3::new(0., 0., extents.z);
    let v1x = v0x + Vec3::new(0., 0., extents.z);
    let v1xy = v0xy + Vec3::new(0., 0., extents.z);
    let v1y = v0y + Vec3::new(0., 0., extents.z);
    Polyline3d::new(vec![
        v0, v0x, v0xy, v0y, v0,
        v1, v1x, v1xy, v1y, v1,
        v1x, v0x, v0xy, v1xy,
        v1y, v0y
    ])
}

/// Creates a [`kiss3d::prelude::Polyline3d`] that draws an arrow starting at `position`, pointing in
/// the direction of `direction_length` with a length of the `direction_length` vector.
pub fn arrow_polyline(position: Vec3, direction_length: Vec3) -> Polyline3d {
    let u = -direction_length;
    let mut axis = u.cross(Vec3::X).normalize_or_zero();
    if axis.length_squared() == 0. {
        axis = u.cross(Vec3::Y).normalize();
    }
    let arrow_head = u.rotate_axis(axis, -std::f32::consts::FRAC_PI_4)
        .normalize() * direction_length.length() / 3.;

    Polyline3d::new(vec![
        position,
        position + direction_length,
        position + direction_length + arrow_head
    ])
}