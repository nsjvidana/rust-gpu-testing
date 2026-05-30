use crate::prelude::*;
use khal::backend::{Backend, Buffer, DeviceValue, Encoder, GpuBackend, GpuBuffer, GpuEncoder};
use khal::re_exports::bytemuck::{AnyBitPattern, NoUninit};
use khal::BufferUsages;

pub use self::visualization::*;

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

mod visualization {
    use kiss3d::color::Color;
    use glamx::*;
    use kiss3d::prelude::{Polyline3d, Window};

    /// Draws a bounding box `bb` where `bb = [min, max]` of the bb's bounds.
    pub fn draw_bb(window: &mut Window, bb: [Vec3; 2], color: Color, width: f32, perspective: bool) {
        let [dx, dy, dz] = Mat3::from_diagonal(bb[1] - bb[0]).to_cols_array_2d()
            .map(|v| Vec3::from(v));
        let btm = [bb[0], bb[1] - dz];
        let top = [bb[0] + dz, bb[1]];

        window.draw_line(btm[0], btm[0] + dx, color, width, perspective);
        window.draw_line(btm[0], btm[0] + dy, color, width, perspective);
        window.draw_line(btm[0], btm[0] + dz, color, width, perspective);
        window.draw_line(btm[1], btm[1] - dx, color, width, perspective);
        window.draw_line(btm[1], btm[1] - dy, color, width, perspective);
        window.draw_line(btm[1], btm[1] - dz, color, width, perspective);
        window.draw_line(top[0], top[0] + dx, color, width, perspective);
        window.draw_line(top[0], top[0] + dy, color, width, perspective);
        window.draw_line(top[0], top[0] + dz, color, width, perspective);
        window.draw_line(top[1], top[1] - dx, color, width, perspective);
        window.draw_line(top[1], top[1] - dy, color, width, perspective);
        window.draw_line(top[1], top[1] - dz, color, width, perspective);
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
}