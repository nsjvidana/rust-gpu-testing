#![cfg_attr(target_arch = "spirv", no_std)]

use khal_std::glamx::Vec2;
use bytemuck::{Pod, Zeroable};
use khal_std::num_traits::Float;

pub mod fdtd1;
pub mod fdtd2;

/// Select between two numerical values `tru` and `fals` of type `typ` depending on a bool `b`.
///
/// Returns `tru` if `b` is true, `fals` otherwise
#[macro_export]
macro_rules! select_val {
    ($b:expr, $tru:expr, $fals:expr, $typ:ty) => {{
        let b_n = (!$b) as usize as $typ;
        let b = $b as usize as $typ;
        $tru * b + $fals * b_n
    }};
}

#[macro_export]
macro_rules! flat_idx_to_vector {
    ($idx:expr, $grid_dim:expr, $vec_type:ty) => {{
        <$vec_type>::new(
            $idx % $grid_dim.x,
            ($idx / $grid_dim.x) % $grid_dim.y,
            $idx / ($grid_dim.x * $grid_dim.y),
        )
    }};
}

#[macro_export]
macro_rules! vector_to_flat_idx {
    ($v:expr, $grid_dim:expr) => {{
        $v.z * $grid_dim.x * $grid_dim.y +
            $v.y * $grid_dim.x +
            $v.x
    }};
}

// pub fn flat_idx_to_vector!(idx: usize, grid_dim: USizeVec3) -> USizeVec3 {
//     USizeVec3::new(
//         idx % grid_dim.x,
//         (idx / grid_dim.x) % grid_dim.y,
//         idx / (grid_dim.x * grid_dim.y),
//     )
// }
//
// pub fn vector_to_flat_idx!(v: USizeVec3, grid_dim: USizeVec3) -> usize {
//     v.z * grid_dim.x * grid_dim.y +
//         v.y * grid_dim.x +
//         v.x
// }

/// Computes `e^(i*theta)`, returning a complex number in polar coordinates.
#[macro_export]
macro_rules! e_i {
    ($theta: expr) => {
        GpuComplexPolar {
            r: 1.,
            theta: $theta
        }
    };
}

/// A complex number in polar coordinates
#[derive(Copy, Clone, Pod, Zeroable, Default)]
#[repr(C)]
pub struct GpuComplexPolar {
    pub r: f32,
    pub theta: f32
}

impl GpuComplexPolar {
    /// Raise this complex number to power `n` using DeMoivre's Theorem
    pub fn powf(self, n: f32) -> Self {
        Self {
            r: self.r.powf(n),
            theta: self.theta * n,
        }
    }
}

impl Into<Vec2> for GpuComplexPolar {
    fn into(self) -> Vec2 {
        let (im_n, re_n) = Float::sin_cos(self.theta);
        self.r * Vec2::new(re_n, im_n)
    }
}

impl From<Vec2> for GpuComplexPolar {
    fn from(vec: Vec2) -> Self {
        let theta = Float::atan2(vec.y, vec.x);
        let r = vec.length();
        Self { r, theta }
    }
}

impl core::ops::AddAssign for GpuComplexPolar {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs
    }
}

impl core::ops::Add for GpuComplexPolar {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        let self_v: Vec2 = self.into();
        let rhs_v: Vec2 = rhs.into();

        Self::from(self_v + rhs_v)
    }
}

impl core::ops::MulAssign<f32> for GpuComplexPolar {
    fn mul_assign(&mut self, rhs: f32) {
        *self = *self * rhs
    }
}

impl core::ops::Mul<f32> for GpuComplexPolar {
    type Output = GpuComplexPolar;
    fn mul(self, rhs: f32) -> Self::Output {
        Self {
            r: self.r * rhs,
            theta: self.theta,
        }
    }
}