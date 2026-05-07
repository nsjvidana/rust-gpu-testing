use glam::Vec2;
use khal::backend::GpuBackend;
use khal::Shader;
use shader_crate::fdtd2::{Fdtd2, MaterialConstants2};

#[derive(Shader)]
struct GpuKernels {
    fdtd2: Fdtd2
}

pub async fn run_fdtd2(backend: &GpuBackend) {

}

pub struct ElectricMaterial {
    /// Relative Magnetic Permeability (X & Y component of tensor diagonal)
    pub mu_r: Vec2,
    /// Relative Electric Permittivity (Z component of tensor diagonal)
    pub eps_r_z: f32,
    /// Refractive Index
    pub n: f32,
    /// Impedance
    pub impedance: f32,
}

impl ElectricMaterial {
    /// Speed of EM wave in free space
    pub const C_0: f32 = 299792458.0;

    pub fn to_gpu(self, dt: f32) -> MaterialConstants2 {
        let c_0_dt = Self::C_0 * dt;
        MaterialConstants2 {
            h_update_coeff: Vec2::new(
                -c_0_dt / self.mu_r.x,
                c_0_dt / self.mu_r.y,
            ),
            en_z_update_coeff: 1. / self.eps_r_z
        }
    }
}