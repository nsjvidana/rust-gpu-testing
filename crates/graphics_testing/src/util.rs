use glam::{UVec3, Vec3};
use kiss3d::prelude::Polyline3d;

pub fn flat_idx_to_vector(idx: u32, grid_dim: UVec3) -> UVec3 {
    UVec3::new(
        idx % grid_dim.x,
        (idx / grid_dim.x) % grid_dim.y,
        idx / (grid_dim.x * grid_dim.y),
    )
}

pub fn vector_to_flat_idx(v: UVec3, grid_dim: UVec3) -> u32 {
    v.z * grid_dim.x * grid_dim.y +
        v.y * grid_dim.x +
        v.x
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