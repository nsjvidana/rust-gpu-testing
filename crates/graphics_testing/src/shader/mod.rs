use crate::error::ObjectError;
use kiss3d::prelude::SceneNode3d;
use rapier3d::geometry::MeshConverter;
use rapier3d_meshloader::*;
use std::path::Path;
use kiss3d::procedural::{IndexBuffer, RenderMesh};
pub use rapier3d::parry as parry3d;
pub use parry3d::math as parrymath;

pub mod run_fdtd1;
pub mod run_fdtd2;

#[derive(Default)]
pub struct ImportedObjects<Material: Default> {
    pub shapes: Vec<LoadedShape>,
    pub materials: Vec<Material>,
    // TODO: remove this
    pub scene_nodes: Vec<SceneNode3d>,
}

impl<Material: Default> ImportedObjects<Material> {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add object shapes from a path. Supported file formats are: `.stl`, `.dae`, `.obj`
    pub fn extend_from_path(&mut self, path: impl AsRef<Path>, scene: &mut SceneNode3d) -> Result<(), Vec<ObjectError>> {
        // TODO: use MeshConverter::ConvexDecomposition / MeshConverter::ConvexDecompositionWithParams()
        let result =
            load_from_path(path, &MeshConverter::TriMesh, rapier3d::math::Vec3::ONE);
        if result.is_err() {
            return Err(vec![ObjectError::MeshLoaderError(result.err().unwrap())])
        }

        let shapes = result.unwrap();
        let has_err = shapes.iter()
            .any(|r| r.is_err());
        if has_err {
            return Err(
                shapes.into_iter()
                    .filter_map(|r| r.err()
                        .map(|e| ObjectError::MeshConversionError(e))
                    )
                    .collect()
            );
        }

        let objs = shapes.into_iter()
            .map(|s| s.unwrap())
            .collect::<Vec<_>>();

        let nodes = objs.iter()
            .map(|o| {
                scene.add_render_mesh(
                    RenderMesh::new(
                        o.raw_mesh.vertices.iter().map(|v| glam::Vec3::from_array(*v)).collect(),
                        Some(o.raw_mesh.normals.iter().map(|v| glam::Vec3::from_array(*v)).collect()),
                        None,
                        Some(IndexBuffer::Unified(o.raw_mesh.faces.clone()))
                    ),
                    glam::Vec3::ONE
                )
            })
            .collect::<Vec<_>>();
        self.scene_nodes.extend(nodes);

        self.shapes.extend(objs);

        Ok(())
    }
}