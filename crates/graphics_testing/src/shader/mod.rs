use crate::error::ObjectError;
use glamx::Vec3;
use kiss3d::prelude::SceneNode3d;
use kiss3d::procedural::{IndexBuffer, RenderMesh};
use rapier3d::geometry::MeshConverter;
use rapier3d_meshloader::*;
use std::path::Path;
use rapier3d::parry::shape::{Ball, SharedShape};

pub mod run_fdtd1;
pub mod run_fdtd2;

#[derive(Default)]
pub struct ImportedObjects<Material: Default> {
    pub shapes: Vec<ImportedShape>,
    pub materials: Vec<Material>,
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
            load_from_path(path, &MeshConverter::TriMesh, Vec3::ONE);
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
            .map(|obj| {
                scene.add_render_mesh(
                    RenderMesh::new(
                        obj.raw_mesh.vertices.iter().map(|v| Vec3::from_array(*v)).collect(),
                        Some(obj.raw_mesh.normals.iter().map(|v| Vec3::from_array(*v)).collect()),
                        None,
                        Some(IndexBuffer::Unified(obj.raw_mesh.faces.clone()))
                    ),
                    Vec3::ONE
                )
            })
            .collect::<Vec<_>>();
        self.scene_nodes.extend(nodes);

        self.shapes.extend(objs.into_iter().map(|v| ImportedShape::Mesh(v)));

        Ok(())
    }

    pub fn push_ball(&mut self, name: String, shape: Ball, mat: Material, scene: &mut SceneNode3d) {
        self.shapes.push(ImportedShape::Generic(name, SharedShape::new(shape)));
        self.materials.push(mat);
        self.scene_nodes.push(scene.add_sphere(shape.radius));
    }
}

pub enum ImportedShape {
    Mesh(LoadedShape),
    Generic(String, SharedShape),
}

impl ImportedShape {
    pub fn get_shape(&self) -> &SharedShape {
        match self {
            ImportedShape::Mesh(lshape) => &lshape.shape,
            ImportedShape::Generic(_, shape) => shape
        }
    }

    pub fn get_name(&self) -> &String {
        match self {
            ImportedShape::Mesh(lshape) => &lshape.raw_mesh.name,
            ImportedShape::Generic(name, _) => name,
        }
    }
}