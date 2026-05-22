use crate::error::ObjectError;
use rapier3d::geometry::MeshConverter;
use rapier3d_meshloader::*;
use std::path::Path;

pub mod run_fdtd1;
pub mod run_fdtd2;

#[derive(Default)]
pub struct ImportedObjects<Material: Default> {
    pub shapes: Vec<LoadedShape>,
    pub materials: Vec<Material>,
}

impl<Material: Default> ImportedObjects<Material> {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add object shapes from a path. Supported file formats are: `.stl`, `.dae`, `.obj`
    pub fn extend_from_path(&mut self, path: impl AsRef<Path>) -> Result<(), Vec<ObjectError>> {
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

        let mut objs = shapes.into_iter()
            .map(|s| s.unwrap())
            .collect::<Vec<_>>();
        self.shapes.append(&mut objs);
        Ok(())
    }
}