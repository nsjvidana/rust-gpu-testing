use std::hash::Hash;
use crate::prelude::GpuResult;
use crate::shader::run_fdtd2::{ElectricMaterial2, FdtdData2};
use crate::shader::ImportedObjects;
use itertools::izip;
use kiss3d::egui;
use kiss3d::prelude::*;
use kiss3d::procedural::{IndexBuffer, RenderMesh};
use rapier3d_meshloader::LoadedShape;
use std::path::PathBuf;

pub struct TestbedWindow2 {
    pub window: Window,
    pub scene: SceneNode3d,
    pub camera: OrbitCamera3d,
    pub object_nodes: Vec<SceneNode3d>,

    pub import_ui: ImportUi2,
    pub object_explorer_ui: ObjectExplorerUi,

    pub max_en_value: f32,
    pub en_color: Color,
    pub alpha_threshold: f32,
}

impl TestbedWindow2 {
    pub async fn new(name: &str, alpha_threshold: f32) -> Self {
        let window = Window::new(name).await;
        let mut camera = OrbitCamera3d::default();
        camera.set_up_axis_dir(Vec3::Z);
        let mut scene = SceneNode3d::empty();
        scene
            .add_light(Light::point(100.0))
            .set_position(Vec3::new(0.0, 2.0, -2.0));
        Self { 
            window,
            scene,
            camera,
            object_nodes: vec![],

            import_ui: ImportUi2::default(),
            object_explorer_ui: ObjectExplorerUi::default(),

            max_en_value: 0.,
            en_color: RED,
            alpha_threshold,
        }
    }
    
    pub fn add_shape_as_object(&mut self, shape: &LoadedShape) {
        let mesh = &shape.raw_mesh;

        let node = self.scene.add_render_mesh(
            RenderMesh::new(
                mesh.vertices.iter().map(|v| Vec3::from_array(*v)).collect(),
                Some(mesh.normals.iter().map(|v| Vec3::from_array(*v)).collect()),
                None,
                Some(IndexBuffer::Unified(mesh.faces.clone()))
            ),
            Vec3::ONE
        );
        self.object_nodes.push(node);
    }

    pub async fn render_loop(
        &mut self,
        data: &mut FdtdData2,
        mut callback: impl AsyncFnMut(&mut Window, &mut FdtdData2) -> GpuResult<()>
    ) -> GpuResult<()> {
        while self.window.render_3d(&mut self.scene, &mut self.camera).await {
            callback(&mut self.window, data).await?;

            if self.import_ui.import_clicked {
                data.import_mesh(&mut self.scene, &self.import_ui.file_path, self.import_ui.material)
                    .unwrap();
            }
            let curr_max_en_mag = data.grid.cells.iter()
                .map(|c| c.en_z)
                .max_by(|a, b| a.total_cmp(b))
                .unwrap();
            if curr_max_en_mag > self.max_en_value {
                println!("New max En magnitude: {}", curr_max_en_mag);
                self.max_en_value = curr_max_en_mag;
            }
            self.render_simulation(&data);

            self.egui_windows(data);
        }
        Ok(())
    }

    pub fn render_simulation(
        &mut self,
        data: &FdtdData2,
    ) {
        todo!()
    }

    pub fn egui_windows(&mut self, data: &mut FdtdData2) {
        self.window.draw_ui(|ctx| {
            egui::Window::new("Import Mesh")
                .show(ctx, |ui| self.import_ui.ui(ui));
            egui::Window::new("Object Explorer")
                .show(ctx, |ui| self.object_explorer_ui.explorer_ui(data, &mut self.object_nodes, ui));
        });
    }
}

#[derive(Default)]
pub struct ImportUi2 {
    pub file_path: PathBuf,
    pub file_path_string: String,
    pub material: ElectricMaterial2,
    pub import_clicked: bool,
}

impl ImportUi2 {
    pub fn ui(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            if ui.text_edit_singleline(&mut self.file_path_string).changed() {
                self.file_path = PathBuf::from(self.file_path_string.as_str());
            }
            if ui.button("Browse").clicked() {
                if let Some(file) = rfd::FileDialog::new()
                    .add_filter("3D Mesh", &["stl", "dae", "obj"])
                    .pick_file()
                {
                    self.file_path = file;
                    self.file_path_string = self.file_path.as_path().to_string_lossy().to_string();
                }
            }
        });

        material_ui(&mut self.material, ui);

        self.import_clicked = ui.button("Import Mesh").clicked();
    }
}

#[derive(Default)]
pub struct ObjectExplorerUi;

impl ObjectExplorerUi {
    pub fn explorer_ui(&mut self, data: &mut FdtdData2, object_nodes: &mut Vec<SceneNode3d>, ui: &mut egui::Ui) {
        let ImportedObjects {
            shapes,
            materials,
            ..
        } = &mut data.imported_objects;
        let speed = data.grid.cell_size.min_element();

        for (node, shape, mat) in izip!(object_nodes, shapes, materials)
        {
            ui.collapsing(&shape.raw_mesh.name, |ui| {
                pose_ui(node, speed, ui);
                material_ui(mat, ui);
            });
        }
    }
}

fn pose_ui(node: &mut SceneNode3d, drag_speed: f32, ui: &mut egui::Ui) {
    ui.collapsing("Transform", |ui| {
        let mut pos = node.position();
        let mut changed = false;
        ui.label("Translation:");
        ui.indent(0, |ui| {
            changed |= ui.add(egui::DragValue::new(&mut pos.x).speed(drag_speed)).changed();
            changed |= ui.add(egui::DragValue::new(&mut pos.y).speed(drag_speed)).changed();
            changed |= ui.add(egui::DragValue::new(&mut pos.z).speed(drag_speed)).changed();
        });
        if changed {
            println!("{}", pos);
            node.set_position(pos);
        }
    });
}

fn material_ui(material: &mut ElectricMaterial2, ui: &mut egui::Ui) {
    ui.collapsing("Material Properties", |ui| {
        ui.label("Relative Permeability (Tensor Diagonal):");
        ui.indent("mu_r_indent", |ui| ui.horizontal(|ui| {
            ui.add(egui::DragValue::new(&mut material.mu_r.x).speed(0.01).range(0.0..=f32::MAX));
            ui.add(egui::DragValue::new(&mut material.mu_r.y).speed(0.01).range(0.0..=f32::MAX));
        }));

        ui.label("Relative Permittivity:");
        ui.indent("eps_r_indent", |ui|
            ui.add(egui::DragValue::new(&mut material.eps_r_z).speed(0.01).range(0.0..=f32::MAX))
        );

        if ui.button("Reset").clicked() {
            *material = ElectricMaterial2::FREE_SPACE;
        }
    });
}
