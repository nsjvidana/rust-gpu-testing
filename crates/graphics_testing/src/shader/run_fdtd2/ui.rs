use crate::error::ObjectError;
use crate::prelude::GpuResult;
use crate::shader::run_fdtd2::{ElectricMaterial2, FdtdData2};
use crate::shader::ImportedObjects;
use itertools::izip;
use kiss3d::egui;
use kiss3d::prelude::*;
use std::path::{Path, PathBuf};

pub struct TestbedWindow2 {
    pub window: Window,
    pub scene: SceneNode3d,
    pub camera: OrbitCamera3d,

    pub simulation_control_ui: SimulationControlUi2,
    pub import_ui: ImportUi2,
    pub object_explorer_ui: ObjectExplorerUi,

    pub max_en_value: f32,
    pub en_color: Color,
    pub grid_bb: [Vec3; 2],
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

            import_ui: ImportUi2::default(),
            object_explorer_ui: ObjectExplorerUi::default(),
            simulation_control_ui: SimulationControlUi2::default(),

            max_en_value: 0.,
            en_color: RED,
            grid_bb: [Vec3::ZERO; 2],
            alpha_threshold,
        }
    }

    pub async fn render_loop(
        &mut self,
        data: &mut FdtdData2,
        mut callback: impl AsyncFnMut(&mut Window, &mut FdtdData2) -> GpuResult<()>
    ) -> GpuResult<()> {
        while self.window.render_3d(&mut self.scene, &mut self.camera).await {
            callback(&mut self.window, data).await?;

            if self.import_ui.import_clicked {
                self.object_explorer_ui.import_mesh(&mut self.scene, &self.import_ui.file_path, self.import_ui.material)
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
            // self.render_simulation(&data);

            self.egui_windows();
        }
        Ok(())
    }

    pub fn render_simulation(
        &mut self,
        data: &FdtdData2,
    ) {
        todo!()
    }

    pub fn compute_grid_bb(&mut self) {
        let ImportedObjects {
            scene_nodes,
            shapes,
            ..
        } = &self.object_explorer_ui.imported_objects;

        let mut min = Vec3::ZERO;
        let mut max = Vec3::ZERO;
        for (node, shape) in izip!(scene_nodes, shapes) {
            let aabb = shape.shape.compute_local_aabb();
            let new_min = Vec3::from_array(aabb.mins.to_array()) + node.position();
            let new_max = Vec3::from_array(aabb.maxs.to_array()) + node.position();
            min = min.min(new_min);
            max = max.max(new_max);
        }
        self.grid_bb = [min, max];
    }

    pub fn egui_windows(&mut self) {
        self.window.draw_ui(|ctx| {
            egui::Window::new("Import Mesh")
                .show(ctx, |ui| self.import_ui.ui(ui));
            egui::Window::new("Object Explorer")
                .show(ctx, |ui| self.object_explorer_ui.ui(ui));
            egui::Window::new("Simulation Control")
                .show(ctx, |ui| self.simulation_control_ui.ui(ui));
        });
    }
}

#[derive(Default)]
pub struct SimulationControlUi2 {
    pub source_max_frequency: f32,
    pub started: bool,
    pub paused: bool,
    pub needs_reset: bool,
}

impl SimulationControlUi2 {
    pub fn ui(&mut self, ui: &mut egui::Ui) {
        ui.label("Gaussian Pulse Max Frequency:");
        ui.add(
            egui::DragValue::new(&mut self.source_max_frequency).speed(0.5).range(1e-20..=f32::MAX)
        );

        ui.horizontal(|ui| {
            self.started |= ui.selectable_label(self.started, "Start").clicked();
            if ui.selectable_label(self.paused, "Pause").clicked() {
                self.paused = !self.paused && self.started;
            }
            if ui.button("Reset").clicked() {
                self.started = false;
                self.paused = false;
                self.needs_reset = true;
            }
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
pub struct ObjectExplorerUi {
    pub imported_objects: ImportedObjects<ElectricMaterial2>
}

impl ObjectExplorerUi {
    pub fn ui(&mut self, ui: &mut egui::Ui) {
        let ImportedObjects {
            scene_nodes,
            shapes,
            materials,
            ..
        } = &mut self.imported_objects;

        for (node, shape, mat) in izip!(scene_nodes, shapes, materials)
        {
            ui.collapsing(&shape.raw_mesh.name, |ui| {
                pose_ui(node, 0.01, ui);
                material_ui(mat, ui);
            });
        }
    }

    pub fn import_mesh(&mut self, scene: &mut SceneNode3d, path: impl AsRef<Path>, material: ElectricMaterial2) -> Result<(), Vec<ObjectError>> {
        self.imported_objects.materials.push(material);
        self.imported_objects.extend_from_path(path, scene)
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
