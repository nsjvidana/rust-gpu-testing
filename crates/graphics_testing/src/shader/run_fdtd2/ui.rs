use crate::error::ObjectError;
use crate::prelude::GpuResult;
use crate::shader::{parry3d, parrymath};
use crate::shader::run_fdtd2::{ElectricMaterial2, FdtdData2, FdtdGrid2, GaussianPulse2};
use crate::shader::ImportedObjects;
use crate::util::draw_bb;
use glam::USizeVec3;
use itertools::izip;
use kiss3d::egui;
use kiss3d::prelude::*;
use shader_crate::flat_idx_to_vector;
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
    /// The `[min, max]` of the FDTD grid's bounds. Includes the Z-level of the 2D grid
    pub grid_bb: [Vec3; 2],
    pub cell_positions: Vec<Vec3>,
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
            cell_positions: vec![],
            alpha_threshold,
        }
    }

    pub async fn render_loop(
        &mut self,
        data: &mut FdtdData2,
        mut callback: impl AsyncFnMut(&mut Window, &mut FdtdData2, &mut SimulationControlUi2) -> GpuResult<()>
    ) -> GpuResult<()> {
        self.update_grid_bb();
        self.update_cell_positions(&data.grid);

        while self.window.render_3d(&mut self.scene, &mut self.camera).await {
            if self.simulation_control_ui.just_started {
                self.update_simulation_data(data);
            }
            callback(&mut self.window, data, &mut self.simulation_control_ui).await?;

            if self.import_ui.import_clicked {
                self.object_explorer_ui.import_mesh(&mut self.scene, &self.import_ui.file_path, self.import_ui.material)
                    .unwrap();
                self.update_grid_bb();
            }

            let curr_max_en_mag = data.grid.cells.iter()
                .map(|c| c.en_z)
                .max_by(|a, b| a.total_cmp(b));
            if let Some(curr_max_en_mag) = curr_max_en_mag {
                if curr_max_en_mag > self.max_en_value {
                    println!("New max En magnitude: {}", curr_max_en_mag);
                    self.max_en_value = curr_max_en_mag;
                }
            }
            self.render_simulation(&data);

            self.egui_windows();
        }
        Ok(())
    }

    pub fn render_simulation(
        &mut self,
        data: &FdtdData2,
    ) {
        let cell_diagonal_len = data.grid.cell_size.length();
        for (c, pos) in data.grid.cells.iter()
            .zip(self.cell_positions.iter())
        {
            // Drawing En field
            let alpha = c.en_z.abs() / self.max_en_value;
            if alpha > self.alpha_threshold {
                let color = self.en_color.with_alpha(alpha);
                let line_len = alpha * cell_diagonal_len * c.en_z.signum();
                self.window.draw_line(*pos, pos + Vec3::new(0., 0., line_len), color, 2., false);
            }
        }

        draw_bb(&mut self.window, self.grid_bb, WHITE, 2., false);
    }

    pub fn update_cell_positions(&mut self, grid: &FdtdGrid2) {
        let cell_size3 = Vec3::from((grid.cell_size, 0.));
        let n_cells3 = USizeVec3::from((grid.n_cells.as_usizevec2(), 1));
        let offset = self.grid_bb[0] +
            Vec3::new(0., 0., self.simulation_control_ui.grid_z_level);
        self.cell_positions = (0..grid.cells.len())
            .map(|i| {
                let i3 = flat_idx_to_vector!(i, n_cells3, USizeVec3);
                i3.as_vec3() * cell_size3 + offset
            })
            .collect();
    }

    pub fn update_grid_bb(&mut self) {
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
        min.z = self.simulation_control_ui.grid_z_level;
        max.z = self.simulation_control_ui.grid_z_level;
        self.grid_bb = [min, max];
    }

    pub fn egui_windows(&mut self) {
        self.window.draw_ui(|ctx| {
            egui::Window::new("Import Mesh")
                .show(ctx, |ui| self.import_ui.ui(ui));
            // TODO: detect change in this UI
            egui::Window::new("Object Explorer")
                .show(ctx, |ui| self.object_explorer_ui.ui(ui));
            egui::Window::new("Simulation Control")
                .show(ctx, |ui| self.simulation_control_ui.ui(ui));
        });
    }

    /// Updates parameters of the simulation with the info typed into the egui UIs.
    pub fn update_simulation_data(&self, data: &mut FdtdData2) {
        let SimulationControlUi2 {
            source_max_frequency,
            ..
        } = &self.simulation_control_ui;
        let ImportedObjects {
            shapes: obj_shapes,
            scene_nodes: obj_nodes,
            materials: obj_mats,
            ..
        } = &self.object_explorer_ui.imported_objects;

        data.materials.truncate(1);
        data.prepare_materials();
        data.materials.extend_from_slice(&obj_mats);

        let pulse = GaussianPulse2::from_max_frequency(*source_max_frequency, 1.);
        data.set_source(pulse, 10);

        // TODO: let user edit these hard-coded stability values
        data.min_wavelength(*source_max_frequency, 20)
            .cfl_condition(3.);

        // Update grid dimensions & grid cells to encompass all objects
        let [min, max] = self.grid_bb;
        let aabb_dimensions = max - min;
        data.grid.n_cells = (aabb_dimensions.xy() / data.grid.cell_size).ceil().as_uvec2();
            data.update_cells();
        let n_cellsi = data.grid.n_cells.as_ivec2();

        // Discretize objects for grid cells
        let voxels = {
            let mut coords = Vec::new();
            for y in 0..n_cellsi.y {
                for x in 0..n_cellsi.x {
                    coords.push(parrymath::IVector::new(x, y, 0));
                }
            }
            parry3d::shape::Voxels::new(
                parrymath::Vec3::new(data.grid.cell_size.x, data.grid.cell_size.y, 0.),
                coords.as_slice()
            )
        };
        let vox_shape = parry3d::shape::Cuboid::new(voxels.voxel_size() / 2.);
        for (i, vox) in voxels.voxels().enumerate() {
            println!("{}", vox.grid_coords);
            let vox_pose = parrymath::Pose::from_translation(vox.center);
            for (obj, node) in izip!(obj_shapes, obj_nodes) {
                let obj_translation = parrymath::Vector::from_array((node.position() - self.grid_bb[0]).to_array());
                let obj_pose = parrymath::Pose::from_translation(obj_translation);
                let hit = parry3d::query::intersection_test(&vox_pose, &vox_shape, &obj_pose, &*obj.shape)
                    .is_ok_and(|b| b);
                if hit {
                    data.grid.cells[i].material_i = i as u32 + 1;
                }
            }
        }

        // TODO: dielectric smoothing. scirs2-ndimage crate can help with this.
    }
}

#[derive(Default)]
pub struct SimulationControlUi2 {
    pub source_max_frequency: f32,
    pub grid_z_level: f32,
    pub started: bool,
    pub paused: bool,
    pub just_started: bool,
    pub needs_reset: bool,
}

impl SimulationControlUi2 {
    pub fn ui(&mut self, ui: &mut egui::Ui) {
        ui.label("Gaussian Pulse Max Frequency:");
        ui.add(
            egui::DragValue::new(&mut self.source_max_frequency).speed(0.5).range(1e-20..=f32::MAX)
        );

        ui.label("Grid Z Level:");
        ui.add(egui::DragValue::new(&mut self.grid_z_level).speed(0.01));

        ui.horizontal(|ui| {
            let prev_started = self.started;
            self.started |= ui.selectable_label(self.started, "Start").clicked();
            self.just_started = !prev_started && self.started;

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

    pub fn reset_buttons(&mut self) {
        self.started = false;
        self.paused = false;
        self.just_started = false;
        self.needs_reset = false;
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
