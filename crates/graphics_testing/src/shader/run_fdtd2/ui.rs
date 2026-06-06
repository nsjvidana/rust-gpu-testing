use crate::error::ObjectError;
use crate::prelude::GpuResult;
use crate::shader::run_fdtd2::{ElectricMaterial2, FdtdData2, FdtdGrid2, GaussianPulse2};
use crate::shader::ImportedObjects;
use crate::shader::{parry3d, parrymath};
use crate::util::draw_bb;
use itertools::izip;
use kiss3d::egui;
use kiss3d::egui::Widget;
use kiss3d::prelude::*;
use glamx::*;
use shader_crate::{flat_idx_to_vector, vector_to_flat_idx};
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
    pub grid_bb_sim: [Vec3; 2],
    pub cell_positions: Vec<Vec3>,
    pub alpha_threshold: f32,
}

impl TestbedWindow2 {
    pub async fn new(name: &str, alpha_threshold: f32) -> Self {
        let window = Window::new(name).await;
        let mut camera = OrbitCamera3d::new_with_frustum(
            core::f32::consts::PI / 4.0, 1e-3, f32::MAX, Vec3::X, Vec3::ZERO
        );
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
            grid_bb_sim: [Vec3::ZERO; 2],
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

        while self.window.render_3d(&mut self.scene, &mut self.camera).await {
            if self.simulation_control_ui.just_started {
                self.update_simulation_data(data);
                self.update_cell_positions(&data.grid);
            }
            else if self.simulation_control_ui.needs_reset {
                self.max_en_value = 0.;
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

        let pos = Vec3::from((self.simulation_control_ui.soft_source_pos, self.simulation_control_ui.grid_z_level));
        self.window.draw_point(pos, RED, 10.);

        if self.simulation_control_ui.started {
            draw_bb(&mut self.window, self.grid_bb_sim, ORANGE, 2., false);
        }
    }

    pub fn update_cell_positions(&mut self, grid: &FdtdGrid2) {
        let cell_size3 = Vec3::from((grid.cell_size, 0.));
        let n_cells3 = USizeVec3::from((grid.n_cells.as_usizevec2(), 1));
        let offset = self.grid_bb_sim[0] +
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

        let mut min = Vec3::MAX;
        let mut max = Vec3::ZERO;
        for (node, shape) in izip!(scene_nodes, shapes) {
            let aabb = shape.shape.compute_local_aabb();
            let new_min = Vec3::from_array(aabb.mins.to_array()) + node.position();
            let new_max = Vec3::from_array(aabb.maxs.to_array()) + node.position();
            min = min.min(new_min);
            max = max.max(new_max);
        }
        min = min.with_xy(min.xy().min(self.simulation_control_ui.soft_source_pos));
        max = max.with_xy(max.xy().max(self.simulation_control_ui.soft_source_pos));
        if min == Vec3::MAX {
            min = Vec3::ZERO
        }
        min.z = self.simulation_control_ui.grid_z_level;
        max.z = self.simulation_control_ui.grid_z_level;
        self.grid_bb = [min, max];
    }

    pub fn egui_windows(&mut self) {
        let mut object_changed = false;
        let mut sim_changed = false;
        self.window.draw_ui(|ctx| {
            egui::Window::new("Import Mesh")
                .show(ctx, |ui| self.import_ui.ui(ui));

            egui::Window::new("Object Explorer")
                .show(ctx, |ui| object_changed = self.object_explorer_ui.ui(ui));

            egui::Window::new("Simulation Control")
                .show(ctx, |ui| sim_changed = self.simulation_control_ui.ui(ui));
        });
        if object_changed || sim_changed {
            self.update_grid_bb();
        }
    }

    /// Updates parameters of the simulation with the info typed into the egui UIs.
    pub fn update_simulation_data(&mut self, data: &mut FdtdData2) {
        let SimulationControlUi2 {
            source_max_frequency,
            source_resolution,
            stability_values2: stability,
            soft_source_pos,
            ..
        } = &self.simulation_control_ui;
        let ImportedObjects {
            shapes: obj_shapes,
            scene_nodes: obj_nodes,
            materials: obj_mats,
            ..
        } = &self.object_explorer_ui.imported_objects;

        data.grid.update_coeffs.clear();
        data.grid.cells.clear();

        data.materials.truncate(1);
        data.prepare_materials();
        data.materials.extend_from_slice(&obj_mats);

        let pulse = GaussianPulse2::from_max_frequency(*source_max_frequency, 1.);
        data.set_source(pulse, *source_resolution);

        data.min_wavelength(*source_max_frequency, stability.cells_per_wavelength)
            .cfl_condition(stability.dt_multiplier);

        // Update grid dimensions & grid cells to encompass all objects
        let cell_size = data.grid.cell_size;
        let spacer_region_offset = Vec3::from(
            (stability.spacer_region_width as f32 * cell_size, 0.)
        );
        self.grid_bb_sim = [
            self.grid_bb[0] - spacer_region_offset,
            self.grid_bb[1] + spacer_region_offset
        ];
        let bb_dimensions_sim = self.grid_bb_sim[1] - self.grid_bb_sim[0];
        data.grid.n_cells = (bb_dimensions_sim.xy() / cell_size).ceil().as_uvec2();
            data.update_cells();

        // Update source cell index
        let grid_dim3 = UVec3::from((data.grid.n_cells, 1));
        let src_pos = soft_source_pos - self.grid_bb_sim[0].xy();
        let src_cell_idx = UVec3::from(((src_pos / cell_size).as_uvec2(), 0));
        data.source_cell_idx = vector_to_flat_idx!(src_cell_idx, grid_dim3);

        let bkg_update_coeff = data.materials[0].to_gpu(data.dt);
        data.grid.update_coeffs.resize(data.grid.cells.len(), bkg_update_coeff);

        // TODO: dielectric smoothing (using averaging?)
    }
}

pub struct SimulationControlUi2 {
    pub source_max_frequency: f32,
    pub source_resolution: usize,
    pub grid_z_level: f32,
    pub stability_values2: StabilityValues2,
    // TODO: let user edit this and see it visibly in the simulation scene.
    pub soft_source_pos: Vec2,

    pub started: bool,
    pub paused: bool,
    pub just_started: bool,
    pub needs_reset: bool,
}

impl SimulationControlUi2 {
    /// Returns true if simulation parameters have been changed
    pub fn ui(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;

        ui.collapsing("Gaussian Pulse Source:", |ui| {
            ui.label("Max Frequency:");
            changed |= egui::DragValue::new(&mut self.source_max_frequency).speed(0.1).range(1e-20..=f32::MAX).ui(ui)
                .changed();
            ui.label("Resolution:");
            changed |= egui::DragValue::new(&mut self.source_resolution).speed(1).range(1..=u32::MAX).ui(ui)
                .changed();
        });

        ui.label("Grid Z Level:");
        changed |= egui::DragValue::new(&mut self.grid_z_level).speed(0.01).ui(ui).changed();

        changed |= self.stability_values2.ui(ui);

        ui.horizontal(|ui| {
            ui.label("Soft Source Position:");
            changed |= egui::DragValue::new(&mut self.soft_source_pos.x).speed(0.01).ui(ui).changed();
            changed |= egui::DragValue::new(&mut self.soft_source_pos.y).speed(0.01).ui(ui).changed();
        });

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

        changed
    }

    pub fn reset_buttons(&mut self) {
        self.started = false;
        self.paused = false;
        self.just_started = false;
        self.needs_reset = false;
    }
}

impl Default for SimulationControlUi2 {
    fn default() -> Self {
        Self {
            source_max_frequency: 2.4e6,
            source_resolution: 10,
            grid_z_level: 0.,
            stability_values2: StabilityValues2::default(),
            soft_source_pos: Vec2::ZERO,
            started: false,
            paused: false,
            just_started: false,
            needs_reset: false,
        }
    }
}

pub struct StabilityValues2 {
    pub cells_per_wavelength: usize,
    pub dt_multiplier: f32,
    pub spacer_region_width: usize,
    pub material_smoothing_resolution: u32,
}

impl StabilityValues2 {
    pub fn ui(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.collapsing("Stability Parameters", |ui| {
            let StabilityValues2 {
                cells_per_wavelength,
                dt_multiplier,
                spacer_region_width,
                material_smoothing_resolution,
            } = self;
            ui.label("Cells per Wavelength:");
            changed |= egui::DragValue::new(cells_per_wavelength).speed(1).ui(ui).changed();
            ui.label("Dt Multiplier:");
            changed |= egui::DragValue::new(dt_multiplier)
                .range((1. + f32::MIN_POSITIVE)..=f32::MAX).speed(0.01).ui(ui).changed();
            ui.label("Spacer Region Width:");
            changed |= egui::DragValue::new(spacer_region_width).speed(1).ui(ui).changed();
            ui.label("Material Smoothing Resolution:");
            changed |= egui::DragValue::new(material_smoothing_resolution)
                .range(1..=u32::MAX).speed(1).ui(ui).changed();
        });
        changed
    }
}

impl Default for StabilityValues2 {
    fn default() -> Self {
        Self {
            cells_per_wavelength: 10,
            dt_multiplier: 2.,
            spacer_region_width: 10,
            material_smoothing_resolution: 1,
        }
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
    /// Returns if an object has been changed in some way
    pub fn ui(&mut self, ui: &mut egui::Ui) -> bool {
        let ImportedObjects {
            scene_nodes,
            shapes,
            materials,
            ..
        } = &mut self.imported_objects;

        let mut objects_were_changed = false;
        for (node, shape, mat) in izip!(scene_nodes, shapes, materials)
        {
            let mut pose  = node.local_transformation();
            let mut pose_changed = false;
            let mut mat_changed = false;
            ui.collapsing(&shape.raw_mesh.name, |ui| {
                pose_changed = pose_ui(&mut pose, 0.01, ui);
                mat_changed = material_ui(mat, ui);
            });
            if pose_changed {
                node.set_position(pose.translation);
                node.set_rotation(pose.rotation);
            }
            objects_were_changed |= pose_changed || mat_changed;
        }

        objects_were_changed
    }

    pub fn import_mesh(&mut self, scene: &mut SceneNode3d, path: impl AsRef<Path>, material: ElectricMaterial2) -> Result<(), Vec<ObjectError>> {
        self.imported_objects.materials.push(material);
        self.imported_objects.extend_from_path(path, scene)
    }
}

fn pose_ui(pose: &mut Pose3, drag_speed: f32, ui: &mut egui::Ui) -> bool {
    let mut changed = false;
    ui.collapsing("Transform", |ui| {
        let pos = &mut pose.translation;
        ui.label("Translation:");
        ui.indent(0, |ui| {
            changed |= ui.add(egui::DragValue::new(&mut pos.x).speed(drag_speed)).changed();
            changed |= ui.add(egui::DragValue::new(&mut pos.y).speed(drag_speed)).changed();
            changed |= ui.add(egui::DragValue::new(&mut pos.z).speed(drag_speed)).changed();
        });
    });
    changed
}

fn material_ui(material: &mut ElectricMaterial2, ui: &mut egui::Ui) -> bool {
    let mut changed = false;
    ui.collapsing("Material Properties", |ui| {
        ui.label("Relative Permeability (Tensor Diagonal):");
        ui.indent("mu_r_indent", |ui| ui.horizontal(|ui| {
            changed |= ui.add(egui::DragValue::new(&mut material.mu_r.x).speed(0.01).range(0.0..=f32::MAX))
                .clicked();
            changed |= ui.add(egui::DragValue::new(&mut material.mu_r.y).speed(0.01).range(0.0..=f32::MAX))
                .clicked();
        }));

        ui.label("Relative Permittivity:");
        ui.indent("eps_r_indent", |ui|
            changed |= ui.add(egui::DragValue::new(&mut material.eps_r_z).speed(0.01).range(0.0..=f32::MAX))
                .clicked()
        );

        if ui.button("Reset").clicked() {
            *material = ElectricMaterial2::FREE_SPACE;
            changed = true;
        }
    });
    changed
}
