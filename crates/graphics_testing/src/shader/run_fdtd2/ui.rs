use kiss3d::prelude::*;

pub struct Testbed2 {
    pub window: Window,
    pub scene: SceneNode3d,
    pub camera: OrbitCamera3d
}

impl Testbed2 {
    pub async fn new() -> Self {
        let window = Window::new("FDTD Testbed 2D").await;
        let mut camera = OrbitCamera3d::default();
        camera.set_up_axis_dir(Vec3::Z);
        let mut scene = SceneNode3d::empty();
        scene
            .add_light(Light::point(100.0))
            .set_position(Vec3::new(0.0, 2.0, -2.0));
        Self {
            window,
            scene,
            camera
        }
    }
}