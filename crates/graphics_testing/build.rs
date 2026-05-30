use khal_builder::KhalBuilder;

fn main() {
    let output_dir = "shaders-spirv";
    KhalBuilder::from_dependency("shader_crate", true).build(output_dir)
}