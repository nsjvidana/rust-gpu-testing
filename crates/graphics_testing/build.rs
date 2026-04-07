use khal_builder::KhalBuilder;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let shader_crate = "../shader_crate";
    let output_dir = "shaders-spirv";

    KhalBuilder::new(shader_crate, true)
        .build(output_dir);

    Ok(())
}