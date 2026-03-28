use spirv_builder::{SpirvBuilder, SpirvMetadata};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let compile_result = SpirvBuilder::new("../shader_crate/", "spirv-unknown-vulkan1.1")
        .spirv_metadata(SpirvMetadata::Full)
        .build()?;
    let spv_path = compile_result.module.unwrap_single();
    println!("cargo::rustc-env=shader_crate.spv={}", spv_path.display());
    println!("cargo:rerun-if-changed=../shader_crate/");
    Ok(())
}