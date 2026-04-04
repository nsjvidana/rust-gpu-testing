#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("The shader is uninitialized. Maybe you tried reading its output while it was uninitialized.")]
    ShaderIsUninitialized,
    #[error("Buffer size cannot be zero")]
    BufferSizeZero,
    #[error("InvalidBinding: group {bind_group_id}, binding {binding}\n max bindings {max_bindings}, max bind groups {max_bind_groups}")]
    InvalidBinding {
        bind_group_id: u32,
        binding: u32,
        max_bindings: u32,
        max_bind_groups: u32,
    },
    #[error("Shader {shader_name} has a None bind group")]
    UndefinedBindGroups {
        shader_name: String
    },
    #[error("Shader {shader_name} bind group {bind_group_id} has a None binding")]
    UndefinedBindings {
        shader_name: String,
        bind_group_id: u32,
    },
    #[error("Attempted downloading a buffer that wasn't configured for downloading")]
    BufferCannotBeDownloaded
}