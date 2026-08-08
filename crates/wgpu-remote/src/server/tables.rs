//! ID → wgpu-handle storage.
//!
//! One table per resource type. Newtype IDs prevent cross-table confusion at
//! compile time; `HashMap` is fine until we have a profile-driven reason to
//! switch to `slab` or a typed arena.

use std::collections::HashMap;

use crate::protocol::ids::{
    BindGroupId, BindGroupLayoutId, BufferId, ComputePipelineId, PipelineLayoutId,
    RenderPipelineId, SamplerId, ShaderModuleId, TextureId, TextureViewId,
};

#[derive(Default)]
pub struct ResourceTables {
    pub buffers: HashMap<BufferId, wgpu::Buffer>,
    pub textures: HashMap<TextureId, wgpu::Texture>,
    pub texture_views: HashMap<TextureViewId, wgpu::TextureView>,
    pub samplers: HashMap<SamplerId, wgpu::Sampler>,
    pub shader_modules: HashMap<ShaderModuleId, wgpu::ShaderModule>,
    pub bind_group_layouts: HashMap<BindGroupLayoutId, wgpu::BindGroupLayout>,
    pub bind_groups: HashMap<BindGroupId, wgpu::BindGroup>,
    pub pipeline_layouts: HashMap<PipelineLayoutId, wgpu::PipelineLayout>,
    pub render_pipelines: HashMap<RenderPipelineId, wgpu::RenderPipeline>,
    pub compute_pipelines: HashMap<ComputePipelineId, wgpu::ComputePipeline>,
}
