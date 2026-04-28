//! Client-side ID minter.
//!
//! Every typed resource (buffer, texture, ...) has its own monotonic counter.
//! IDs are unique *within their type*, since the protocol uses typed handles
//! anyway. A `u64` is plenty — at one ID per nanosecond it'd take ~600 years
//! to wrap.

use std::sync::atomic::{AtomicU64, Ordering};

use wgpu_remote_protocol::ids::{
    BindGroupId, BindGroupLayoutId, BufferId, ComputePipelineId, PipelineLayoutId,
    RenderPipelineId, SamplerId, ShaderModuleId, TextureId, TextureViewId,
};

#[derive(Default)]
#[allow(dead_code)]
pub(crate) struct IdMinter {
    buffer: AtomicU64,
    texture: AtomicU64,
    texture_view: AtomicU64,
    sampler: AtomicU64,
    shader_module: AtomicU64,
    bind_group_layout: AtomicU64,
    bind_group: AtomicU64,
    pipeline_layout: AtomicU64,
    render_pipeline: AtomicU64,
    compute_pipeline: AtomicU64,
}

macro_rules! mint_fn {
    ($field:ident, $ty:ident, $method:ident) => {
        #[allow(dead_code)]
        pub(crate) fn $method(&self) -> $ty {
            $ty::new(self.$field.fetch_add(1, Ordering::Relaxed) + 1)
        }
    };
}

impl IdMinter {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    mint_fn!(buffer, BufferId, mint_buffer);
    mint_fn!(texture, TextureId, mint_texture);
    mint_fn!(texture_view, TextureViewId, mint_texture_view);
    mint_fn!(sampler, SamplerId, mint_sampler);
    mint_fn!(shader_module, ShaderModuleId, mint_shader_module);
    mint_fn!(bind_group_layout, BindGroupLayoutId, mint_bind_group_layout);
    mint_fn!(bind_group, BindGroupId, mint_bind_group);
    mint_fn!(pipeline_layout, PipelineLayoutId, mint_pipeline_layout);
    mint_fn!(render_pipeline, RenderPipelineId, mint_render_pipeline);
    mint_fn!(compute_pipeline, ComputePipelineId, mint_compute_pipeline);
}
