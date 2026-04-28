//! Owned, `Serialize`-able descriptor types for v1.
//!
//! Strategy: `wgpu-types` already exposes most descriptors as generic over
//! their label type (`BufferDescriptor<L>`, `TextureDescriptor<L, V>`, etc.)
//! with `serde` derives behind the `serde` feature. We reuse those directly
//! by picking owned label/view types — no hand-mirroring needed.
//!
//! For descriptors that reference *other resources* (bind groups, pipeline
//! layouts, render/compute pipelines), the original types live in the `wgpu`
//! crate and carry `&BindGroupLayout`, `&Buffer`, etc. — references that
//! can't cross a process boundary. Those we mirror by hand, swapping the
//! references for our [`crate::ids`] handles.

use serde::{Deserialize, Serialize};
use wgpu_types::{
    BindGroupLayoutEntry, BufferAddress, BufferSize, ColorTargetState, DepthStencilState,
    MultisampleState, PrimitiveState, PushConstantRange, ShaderStages, TextureAspect,
    TextureFormat, TextureUsages, TextureViewDimension, VertexAttribute, VertexStepMode,
};

use crate::ids::{
    BindGroupLayoutId, BufferId, PipelineLayoutId, SamplerId, ShaderModuleId, TextureViewId,
};

// -- Descriptors reused from wgpu-types via type aliases ---------------------

/// `wgpu_types::BufferDescriptor<Option<String>>`.
pub type BufferDescriptor = wgpu_types::BufferDescriptor<Option<String>>;

/// `wgpu_types::TextureDescriptor<Option<String>, Vec<TextureFormat>>`.
pub type TextureDescriptor = wgpu_types::TextureDescriptor<Option<String>, Vec<TextureFormat>>;

/// Hand-mirrored — `wgpu_types::TextureViewDescriptor` doesn't derive `serde`.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TextureViewDescriptor {
    pub label: Option<String>,
    pub format: Option<TextureFormat>,
    pub dimension: Option<TextureViewDimension>,
    pub usage: Option<TextureUsages>,
    pub aspect: TextureAspect,
    pub base_mip_level: u32,
    pub mip_level_count: Option<u32>,
    pub base_array_layer: u32,
    pub array_layer_count: Option<u32>,
}

/// `wgpu_types::SamplerDescriptor<Option<String>>`.
pub type SamplerDescriptor = wgpu_types::SamplerDescriptor<Option<String>>;

/// `wgpu_types::CommandEncoderDescriptor<Option<String>>`.
pub type CommandEncoderDescriptor = wgpu_types::CommandEncoderDescriptor<Option<String>>;

// -- Descriptors that reference resources by ID (hand-mirrored) --------------

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BindGroupLayoutDescriptor {
    pub label: Option<String>,
    pub entries: Vec<BindGroupLayoutEntry>,
}

/// Mirror of `wgpu::BindingResource` with IDs in place of references.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum BindingResource {
    Buffer {
        buffer: BufferId,
        offset: BufferAddress,
        size: Option<BufferSize>,
    },
    BufferArray(Vec<BufferBinding>),
    Sampler(SamplerId),
    SamplerArray(Vec<SamplerId>),
    TextureView(TextureViewId),
    TextureViewArray(Vec<TextureViewId>),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BufferBinding {
    pub buffer: BufferId,
    pub offset: BufferAddress,
    pub size: Option<BufferSize>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BindGroupEntry {
    pub binding: u32,
    pub resource: BindingResource,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BindGroupDescriptor {
    pub label: Option<String>,
    pub layout: BindGroupLayoutId,
    pub entries: Vec<BindGroupEntry>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PipelineLayoutDescriptor {
    pub label: Option<String>,
    pub bind_group_layouts: Vec<BindGroupLayoutId>,
    pub push_constant_ranges: Vec<PushConstantRange>,
}

/// Source for a shader module. WGSL is the v1 default; the SPIR-V variant is
/// included so apps that already produce SPIR-V (e.g. via `naga` ahead of time)
/// don't need a workaround.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ShaderSource {
    Wgsl(String),
    SpirV(Vec<u32>),
    Glsl {
        shader: String,
        stage: ShaderStages,
        defines: Vec<(String, String)>,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ShaderModuleDescriptor {
    pub label: Option<String>,
    pub source: ShaderSource,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VertexBufferLayout {
    pub array_stride: BufferAddress,
    pub step_mode: VertexStepMode,
    pub attributes: Vec<VertexAttribute>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VertexState {
    pub module: ShaderModuleId,
    pub entry_point: Option<String>,
    pub constants: Vec<(String, f64)>,
    pub buffers: Vec<VertexBufferLayout>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FragmentState {
    pub module: ShaderModuleId,
    pub entry_point: Option<String>,
    pub constants: Vec<(String, f64)>,
    pub targets: Vec<Option<ColorTargetState>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RenderPipelineDescriptor {
    pub label: Option<String>,
    pub layout: Option<PipelineLayoutId>,
    pub vertex: VertexState,
    pub primitive: PrimitiveState,
    pub depth_stencil: Option<DepthStencilState>,
    pub multisample: MultisampleState,
    pub fragment: Option<FragmentState>,
    pub multiview: Option<u32>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ComputePipelineDescriptor {
    pub label: Option<String>,
    pub layout: Option<PipelineLayoutId>,
    pub module: ShaderModuleId,
    pub entry_point: Option<String>,
    pub constants: Vec<(String, f64)>,
}

