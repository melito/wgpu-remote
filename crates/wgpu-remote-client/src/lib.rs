//! Client crate.
//!
//! Two layers:
//! - [`Client`]: low-level. Send any `Action`, await a `Response`.
//! - [`Instance`] / [`Adapter`] / [`Device`] / etc.: wgpu-shaped facade that
//!   delegates to the client. Existing wgpu apps swap their imports from
//!   `wgpu` to `wgpu_remote_client` and the rest of the code stays the same
//!   (modulo the parts of the wgpu surface we haven't mirrored yet).
//!
//! In v1 only the compute path is implemented. Render-pipeline / texture /
//! sampler types are stubbed in the protocol layer and will land in v1.1.

pub mod client;
pub mod encoder;
pub mod ids;
pub mod instance;
pub mod resources;

pub use client::{Client, ClientError};
pub use encoder::{
    ColorAttachment, CommandBuffer, CommandEncoder, ComputePass, DepthStencilAttachment,
    RenderPass, RenderPassDescriptor,
};
pub use instance::{Adapter, Device, Instance, Queue};
pub use resources::{
    BindGroup, BindGroupLayout, Buffer, ComputePipeline, PipelineLayout, RenderPipeline, Sampler,
    ShaderModule, Texture, TextureView,
};

// Re-export the protocol types users will pass into descriptors.
pub use wgpu_remote_protocol::descriptors;
pub use wgpu_remote_protocol::ids as protocol_ids;
pub use wgpu_types::{
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferUsages, ShaderStages,
};
