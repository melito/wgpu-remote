//! Convenience re-exports symmetrical to `wgpu_remote::client::prelude`, but
//! specialized to the in-memory transport. Use in tests + examples that don't
//! want to set up TLS:
//!
//! ```ignore
//! use wgpu_remote_tests::prelude::in_memory::*;
//! let (client_conn, server_conn) = pair();
//! let instance = Instance::new(Client::new(client_conn));
//! ```

pub mod in_memory {
    use crate::InMemoryConnection;

    pub use crate::{InMemoryConnection as Connection, InMemoryTransport, pair};

    pub type Client = wgpu_remote::client::Client<InMemoryConnection>;
    pub type Instance = wgpu_remote::client::Instance<InMemoryConnection>;
    pub type Adapter = wgpu_remote::client::Adapter<InMemoryConnection>;
    pub type Device = wgpu_remote::client::Device<InMemoryConnection>;
    pub type Queue = wgpu_remote::client::Queue<InMemoryConnection>;

    pub type Buffer = wgpu_remote::client::Buffer<InMemoryConnection>;
    pub type Texture = wgpu_remote::client::Texture<InMemoryConnection>;
    pub type TextureView = wgpu_remote::client::TextureView<InMemoryConnection>;
    pub type Sampler = wgpu_remote::client::Sampler<InMemoryConnection>;
    pub type ShaderModule = wgpu_remote::client::ShaderModule<InMemoryConnection>;
    pub type BindGroupLayout = wgpu_remote::client::BindGroupLayout<InMemoryConnection>;
    pub type BindGroup = wgpu_remote::client::BindGroup<InMemoryConnection>;
    pub type PipelineLayout = wgpu_remote::client::PipelineLayout<InMemoryConnection>;
    pub type ComputePipeline = wgpu_remote::client::ComputePipeline<InMemoryConnection>;
    pub type RenderPipeline = wgpu_remote::client::RenderPipeline<InMemoryConnection>;

    pub type CommandEncoder = wgpu_remote::client::CommandEncoder<InMemoryConnection>;
    pub type ComputePass<'enc> = wgpu_remote::client::ComputePass<'enc, InMemoryConnection>;
    pub type RenderPass<'enc> = wgpu_remote::client::RenderPass<'enc, InMemoryConnection>;
    pub type RenderPassDescriptor<'a> =
        wgpu_remote::client::RenderPassDescriptor<'a, InMemoryConnection>;
    pub type ColorAttachment<'a> = wgpu_remote::client::ColorAttachment<'a, InMemoryConnection>;
    pub type DepthStencilAttachment<'a> =
        wgpu_remote::client::DepthStencilAttachment<'a, InMemoryConnection>;
    pub use wgpu_remote::client::{CommandBuffer, ClientError};

    pub use wgpu_remote::client::descriptors::{
        BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindingResource,
        BufferBinding, BufferDescriptor, ComputePipelineDescriptor, FragmentState,
        PipelineLayoutDescriptor, RenderPipelineDescriptor, SamplerDescriptor,
        ShaderModuleDescriptor, ShaderSource, TextureDescriptor, TextureViewDescriptor,
        VertexBufferLayout, VertexState,
    };
    pub use wgpu_remote::client::{
        BindGroupLayoutEntry, BindingType, BufferBindingType, BufferUsages, ShaderStages,
    };
}
