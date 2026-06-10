//! Convenience re-exports for callers that don't want to write the
//! [`Connection`](wgpu_remote_transport::Connection) generic at every type.
//!
//! Prelude flavors mirror the transports the workspace ships:
//!
//! - [`quic`]: `Buffer`, `Device`, … specialized to `QuicConnection`.
//! - [`iroh`]: same types specialized to `IrohConnection`.
//! - The in-memory equivalent lives in `wgpu-remote-tests::prelude::in_memory`
//!   (closer to where it's used).
//!
//! Typical use:
//! ```ignore
//! use wgpu_remote_client::prelude::quic::*;
//! let endpoint = QuicEndpoint::client(cert)?;
//! let conn = endpoint.connect(addr, "localhost").await?;
//! let instance = Instance::new(Client::new(conn));
//! let adapter = instance.request_adapter().await?;
//! let (device, queue) = adapter.request_device().await?;
//! let buf = device.create_buffer(&BufferDescriptor { /* ... */ });
//! ```

#[cfg(feature = "quic")]
pub mod quic {
    //! Type aliases specialized to `QuicConnection`. With this prelude, every
    //! facade type loses its `<C>` parameter for the common QUIC use case.

    use wgpu_remote_transport::quic::QuicConnection;

    pub use wgpu_remote_transport::quic::{QuicConnection as Connection, QuicEndpoint};

    pub type Client = crate::Client<QuicConnection>;
    pub type Instance = crate::Instance<QuicConnection>;
    pub type Adapter = crate::Adapter<QuicConnection>;
    pub type Device = crate::Device<QuicConnection>;
    pub type Queue = crate::Queue<QuicConnection>;

    pub type Buffer = crate::Buffer<QuicConnection>;
    pub type Texture = crate::Texture<QuicConnection>;
    pub type TextureView = crate::TextureView<QuicConnection>;
    pub type Sampler = crate::Sampler<QuicConnection>;
    pub type ShaderModule = crate::ShaderModule<QuicConnection>;
    pub type BindGroupLayout = crate::BindGroupLayout<QuicConnection>;
    pub type BindGroup = crate::BindGroup<QuicConnection>;
    pub type PipelineLayout = crate::PipelineLayout<QuicConnection>;
    pub type ComputePipeline = crate::ComputePipeline<QuicConnection>;
    pub type RenderPipeline = crate::RenderPipeline<QuicConnection>;

    pub type CommandEncoder = crate::CommandEncoder<QuicConnection>;
    pub type ComputePass<'enc> = crate::ComputePass<'enc, QuicConnection>;
    pub type RenderPass<'enc> = crate::RenderPass<'enc, QuicConnection>;
    pub type RenderPassDescriptor<'a> = crate::RenderPassDescriptor<'a, QuicConnection>;
    pub type ColorAttachment<'a> = crate::ColorAttachment<'a, QuicConnection>;
    pub type DepthStencilAttachment<'a> = crate::DepthStencilAttachment<'a, QuicConnection>;
    pub use crate::CommandBuffer;
    pub use crate::ClientError;

    // Descriptor types (transport-agnostic) are re-exported as-is so a single
    // `use prelude::quic::*` covers everything the user needs.
    pub use crate::descriptors::{
        BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindingResource,
        BufferBinding, BufferDescriptor, ComputePipelineDescriptor, FragmentState,
        PipelineLayoutDescriptor, RenderPipelineDescriptor, SamplerDescriptor, ShaderModuleDescriptor,
        ShaderSource, TextureDescriptor, TextureViewDescriptor, VertexBufferLayout, VertexState,
    };
    pub use crate::{
        BindGroupLayoutEntry, BindingType, BufferBindingType, BufferUsages, ShaderStages,
    };
}

#[cfg(feature = "iroh")]
pub mod iroh {
    //! Type aliases specialized to `IrohConnection`.

    use wgpu_remote_transport_iroh::IrohConnection;

    pub use wgpu_remote_transport_iroh::{IrohConnection as Connection, IrohEndpoint};

    pub type Client = crate::Client<IrohConnection>;
    pub type Instance = crate::Instance<IrohConnection>;
    pub type Adapter = crate::Adapter<IrohConnection>;
    pub type Device = crate::Device<IrohConnection>;
    pub type Queue = crate::Queue<IrohConnection>;

    pub type Buffer = crate::Buffer<IrohConnection>;
    pub type Texture = crate::Texture<IrohConnection>;
    pub type TextureView = crate::TextureView<IrohConnection>;
    pub type Sampler = crate::Sampler<IrohConnection>;
    pub type ShaderModule = crate::ShaderModule<IrohConnection>;
    pub type BindGroupLayout = crate::BindGroupLayout<IrohConnection>;
    pub type BindGroup = crate::BindGroup<IrohConnection>;
    pub type PipelineLayout = crate::PipelineLayout<IrohConnection>;
    pub type ComputePipeline = crate::ComputePipeline<IrohConnection>;
    pub type RenderPipeline = crate::RenderPipeline<IrohConnection>;

    pub type CommandEncoder = crate::CommandEncoder<IrohConnection>;
    pub type ComputePass<'enc> = crate::ComputePass<'enc, IrohConnection>;
    pub type RenderPass<'enc> = crate::RenderPass<'enc, IrohConnection>;
    pub type RenderPassDescriptor<'a> = crate::RenderPassDescriptor<'a, IrohConnection>;
    pub type ColorAttachment<'a> = crate::ColorAttachment<'a, IrohConnection>;
    pub type DepthStencilAttachment<'a> = crate::DepthStencilAttachment<'a, IrohConnection>;
    pub use crate::CommandBuffer;
    pub use crate::ClientError;

    pub use crate::descriptors::{
        BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindingResource,
        BufferBinding, BufferDescriptor, ComputePipelineDescriptor, FragmentState,
        PipelineLayoutDescriptor, RenderPipelineDescriptor, SamplerDescriptor, ShaderModuleDescriptor,
        ShaderSource, TextureDescriptor, TextureViewDescriptor, VertexBufferLayout, VertexState,
    };
    pub use crate::{
        BindGroupLayoutEntry, BindingType, BufferBindingType, BufferUsages, ShaderStages,
    };
}
