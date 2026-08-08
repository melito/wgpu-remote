//! Convenience re-exports for callers that don't want to write the
//! [`Connection`](crate::transport::Connection) generic at every type.
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
//! use crate::client::prelude::quic::*;
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

    use crate::transport::quic::QuicConnection;

    pub use crate::transport::quic::{QuicConnection as Connection, QuicEndpoint};

    pub type Client = crate::client::Client<QuicConnection>;
    pub type Instance = crate::client::Instance<QuicConnection>;
    pub type Adapter = crate::client::Adapter<QuicConnection>;
    pub type Device = crate::client::Device<QuicConnection>;
    pub type Queue = crate::client::Queue<QuicConnection>;

    pub type Buffer = crate::client::Buffer<QuicConnection>;
    pub type Texture = crate::client::Texture<QuicConnection>;
    pub type TextureView = crate::client::TextureView<QuicConnection>;
    pub type Sampler = crate::client::Sampler<QuicConnection>;
    pub type ShaderModule = crate::client::ShaderModule<QuicConnection>;
    pub type BindGroupLayout = crate::client::BindGroupLayout<QuicConnection>;
    pub type BindGroup = crate::client::BindGroup<QuicConnection>;
    pub type PipelineLayout = crate::client::PipelineLayout<QuicConnection>;
    pub type ComputePipeline = crate::client::ComputePipeline<QuicConnection>;
    pub type RenderPipeline = crate::client::RenderPipeline<QuicConnection>;

    pub type CommandEncoder = crate::client::CommandEncoder<QuicConnection>;
    pub type ComputePass<'enc> = crate::client::ComputePass<'enc, QuicConnection>;
    pub type RenderPass<'enc> = crate::client::RenderPass<'enc, QuicConnection>;
    pub type RenderPassDescriptor<'a> = crate::client::RenderPassDescriptor<'a, QuicConnection>;
    pub type ColorAttachment<'a> = crate::client::ColorAttachment<'a, QuicConnection>;
    pub type DepthStencilAttachment<'a> = crate::client::DepthStencilAttachment<'a, QuicConnection>;
    pub use crate::client::CommandBuffer;
    pub use crate::client::ClientError;

    // Descriptor types (transport-agnostic) are re-exported as-is so a single
    // `use prelude::quic::*` covers everything the user needs.
    pub use crate::client::descriptors::{
        BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindingResource,
        BufferBinding, BufferDescriptor, ComputePipelineDescriptor, FragmentState,
        PipelineLayoutDescriptor, RenderPipelineDescriptor, SamplerDescriptor, ShaderModuleDescriptor,
        ShaderSource, TextureDescriptor, TextureViewDescriptor, VertexBufferLayout, VertexState,
    };
    pub use crate::client::{
        BindGroupLayoutEntry, BindingType, BufferBindingType, BufferUsages, ShaderStages,
    };
}

#[cfg(feature = "iroh")]
pub mod iroh {
    //! Type aliases specialized to `IrohConnection`.

    use crate::transport::iroh::IrohConnection;

    pub use crate::transport::iroh::{IrohConnection as Connection, IrohEndpoint};

    pub type Client = crate::client::Client<IrohConnection>;
    pub type Instance = crate::client::Instance<IrohConnection>;
    pub type Adapter = crate::client::Adapter<IrohConnection>;
    pub type Device = crate::client::Device<IrohConnection>;
    pub type Queue = crate::client::Queue<IrohConnection>;

    pub type Buffer = crate::client::Buffer<IrohConnection>;
    pub type Texture = crate::client::Texture<IrohConnection>;
    pub type TextureView = crate::client::TextureView<IrohConnection>;
    pub type Sampler = crate::client::Sampler<IrohConnection>;
    pub type ShaderModule = crate::client::ShaderModule<IrohConnection>;
    pub type BindGroupLayout = crate::client::BindGroupLayout<IrohConnection>;
    pub type BindGroup = crate::client::BindGroup<IrohConnection>;
    pub type PipelineLayout = crate::client::PipelineLayout<IrohConnection>;
    pub type ComputePipeline = crate::client::ComputePipeline<IrohConnection>;
    pub type RenderPipeline = crate::client::RenderPipeline<IrohConnection>;

    pub type CommandEncoder = crate::client::CommandEncoder<IrohConnection>;
    pub type ComputePass<'enc> = crate::client::ComputePass<'enc, IrohConnection>;
    pub type RenderPass<'enc> = crate::client::RenderPass<'enc, IrohConnection>;
    pub type RenderPassDescriptor<'a> = crate::client::RenderPassDescriptor<'a, IrohConnection>;
    pub type ColorAttachment<'a> = crate::client::ColorAttachment<'a, IrohConnection>;
    pub type DepthStencilAttachment<'a> = crate::client::DepthStencilAttachment<'a, IrohConnection>;
    pub use crate::client::CommandBuffer;
    pub use crate::client::ClientError;

    pub use crate::client::descriptors::{
        BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindingResource,
        BufferBinding, BufferDescriptor, ComputePipelineDescriptor, FragmentState,
        PipelineLayoutDescriptor, RenderPipelineDescriptor, SamplerDescriptor, ShaderModuleDescriptor,
        ShaderSource, TextureDescriptor, TextureViewDescriptor, VertexBufferLayout, VertexState,
    };
    pub use crate::client::{
        BindGroupLayoutEntry, BindingType, BufferBindingType, BufferUsages, ShaderStages,
    };
}
