//! Client → server messages.

use bytes::Bytes;
use serde::{Deserialize, Serialize};

use crate::protocol::descriptors::{
    BindGroupDescriptor, BindGroupLayoutDescriptor, BufferDescriptor, CommandEncoderDescriptor,
    ComputePipelineDescriptor, PipelineLayoutDescriptor, RenderPipelineDescriptor,
    SamplerDescriptor, ShaderModuleDescriptor, TextureDescriptor, TextureViewDescriptor,
};
use crate::protocol::ids::*;

/// Correlates a request with its [`Response`](crate::protocol::Response). Client-minted,
/// monotonic. `None` is allowed for fire-and-forget actions.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd, Serialize, Deserialize)]
#[repr(transparent)]
pub struct RequestId(pub u64);

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Frame {
    pub request_id: Option<RequestId>,
    pub action: Action,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Action {
    // ---- Handshake ----
    /// Sent immediately after connection. Server responds with its own version.
    Hello { protocol_version: u32 },

    // ---- Adapter / device ----
    RequestAdapter {
        // TODO: RequestAdapterOptions mirror — power preference, force fallback,
        // compatible surface (which we don't have in v1).
    },
    RequestDevice {
        // wgpu_types::DeviceDescriptor<Option<String>>; defined inline rather
        // than aliased because it carries the Trace enum which we want to pin
        // to Off in v1.
        label: Option<String>,
    },

    // ---- Resource creation ----
    CreateBuffer {
        id: BufferId,
        desc: BufferDescriptor,
    },
    CreateTexture {
        id: TextureId,
        desc: TextureDescriptor,
    },
    CreateTextureView {
        id: TextureViewId,
        texture: TextureId,
        desc: TextureViewDescriptor,
    },
    CreateSampler {
        id: SamplerId,
        desc: SamplerDescriptor,
    },
    CreateShaderModule {
        id: ShaderModuleId,
        desc: ShaderModuleDescriptor,
    },
    CreateBindGroupLayout {
        id: BindGroupLayoutId,
        desc: BindGroupLayoutDescriptor,
    },
    CreateBindGroup {
        id: BindGroupId,
        desc: BindGroupDescriptor,
    },
    CreatePipelineLayout {
        id: PipelineLayoutId,
        desc: PipelineLayoutDescriptor,
    },
    CreateRenderPipeline {
        id: RenderPipelineId,
        desc: RenderPipelineDescriptor,
    },
    CreateComputePipeline {
        id: ComputePipelineId,
        desc: ComputePipelineDescriptor,
    },
    CreateCommandEncoder {
        id: CommandBufferId,
        desc: CommandEncoderDescriptor,
    },

    /// Explicit destruction. Client emits this when its wrapper handle drops.
    Destroy(ResourceId),

    // ---- Data transfer ----
    /// Upload bytes into a buffer at offset.
    WriteBuffer {
        buffer: BufferId,
        offset: u64,
        data: Bytes,
    },

    // ---- Submission ----
    /// Submit pre-encoded command buffers. The recording payload is an opaque
    /// blob in v1 — its inner structure is defined separately.
    Submit { recordings: Vec<Bytes> },

    // ---- Readback (latency-sensitive) ----
    /// Read a buffer range back to the client. Server waits on the relevant
    /// fence, copies the bytes, replies with [`Response::BufferData`].
    ///
    /// [`Response::BufferData`]: crate::protocol::Response::BufferData
    MapBufferForRead {
        buffer: BufferId,
        offset: u64,
        size: u64,
    },
}
