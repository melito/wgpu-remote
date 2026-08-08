//! Wire format for proxying wgpu calls to a remote GPU.
//!
//! This crate is pure data — no I/O, no async. The shape is:
//!
//! - [`ids`]: typed resource handles minted by the client.
//! - [`actions`]: client → server messages.
//! - [`responses`]: server → client messages.
//! - [`codec`]: length-delimited bincode framing.
//! - [`version`]: protocol version handshake.

pub mod actions;
pub mod codec;
pub mod commands;
pub mod descriptors;
pub mod ids;
pub mod responses;
pub mod version;

pub use actions::{Action, RequestId};
pub use codec::{CodecError, decode_frame, encode_frame};
pub use ids::{
    BindGroupId, BindGroupLayoutId, BufferId, CommandBufferId, ComputePipelineId, DeviceId,
    PipelineLayoutId, QueueId, RenderPipelineId, ResourceId, SamplerId, ShaderModuleId, TextureId,
    TextureViewId,
};
pub use responses::Response;
pub use version::PROTOCOL_VERSION;
