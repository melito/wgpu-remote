//! Command-buffer recording wire format.
//!
//! A recorded command buffer is a `Vec<EncoderCommand>`. Pass-scoped commands
//! are nested inside their `Begin*Pass` variants — this matches the shape of
//! wgpu's `CommandEncoder` API and lets the replay engine reconstruct passes
//! through Rust's normal borrow rules instead of tagging every command with a
//! pass ID.
//!
//! v1 covers the compute path end-to-end. Render-pass commands are stubbed
//! out with `non_exhaustive` so adding them later is a non-breaking addition
//! to the *enum*, even though it remains a wire-format breaking change (which
//! is what [`crate::PROTOCOL_VERSION`] is for).

use bytes::Bytes;
use serde::{Deserialize, Serialize};
use wgpu_types::{
    BufferAddress, Color, DynamicOffset, Extent3d, Operations, Origin3d, TextureAspect,
};

use crate::ids::{
    BindGroupId, BufferId, ComputePipelineId, RenderPipelineId, TextureId, TextureViewId,
};

/// One recorded command buffer. Encoded into the `Bytes` payload of
/// [`Action::Submit`](crate::Action::Submit).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CommandBufferRecording {
    pub label: Option<String>,
    pub commands: Vec<EncoderCommand>,
}

impl CommandBufferRecording {
    /// Convenience: encode a recording into the wire bytes that `Submit`
    /// expects.
    pub fn encode(&self) -> Result<Bytes, bincode::error::EncodeError> {
        let buf = bincode::serde::encode_to_vec(self, bincode::config::standard())?;
        Ok(Bytes::from(buf))
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, bincode::error::DecodeError> {
        let (recording, _) =
            bincode::serde::decode_from_slice(bytes, bincode::config::standard())?;
        Ok(recording)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[non_exhaustive]
pub enum EncoderCommand {
    CopyBufferToBuffer {
        source: BufferId,
        source_offset: BufferAddress,
        destination: BufferId,
        destination_offset: BufferAddress,
        size: BufferAddress,
    },
    CopyBufferToTexture {
        source: ImageCopyBuffer,
        destination: ImageCopyTexture,
        copy_size: Extent3d,
    },
    CopyTextureToBuffer {
        source: ImageCopyTexture,
        destination: ImageCopyBuffer,
        copy_size: Extent3d,
    },
    ClearBuffer {
        buffer: BufferId,
        offset: BufferAddress,
        size: Option<BufferAddress>,
    },
    BeginComputePass {
        label: Option<String>,
        commands: Vec<ComputeCommand>,
    },
    BeginRenderPass {
        label: Option<String>,
        color_attachments: Vec<Option<RenderPassColorAttachment>>,
        depth_stencil_attachment: Option<RenderPassDepthStencilAttachment>,
        commands: Vec<RenderCommand>,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RenderPassColorAttachment {
    pub view: TextureViewId,
    pub depth_slice: Option<u32>,
    pub resolve_target: Option<TextureViewId>,
    pub ops: Operations<Color>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RenderPassDepthStencilAttachment {
    pub view: TextureViewId,
    pub depth_ops: Option<Operations<f32>>,
    pub stencil_ops: Option<Operations<u32>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ComputeCommand {
    SetPipeline(ComputePipelineId),
    SetBindGroup {
        index: u32,
        group: BindGroupId,
        offsets: Vec<DynamicOffset>,
    },
    SetPushConstants {
        offset: u32,
        data: Bytes,
    },
    DispatchWorkgroups {
        x: u32,
        y: u32,
        z: u32,
    },
    DispatchWorkgroupsIndirect {
        indirect_buffer: BufferId,
        indirect_offset: BufferAddress,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[non_exhaustive]
pub enum RenderCommand {
    SetPipeline(RenderPipelineId),
    SetBindGroup {
        index: u32,
        group: BindGroupId,
        offsets: Vec<DynamicOffset>,
    },
    SetVertexBuffer {
        slot: u32,
        buffer: BufferId,
        offset: BufferAddress,
        size: Option<BufferAddress>,
    },
    SetIndexBuffer {
        buffer: BufferId,
        format: wgpu_types::IndexFormat,
        offset: BufferAddress,
        size: Option<BufferAddress>,
    },
    Draw {
        vertices: std::ops::Range<u32>,
        instances: std::ops::Range<u32>,
    },
    DrawIndexed {
        indices: std::ops::Range<u32>,
        base_vertex: i32,
        instances: std::ops::Range<u32>,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ImageCopyBuffer {
    pub buffer: BufferId,
    pub layout: wgpu_types::TexelCopyBufferLayout,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ImageCopyTexture {
    pub texture: TextureId,
    pub mip_level: u32,
    pub origin: Origin3d,
    pub aspect: TextureAspect,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recording_roundtrip() {
        let rec = CommandBufferRecording {
            label: Some("test".into()),
            commands: vec![EncoderCommand::BeginComputePass {
                label: None,
                commands: vec![
                    ComputeCommand::SetPipeline(ComputePipelineId::new(7)),
                    ComputeCommand::DispatchWorkgroups { x: 4, y: 1, z: 1 },
                ],
            }],
        };
        let bytes = rec.encode().unwrap();
        let decoded = CommandBufferRecording::decode(&bytes).unwrap();
        assert_eq!(decoded.label.as_deref(), Some("test"));
        assert_eq!(decoded.commands.len(), 1);
    }
}
