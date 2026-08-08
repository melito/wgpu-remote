//! Local recorders for command buffers and passes.
//!
//! No network round-trip happens during recording — all calls just push into
//! a `Vec<EncoderCommand>` / `Vec<ComputeCommand>`. The recording is shipped
//! to the server when [`Queue::submit`](crate::client::Queue::submit) is called, in
//! one batched [`Action::Submit`](crate::protocol::Action::Submit).
//!
//! This is the latency optimization the original architecture document
//! flagged: thousands of draw / dispatch calls per frame would otherwise be
//! thousands of round-trips. Buffering them locally and flushing on submit
//! collapses that to one round-trip per frame.

use std::sync::Arc;

use crate::protocol::{
    commands::{
        CommandBufferRecording, ComputeCommand, EncoderCommand, ImageCopyBuffer, ImageCopyTexture,
        RenderCommand, RenderPassColorAttachment as ProtoColorAttachment,
        RenderPassDepthStencilAttachment as ProtoDepthAttachment,
    },
    ids::BindGroupId,
};
use crate::transport::Connection;
use wgpu_types::{
    Color, Extent3d, IndexFormat, Operations, Origin3d, TexelCopyBufferLayout, TextureAspect,
};

use crate::client::{
    Client,
    resources::{BindGroup, Buffer, ComputePipeline, RenderPipeline, Texture, TextureView},
};

/// A `wgpu::CommandEncoder` mirror. Calls accumulate into an internal recording.
pub struct CommandEncoder<C: Connection + Clone + 'static> {
    label: Option<String>,
    commands: Vec<EncoderCommand>,
    /// Held for symmetry with `wgpu::CommandEncoder` (which lives on a
    /// `Device`). Currently unused — recordings are pure data — but keeping
    /// it means the encoder can later issue commands that need a connection
    /// (e.g. resolving query sets) without a backwards-incompatible shape
    /// change.
    _client: Arc<Client<C>>,
}

impl<C: Connection + Clone + 'static> CommandEncoder<C> {
    pub(crate) fn new(label: Option<String>, client: Arc<Client<C>>) -> Self {
        Self {
            label,
            commands: Vec::new(),
            _client: client,
        }
    }

    pub fn copy_buffer_to_buffer(
        &mut self,
        source: &Buffer<C>,
        source_offset: u64,
        destination: &Buffer<C>,
        destination_offset: u64,
        size: u64,
    ) {
        self.commands.push(EncoderCommand::CopyBufferToBuffer {
            source: source.id(),
            source_offset,
            destination: destination.id(),
            destination_offset,
            size,
        });
    }

    pub fn clear_buffer(&mut self, buffer: &Buffer<C>, offset: u64, size: Option<u64>) {
        self.commands.push(EncoderCommand::ClearBuffer {
            buffer: buffer.id(),
            offset,
            size,
        });
    }

    /// Open a compute pass. The returned [`ComputePass`] borrows `&mut self`,
    /// matching wgpu's API and statically preventing the encoder from being
    /// used while a pass is open.
    pub fn begin_compute_pass(&mut self, label: Option<String>) -> ComputePass<'_, C> {
        ComputePass {
            encoder: self,
            label,
            commands: Vec::new(),
        }
    }

    /// Open a render pass. Attachments are described by [`ColorAttachment`]
    /// and [`DepthStencilAttachment`] which take the facade's typed
    /// [`TextureView`] handles.
    pub fn begin_render_pass<'a>(
        &mut self,
        descriptor: RenderPassDescriptor<'a, C>,
    ) -> RenderPass<'_, C> {
        let color_attachments = descriptor
            .color_attachments
            .iter()
            .map(|maybe| {
                maybe.as_ref().map(|att| ProtoColorAttachment {
                    view: att.view.id(),
                    depth_slice: att.depth_slice,
                    resolve_target: att.resolve_target.map(|v| v.id()),
                    ops: att.ops,
                })
            })
            .collect();
        let depth_stencil_attachment =
            descriptor.depth_stencil_attachment.as_ref().map(|att| ProtoDepthAttachment {
                view: att.view.id(),
                depth_ops: att.depth_ops,
                stencil_ops: att.stencil_ops,
            });
        RenderPass {
            encoder: self,
            label: descriptor.label,
            color_attachments,
            depth_stencil_attachment,
            commands: Vec::new(),
        }
    }

    /// Copy a region of a texture into a buffer. Mirrors wgpu's
    /// `copy_texture_to_buffer`. The `layout` describes how the texels are
    /// arranged in the destination buffer.
    pub fn copy_texture_to_buffer(
        &mut self,
        source: &Texture<C>,
        source_mip_level: u32,
        source_origin: Origin3d,
        source_aspect: TextureAspect,
        destination: &Buffer<C>,
        destination_layout: TexelCopyBufferLayout,
        copy_size: Extent3d,
    ) {
        self.commands.push(EncoderCommand::CopyTextureToBuffer {
            source: ImageCopyTexture {
                texture: source.id(),
                mip_level: source_mip_level,
                origin: source_origin,
                aspect: source_aspect,
            },
            destination: ImageCopyBuffer {
                buffer: destination.id(),
                layout: destination_layout,
            },
            copy_size,
        });
    }

    pub fn finish(self) -> CommandBuffer {
        CommandBuffer {
            recording: CommandBufferRecording {
                label: self.label,
                commands: self.commands,
            },
        }
    }
}

/// Pass-scoped recorder. On Drop the recorded ops are appended to the parent
/// encoder under a `BeginComputePass` variant.
pub struct ComputePass<'enc, C: Connection + Clone + 'static> {
    encoder: &'enc mut CommandEncoder<C>,
    label: Option<String>,
    commands: Vec<ComputeCommand>,
}

impl<C: Connection + Clone + 'static> ComputePass<'_, C> {
    pub fn set_pipeline(&mut self, pipeline: &ComputePipeline<C>) {
        self.commands
            .push(ComputeCommand::SetPipeline(pipeline.id()));
    }

    pub fn set_bind_group(&mut self, index: u32, group: &BindGroup<C>, offsets: &[u32]) {
        self.commands.push(ComputeCommand::SetBindGroup {
            index,
            group: group.id(),
            offsets: offsets.to_vec(),
        });
    }

    pub fn dispatch_workgroups(&mut self, x: u32, y: u32, z: u32) {
        self.commands
            .push(ComputeCommand::DispatchWorkgroups { x, y, z });
    }

    /// Explicit close — equivalent to letting the pass drop. Returns
    /// nothing; the pass commands are flushed into the encoder.
    pub fn end(self) {
        // Drop runs.
    }
}

impl<C: Connection + Clone + 'static> Drop for ComputePass<'_, C> {
    fn drop(&mut self) {
        let label = self.label.take();
        let commands = std::mem::take(&mut self.commands);
        self.encoder
            .commands
            .push(EncoderCommand::BeginComputePass { label, commands });
    }
}

/// Result of [`CommandEncoder::finish`]. Owns a serialized-ready recording.
pub struct CommandBuffer {
    pub(crate) recording: CommandBufferRecording,
}

impl CommandBuffer {
    /// Wrap a hand-built recording. Intended for consumers that record
    /// command buffers without going through this crate's
    /// [`CommandEncoder`] — notably, the `wgpu-remote-wgpu` drop-in, where
    /// wgpu's own `CommandEncoder` lifetime model doesn't fit through the
    /// facade's `&mut`-borrowing pass types.
    pub fn from_recording(recording: CommandBufferRecording) -> Self {
        Self { recording }
    }

    pub(crate) fn into_recording(self) -> CommandBufferRecording {
        self.recording
    }
}

// `BindGroupId` re-exported for symmetry with the rest of the surface; not
// directly used in this module's public API, but consumers that build raw
// `ComputeCommand` values may want it.
#[allow(dead_code)]
fn _bind_group_id_marker() -> Option<BindGroupId> {
    None
}

// ---- Render pass --------------------------------------------------------

/// User-facing render pass descriptor — references facade [`TextureView`]
/// handles directly. Translated to the protocol form (with IDs) when the
/// pass is opened.
pub struct RenderPassDescriptor<'a, C: Connection + Clone + 'static> {
    pub label: Option<String>,
    pub color_attachments: &'a [Option<ColorAttachment<'a, C>>],
    pub depth_stencil_attachment: Option<DepthStencilAttachment<'a, C>>,
}

pub struct ColorAttachment<'a, C: Connection + Clone + 'static> {
    pub view: &'a TextureView<C>,
    pub depth_slice: Option<u32>,
    pub resolve_target: Option<&'a TextureView<C>>,
    pub ops: Operations<Color>,
}

pub struct DepthStencilAttachment<'a, C: Connection + Clone + 'static> {
    pub view: &'a TextureView<C>,
    pub depth_ops: Option<Operations<f32>>,
    pub stencil_ops: Option<Operations<u32>>,
}

/// Render pass recorder. Drops back into the parent encoder under a
/// `BeginRenderPass` variant.
pub struct RenderPass<'enc, C: Connection + Clone + 'static> {
    encoder: &'enc mut CommandEncoder<C>,
    label: Option<String>,
    color_attachments: Vec<Option<ProtoColorAttachment>>,
    depth_stencil_attachment: Option<ProtoDepthAttachment>,
    commands: Vec<RenderCommand>,
}

impl<C: Connection + Clone + 'static> RenderPass<'_, C> {
    pub fn set_pipeline(&mut self, pipeline: &RenderPipeline<C>) {
        self.commands
            .push(RenderCommand::SetPipeline(pipeline.id()));
    }

    pub fn set_bind_group(&mut self, index: u32, group: &BindGroup<C>, offsets: &[u32]) {
        self.commands.push(RenderCommand::SetBindGroup {
            index,
            group: group.id(),
            offsets: offsets.to_vec(),
        });
    }

    pub fn set_vertex_buffer(&mut self, slot: u32, buffer: &Buffer<C>, offset: u64, size: Option<u64>) {
        self.commands.push(RenderCommand::SetVertexBuffer {
            slot,
            buffer: buffer.id(),
            offset,
            size,
        });
    }

    pub fn set_index_buffer(
        &mut self,
        buffer: &Buffer<C>,
        format: IndexFormat,
        offset: u64,
        size: Option<u64>,
    ) {
        self.commands.push(RenderCommand::SetIndexBuffer {
            buffer: buffer.id(),
            format,
            offset,
            size,
        });
    }

    pub fn draw(&mut self, vertices: std::ops::Range<u32>, instances: std::ops::Range<u32>) {
        self.commands.push(RenderCommand::Draw {
            vertices,
            instances,
        });
    }

    pub fn draw_indexed(
        &mut self,
        indices: std::ops::Range<u32>,
        base_vertex: i32,
        instances: std::ops::Range<u32>,
    ) {
        self.commands.push(RenderCommand::DrawIndexed {
            indices,
            base_vertex,
            instances,
        });
    }

    /// Explicit close — equivalent to letting the pass drop.
    pub fn end(self) {}
}

impl<C: Connection + Clone + 'static> Drop for RenderPass<'_, C> {
    fn drop(&mut self) {
        let label = self.label.take();
        let color_attachments = std::mem::take(&mut self.color_attachments);
        let depth_stencil_attachment = self.depth_stencil_attachment.take();
        let commands = std::mem::take(&mut self.commands);
        self.encoder.commands.push(EncoderCommand::BeginRenderPass {
            label,
            color_attachments,
            depth_stencil_attachment,
            commands,
        });
    }
}
