//! Local recorders for command buffers and passes.
//!
//! No network round-trip happens during recording — all calls just push into
//! a `Vec<EncoderCommand>` / `Vec<ComputeCommand>`. The recording is shipped
//! to the server when [`Queue::submit`](crate::Queue::submit) is called, in
//! one batched [`Action::Submit`](wgpu_remote_protocol::Action::Submit).
//!
//! This is the latency optimization the original architecture document
//! flagged: thousands of draw / dispatch calls per frame would otherwise be
//! thousands of round-trips. Buffering them locally and flushing on submit
//! collapses that to one round-trip per frame.

use std::sync::Arc;

use wgpu_remote_protocol::{
    commands::{CommandBufferRecording, ComputeCommand, EncoderCommand},
    ids::BindGroupId,
};
use wgpu_remote_transport::Connection;

use crate::{
    Client,
    resources::{BindGroup, Buffer, ComputePipeline},
};

/// A `wgpu::CommandEncoder` mirror. Calls accumulate into [`recording`].
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
