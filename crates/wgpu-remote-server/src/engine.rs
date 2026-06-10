//! Replay engine. Owns a `wgpu::Instance` + `Device` + `Queue` and a set of
//! [`ResourceTables`]; receives [`Frame`]s,
//! dispatches against `wgpu`, returns [`ResponseFrame`]s.
//!
//! v1 scope: handshake, buffer create/destroy/write, buffer readback. Other
//! resource types are wired up next; for now they return `Response::Ok` after
//! stub creation so the protocol round-trips without crashing.

use std::sync::Mutex;

use bytes::Bytes;
use thiserror::Error;
use tokio::sync::oneshot;
use wgpu_remote_protocol::{
    Action, PROTOCOL_VERSION, Response,
    actions::Frame,
    commands::{
        CommandBufferRecording, ComputeCommand, EncoderCommand, ImageCopyBuffer, ImageCopyTexture,
        RenderCommand,
    },
    descriptors::{
        BindGroupDescriptor, BindGroupLayoutDescriptor, BindingResource, ComputePipelineDescriptor,
        PipelineLayoutDescriptor, RenderPipelineDescriptor, SamplerDescriptor,
        ShaderModuleDescriptor, ShaderSource, TextureDescriptor, TextureViewDescriptor,
    },
    ids::{
        BindGroupId, BindGroupLayoutId, BufferId, ComputePipelineId, PipelineLayoutId,
        RenderPipelineId, ResourceId, SamplerId, ShaderModuleId, TextureId, TextureViewId,
    },
    responses::{ErrorCode, ResponseFrame},
};

use crate::tables::ResourceTables;

#[derive(Debug, Error)]
pub enum EngineError {
    #[error("failed to find a wgpu adapter")]
    NoAdapter,
    #[error("failed to create wgpu device: {0}")]
    DeviceCreation(String),
}

/// Holds the shared wgpu state plus the resource tables. The tables sit
/// behind a `Mutex` because dispatch may be driven from multiple tasks
/// (one per stream) once the server loop lands.
pub struct Engine {
    _instance: wgpu::Instance,
    _adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    tables: Mutex<ResourceTables>,
}

impl Engine {
    /// Build an engine with a fresh wgpu instance and a high-performance
    /// adapter. Compute + offscreen render only — no surface/swapchain
    /// support in v1.
    pub async fn new() -> Result<Self, EngineError> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .map_err(|_| EngineError::NoAdapter)?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("wgpu-remote-server"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
                experimental_features: wgpu::ExperimentalFeatures::default(),
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|e| EngineError::DeviceCreation(e.to_string()))?;

        Ok(Self {
            _instance: instance,
            _adapter: adapter,
            device,
            queue,
            tables: Mutex::new(ResourceTables::default()),
        })
    }

    /// Dispatch one client frame. Returns a response if the action expects one.
    pub async fn dispatch(&self, frame: Frame) -> Option<ResponseFrame> {
        let request_id = frame.request_id;
        let response = self.handle(frame.action).await;
        match (request_id, response) {
            (Some(rid), Some(resp)) => Some(ResponseFrame {
                request_id: rid,
                response: resp,
            }),
            // Fire-and-forget action with no response — drop.
            (None, _) => None,
            // Action returned a response but client supplied no request ID:
            // the only way that happens today is the client failing to set
            // one for a request-shaped action. Surface as an error response
            // would require a request_id we don't have, so drop with a log
            // hook later.
            (Some(_), None) => None,
        }
    }

    async fn handle(&self, action: Action) -> Option<Response> {
        match action {
            Action::Hello { protocol_version } => {
                if protocol_version != PROTOCOL_VERSION {
                    Some(Response::Error {
                        code: ErrorCode::ProtocolVersionMismatch,
                        message: format!(
                            "client v{protocol_version}, server v{PROTOCOL_VERSION}"
                        ),
                    })
                } else {
                    Some(Response::HelloAck {
                        protocol_version: PROTOCOL_VERSION,
                    })
                }
            }

            // Adapter/device acquisition is collapsed in v1: the engine has
            // exactly one device, created at startup. These are acks only.
            Action::RequestAdapter { .. } | Action::RequestDevice { .. } => Some(Response::Ok),

            Action::CreateBuffer { id, desc } => {
                let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: desc.label.as_deref(),
                    size: desc.size,
                    usage: desc.usage,
                    mapped_at_creation: desc.mapped_at_creation,
                });
                self.tables.lock().unwrap().buffers.insert(id, buffer);
                Some(Response::Ok)
            }

            Action::Destroy(ResourceId::Buffer(id)) => {
                self.tables.lock().unwrap().buffers.remove(&id);
                Some(Response::Ok)
            }

            Action::WriteBuffer {
                buffer,
                offset,
                data,
            } => match self.with_buffer(buffer, |b| {
                self.queue.write_buffer(b, offset, &data);
            }) {
                Ok(()) => {
                    // write_buffer is scheduled for the next submission, so
                    // flush an empty submit to make the data visible before
                    // any subsequent map_async sees this buffer's range.
                    self.queue.submit(std::iter::empty());
                    Some(Response::Ok)
                }
                Err(resp) => Some(resp),
            },

            Action::MapBufferForRead {
                buffer,
                offset,
                size,
            } => Some(self.read_buffer(buffer, offset, size).await),

            Action::CreateShaderModule { id, desc } => Some(self.create_shader_module(id, desc)),

            Action::CreateBindGroupLayout { id, desc } => {
                Some(self.create_bind_group_layout(id, desc))
            }

            Action::CreateBindGroup { id, desc } => Some(self.create_bind_group(id, desc)),

            Action::CreatePipelineLayout { id, desc } => {
                Some(self.create_pipeline_layout(id, desc))
            }

            Action::CreateComputePipeline { id, desc } => {
                Some(self.create_compute_pipeline(id, desc))
            }

            Action::Submit { recordings } => Some(self.submit(recordings)),

            Action::CreateTexture { id, desc } => Some(self.create_texture(id, desc)),
            Action::CreateTextureView {
                id,
                texture,
                desc,
            } => Some(self.create_texture_view(id, texture, desc)),
            Action::CreateSampler { id, desc } => Some(self.create_sampler(id, desc)),
            Action::CreateRenderPipeline { id, desc } => {
                Some(self.create_render_pipeline(id, desc))
            }

            // Command encoders are recorded entirely on the client; the
            // server never sees a CreateCommandEncoder before Submit. Treat
            // it as a no-op ack for any client that emits it.
            Action::CreateCommandEncoder { .. } | Action::Destroy(_) => Some(Response::Ok),
        }
    }

    fn with_buffer<R>(
        &self,
        id: BufferId,
        f: impl FnOnce(&wgpu::Buffer) -> R,
    ) -> Result<R, Response> {
        let tables = self.tables.lock().unwrap();
        match tables.buffers.get(&id) {
            Some(b) => Ok(f(b)),
            None => Err(Response::Error {
                code: ErrorCode::UnknownResource,
                message: format!("unknown BufferId({})", id.raw()),
            }),
        }
    }

    /// Map a buffer range for read, copy bytes out, unmap. Drives `device.poll`
    /// in a blocking task to step the wgpu state machine while we await the
    /// map-async callback.
    async fn read_buffer(&self, id: BufferId, offset: u64, size: u64) -> Response {
        // Snapshot the buffer handle so we can release the table lock before
        // any awaits.
        let buffer = match self.tables.lock().unwrap().buffers.get(&id).cloned() {
            Some(b) => b,
            None => {
                return Response::Error {
                    code: ErrorCode::UnknownResource,
                    message: format!("unknown BufferId({})", id.raw()),
                };
            }
        };

        let (tx, rx) = oneshot::channel();
        buffer
            .slice(offset..offset + size)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });

        // Drive device.poll on a blocking thread until the map callback fires
        // or errors out. We use Wait to block inside wgpu rather than busy-loop.
        let device = self.device.clone();
        let poll_handle = tokio::task::spawn_blocking(move || {
            device.poll(wgpu::PollType::wait_indefinitely()).map(|_| ())
        });

        let map_result = rx.await;
        let _ = poll_handle.await; // ensure the blocking task is drained

        match map_result {
            Ok(Ok(())) => {
                let view = buffer.slice(offset..offset + size).get_mapped_range();
                let bytes = Bytes::copy_from_slice(&view);
                drop(view);
                buffer.unmap();
                Response::BufferData { data: bytes }
            }
            Ok(Err(e)) => Response::Error {
                code: ErrorCode::Internal,
                message: format!("buffer map_async failed: {e:?}"),
            },
            Err(_) => Response::Error {
                code: ErrorCode::Internal,
                message: "map callback channel dropped".into(),
            },
        }
    }

    fn create_shader_module(&self, id: ShaderModuleId, desc: ShaderModuleDescriptor) -> Response {
        let module = match desc.source {
            ShaderSource::Wgsl(src) => self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: desc.label.as_deref(),
                source: wgpu::ShaderSource::Wgsl(src.into()),
            }),
            ShaderSource::SpirV(_) | ShaderSource::Glsl { .. } => {
                return Response::Error {
                    code: ErrorCode::InvalidArgument,
                    message: "only WGSL shader sources are supported in v1".into(),
                };
            }
        };
        self.tables.lock().unwrap().shader_modules.insert(id, module);
        Response::Ok
    }

    fn create_bind_group_layout(
        &self,
        id: BindGroupLayoutId,
        desc: BindGroupLayoutDescriptor,
    ) -> Response {
        let bgl = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: desc.label.as_deref(),
                entries: &desc.entries,
            });
        self.tables.lock().unwrap().bind_group_layouts.insert(id, bgl);
        Response::Ok
    }

    fn create_bind_group(&self, id: BindGroupId, desc: BindGroupDescriptor) -> Response {
        // Hold the lock for the full descriptor build — we need stable refs to
        // the buffers/samplers/views for the duration of create_bind_group.
        let tables = self.tables.lock().unwrap();
        let layout = match tables.bind_group_layouts.get(&desc.layout) {
            Some(l) => l,
            None => {
                return Response::Error {
                    code: ErrorCode::UnknownResource,
                    message: format!("unknown BindGroupLayoutId({})", desc.layout.raw()),
                };
            }
        };

        let entries: Result<Vec<wgpu::BindGroupEntry<'_>>, Response> = desc
            .entries
            .iter()
            .map(|e| {
                let resource = match &e.resource {
                    BindingResource::Buffer { buffer, offset, size } => {
                        let b = tables.buffers.get(buffer).ok_or_else(|| Response::Error {
                            code: ErrorCode::UnknownResource,
                            message: format!("unknown BufferId({})", buffer.raw()),
                        })?;
                        wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: b,
                            offset: *offset,
                            size: *size,
                        })
                    }
                    BindingResource::Sampler(id) => {
                        let s = tables.samplers.get(id).ok_or_else(|| Response::Error {
                            code: ErrorCode::UnknownResource,
                            message: format!("unknown SamplerId({})", id.raw()),
                        })?;
                        wgpu::BindingResource::Sampler(s)
                    }
                    BindingResource::TextureView(id) => {
                        let v = tables.texture_views.get(id).ok_or_else(|| Response::Error {
                            code: ErrorCode::UnknownResource,
                            message: format!("unknown TextureViewId({})", id.raw()),
                        })?;
                        wgpu::BindingResource::TextureView(v)
                    }
                    // Arrays and other variants land with the next iteration.
                    other => {
                        return Err(Response::Error {
                            code: ErrorCode::InvalidArgument,
                            message: format!("BindingResource variant not yet supported: {other:?}"),
                        });
                    }
                };
                Ok(wgpu::BindGroupEntry {
                    binding: e.binding,
                    resource,
                })
            })
            .collect();

        let entries = match entries {
            Ok(v) => v,
            Err(e) => return e,
        };

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: desc.label.as_deref(),
            layout,
            entries: &entries,
        });
        // Drop the immutable borrow of the table before mutating it.
        drop(tables);
        self.tables.lock().unwrap().bind_groups.insert(id, bg);
        Response::Ok
    }

    fn create_pipeline_layout(
        &self,
        id: PipelineLayoutId,
        desc: PipelineLayoutDescriptor,
    ) -> Response {
        let tables = self.tables.lock().unwrap();
        let bgls: Result<Vec<&wgpu::BindGroupLayout>, Response> = desc
            .bind_group_layouts
            .iter()
            .map(|bgl_id| {
                tables
                    .bind_group_layouts
                    .get(bgl_id)
                    .ok_or_else(|| Response::Error {
                        code: ErrorCode::UnknownResource,
                        message: format!("unknown BindGroupLayoutId({})", bgl_id.raw()),
                    })
            })
            .collect();
        let bgls = match bgls {
            Ok(v) => v,
            Err(e) => return e,
        };
        let layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: desc.label.as_deref(),
                bind_group_layouts: &bgls,
                push_constant_ranges: &desc.push_constant_ranges,
            });
        drop(tables);
        self.tables.lock().unwrap().pipeline_layouts.insert(id, layout);
        Response::Ok
    }

    fn create_compute_pipeline(
        &self,
        id: ComputePipelineId,
        desc: ComputePipelineDescriptor,
    ) -> Response {
        let tables = self.tables.lock().unwrap();
        let module = match tables.shader_modules.get(&desc.module) {
            Some(m) => m,
            None => {
                return Response::Error {
                    code: ErrorCode::UnknownResource,
                    message: format!("unknown ShaderModuleId({})", desc.module.raw()),
                };
            }
        };
        let layout = match desc.layout {
            Some(layout_id) => match tables.pipeline_layouts.get(&layout_id) {
                Some(l) => Some(l),
                None => {
                    return Response::Error {
                        code: ErrorCode::UnknownResource,
                        message: format!("unknown PipelineLayoutId({})", layout_id.raw()),
                    };
                }
            },
            None => None,
        };
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: desc.label.as_deref(),
                layout,
                module,
                entry_point: desc.entry_point.as_deref(),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        drop(tables);
        self.tables.lock().unwrap().compute_pipelines.insert(id, pipeline);
        Response::Ok
    }

    fn submit(&self, recordings: Vec<Bytes>) -> Response {
        let mut command_buffers = Vec::with_capacity(recordings.len());
        for raw in &recordings {
            let recording = match CommandBufferRecording::decode(raw) {
                Ok(r) => r,
                Err(e) => {
                    return Response::Error {
                        code: ErrorCode::InvalidArgument,
                        message: format!("recording decode failed: {e}"),
                    };
                }
            };
            match self.replay_recording(recording) {
                Ok(cb) => command_buffers.push(cb),
                Err(resp) => return resp,
            }
        }

        let _index = self.queue.submit(command_buffers);
        Response::Ok
    }

    fn replay_recording(
        &self,
        recording: CommandBufferRecording,
    ) -> Result<wgpu::CommandBuffer, Response> {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: recording.label.as_deref(),
            });

        let tables = self.tables.lock().unwrap();
        for cmd in recording.commands {
            self.replay_encoder_command(&tables, &mut encoder, cmd)?;
        }
        drop(tables);
        Ok(encoder.finish())
    }

    fn replay_encoder_command(
        &self,
        tables: &ResourceTables,
        encoder: &mut wgpu::CommandEncoder,
        cmd: EncoderCommand,
    ) -> Result<(), Response> {
        match cmd {
            EncoderCommand::CopyBufferToBuffer {
                source,
                source_offset,
                destination,
                destination_offset,
                size,
            } => {
                let src = tables.buffers.get(&source).ok_or_else(|| Response::Error {
                    code: ErrorCode::UnknownResource,
                    message: format!("unknown BufferId({})", source.raw()),
                })?;
                let dst = tables.buffers.get(&destination).ok_or_else(|| Response::Error {
                    code: ErrorCode::UnknownResource,
                    message: format!("unknown BufferId({})", destination.raw()),
                })?;
                encoder.copy_buffer_to_buffer(src, source_offset, dst, destination_offset, size);
                Ok(())
            }
            EncoderCommand::ClearBuffer {
                buffer,
                offset,
                size,
            } => {
                let b = tables.buffers.get(&buffer).ok_or_else(|| Response::Error {
                    code: ErrorCode::UnknownResource,
                    message: format!("unknown BufferId({})", buffer.raw()),
                })?;
                encoder.clear_buffer(b, offset, size);
                Ok(())
            }
            EncoderCommand::BeginComputePass { label, commands } => {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: label.as_deref(),
                    timestamp_writes: None,
                });
                for c in commands {
                    self.replay_compute_command(tables, &mut pass, c)?;
                }
                Ok(())
            }
            EncoderCommand::CopyBufferToTexture {
                source,
                destination,
                copy_size,
            } => {
                let src = lookup_image_copy_buffer(tables, &source)?;
                let dst = lookup_image_copy_texture(tables, &destination)?;
                encoder.copy_buffer_to_texture(src, dst, copy_size);
                Ok(())
            }
            EncoderCommand::CopyTextureToBuffer {
                source,
                destination,
                copy_size,
            } => {
                let src = lookup_image_copy_texture(tables, &source)?;
                let dst = lookup_image_copy_buffer(tables, &destination)?;
                encoder.copy_texture_to_buffer(src, dst, copy_size);
                Ok(())
            }
            EncoderCommand::BeginRenderPass {
                label,
                color_attachments,
                depth_stencil_attachment,
                commands,
            } => {
                // Build the wgpu attachments. References into the tables must
                // outlive the begin_render_pass call, so we materialize the
                // wgpu attachment structs into a Vec before borrowing it.
                let color_atts: Result<Vec<Option<wgpu::RenderPassColorAttachment<'_>>>, Response> =
                    color_attachments
                        .iter()
                        .map(|maybe| match maybe {
                            None => Ok(None),
                            Some(att) => {
                                let view = tables
                                    .texture_views
                                    .get(&att.view)
                                    .ok_or_else(|| Response::Error {
                                        code: ErrorCode::UnknownResource,
                                        message: format!(
                                            "unknown TextureViewId({})",
                                            att.view.raw()
                                        ),
                                    })?;
                                let resolve = match att.resolve_target {
                                    Some(rt) => Some(
                                        tables.texture_views.get(&rt).ok_or_else(|| {
                                            Response::Error {
                                                code: ErrorCode::UnknownResource,
                                                message: format!(
                                                    "unknown TextureViewId({})",
                                                    rt.raw()
                                                ),
                                            }
                                        })?,
                                    ),
                                    None => None,
                                };
                                Ok(Some(wgpu::RenderPassColorAttachment {
                                    view,
                                    depth_slice: att.depth_slice,
                                    resolve_target: resolve,
                                    ops: att.ops,
                                }))
                            }
                        })
                        .collect();
                let color_atts = color_atts?;

                let depth_att = match &depth_stencil_attachment {
                    Some(att) => {
                        let view = tables.texture_views.get(&att.view).ok_or_else(|| {
                            Response::Error {
                                code: ErrorCode::UnknownResource,
                                message: format!(
                                    "unknown TextureViewId({})",
                                    att.view.raw()
                                ),
                            }
                        })?;
                        Some(wgpu::RenderPassDepthStencilAttachment {
                            view,
                            depth_ops: att.depth_ops,
                            stencil_ops: att.stencil_ops,
                        })
                    }
                    None => None,
                };

                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: label.as_deref(),
                    color_attachments: &color_atts,
                    depth_stencil_attachment: depth_att,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                for c in commands {
                    self.replay_render_command(tables, &mut pass, c)?;
                }
                Ok(())
            }
            // The enum is non_exhaustive; a future variant added without a
            // matching arm should fail loudly rather than be misinterpreted.
            _ => Err(Response::Error {
                code: ErrorCode::InvalidArgument,
                message: "unknown encoder command variant — protocol/server version skew?".into(),
            }),
        }
    }

    fn replay_render_command(
        &self,
        tables: &ResourceTables,
        pass: &mut wgpu::RenderPass<'_>,
        cmd: RenderCommand,
    ) -> Result<(), Response> {
        match cmd {
            RenderCommand::SetPipeline(id) => {
                let p = tables
                    .render_pipelines
                    .get(&id)
                    .ok_or_else(|| Response::Error {
                        code: ErrorCode::UnknownResource,
                        message: format!("unknown RenderPipelineId({})", id.raw()),
                    })?;
                pass.set_pipeline(p);
                Ok(())
            }
            RenderCommand::SetBindGroup {
                index,
                group,
                offsets,
            } => {
                let g = tables.bind_groups.get(&group).ok_or_else(|| Response::Error {
                    code: ErrorCode::UnknownResource,
                    message: format!("unknown BindGroupId({})", group.raw()),
                })?;
                pass.set_bind_group(index, g, &offsets);
                Ok(())
            }
            RenderCommand::SetVertexBuffer {
                slot,
                buffer,
                offset,
                size,
            } => {
                let b = tables.buffers.get(&buffer).ok_or_else(|| Response::Error {
                    code: ErrorCode::UnknownResource,
                    message: format!("unknown BufferId({})", buffer.raw()),
                })?;
                let slice = match size {
                    Some(s) => b.slice(offset..offset + s),
                    None => b.slice(offset..),
                };
                pass.set_vertex_buffer(slot, slice);
                Ok(())
            }
            RenderCommand::SetIndexBuffer {
                buffer,
                format,
                offset,
                size,
            } => {
                let b = tables.buffers.get(&buffer).ok_or_else(|| Response::Error {
                    code: ErrorCode::UnknownResource,
                    message: format!("unknown BufferId({})", buffer.raw()),
                })?;
                let slice = match size {
                    Some(s) => b.slice(offset..offset + s),
                    None => b.slice(offset..),
                };
                pass.set_index_buffer(slice, format);
                Ok(())
            }
            RenderCommand::Draw {
                vertices,
                instances,
            } => {
                pass.draw(vertices, instances);
                Ok(())
            }
            RenderCommand::DrawIndexed {
                indices,
                base_vertex,
                instances,
            } => {
                pass.draw_indexed(indices, base_vertex, instances);
                Ok(())
            }
            _ => Err(Response::Error {
                code: ErrorCode::InvalidArgument,
                message: "unknown render command variant — protocol/server version skew?".into(),
            }),
        }
    }

    fn replay_compute_command(
        &self,
        tables: &ResourceTables,
        pass: &mut wgpu::ComputePass<'_>,
        cmd: ComputeCommand,
    ) -> Result<(), Response> {
        match cmd {
            ComputeCommand::SetPipeline(id) => {
                let p = tables.compute_pipelines.get(&id).ok_or_else(|| Response::Error {
                    code: ErrorCode::UnknownResource,
                    message: format!("unknown ComputePipelineId({})", id.raw()),
                })?;
                pass.set_pipeline(p);
                Ok(())
            }
            ComputeCommand::SetBindGroup { index, group, offsets } => {
                let g = tables.bind_groups.get(&group).ok_or_else(|| Response::Error {
                    code: ErrorCode::UnknownResource,
                    message: format!("unknown BindGroupId({})", group.raw()),
                })?;
                pass.set_bind_group(index, g, &offsets);
                Ok(())
            }
            ComputeCommand::SetPushConstants { offset, data } => {
                pass.set_push_constants(offset, &data);
                Ok(())
            }
            ComputeCommand::DispatchWorkgroups { x, y, z } => {
                pass.dispatch_workgroups(x, y, z);
                Ok(())
            }
            ComputeCommand::DispatchWorkgroupsIndirect {
                indirect_buffer,
                indirect_offset,
            } => {
                let b = tables
                    .buffers
                    .get(&indirect_buffer)
                    .ok_or_else(|| Response::Error {
                        code: ErrorCode::UnknownResource,
                        message: format!("unknown BufferId({})", indirect_buffer.raw()),
                    })?;
                pass.dispatch_workgroups_indirect(b, indirect_offset);
                Ok(())
            }
            _ => Err(Response::Error {
                code: ErrorCode::InvalidArgument,
                message: "unknown compute command variant — protocol/server version skew?".into(),
            }),
        }
    }

    fn create_texture(&self, id: TextureId, desc: TextureDescriptor) -> Response {
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: desc.label.as_deref(),
            size: desc.size,
            mip_level_count: desc.mip_level_count,
            sample_count: desc.sample_count,
            dimension: desc.dimension,
            format: desc.format,
            usage: desc.usage,
            view_formats: &desc.view_formats,
        });
        self.tables.lock().unwrap().textures.insert(id, texture);
        Response::Ok
    }

    fn create_texture_view(
        &self,
        id: TextureViewId,
        texture: TextureId,
        desc: TextureViewDescriptor,
    ) -> Response {
        let tables = self.tables.lock().unwrap();
        let tex = match tables.textures.get(&texture) {
            Some(t) => t,
            None => {
                return Response::Error {
                    code: ErrorCode::UnknownResource,
                    message: format!("unknown TextureId({})", texture.raw()),
                };
            }
        };
        let view = tex.create_view(&wgpu::TextureViewDescriptor {
            label: desc.label.as_deref(),
            format: desc.format,
            dimension: desc.dimension,
            usage: desc.usage,
            aspect: desc.aspect,
            base_mip_level: desc.base_mip_level,
            mip_level_count: desc.mip_level_count,
            base_array_layer: desc.base_array_layer,
            array_layer_count: desc.array_layer_count,
        });
        drop(tables);
        self.tables.lock().unwrap().texture_views.insert(id, view);
        Response::Ok
    }

    fn create_sampler(&self, id: SamplerId, desc: SamplerDescriptor) -> Response {
        let sampler = self.device.create_sampler(&wgpu::SamplerDescriptor {
            label: desc.label.as_deref(),
            address_mode_u: desc.address_mode_u,
            address_mode_v: desc.address_mode_v,
            address_mode_w: desc.address_mode_w,
            mag_filter: desc.mag_filter,
            min_filter: desc.min_filter,
            mipmap_filter: desc.mipmap_filter,
            lod_min_clamp: desc.lod_min_clamp,
            lod_max_clamp: desc.lod_max_clamp,
            compare: desc.compare,
            anisotropy_clamp: desc.anisotropy_clamp,
            border_color: desc.border_color,
        });
        self.tables.lock().unwrap().samplers.insert(id, sampler);
        Response::Ok
    }

    fn create_render_pipeline(
        &self,
        id: RenderPipelineId,
        desc: RenderPipelineDescriptor,
    ) -> Response {
        let tables = self.tables.lock().unwrap();

        // Resolve referenced resources up front.
        let layout = match desc.layout {
            Some(l) => match tables.pipeline_layouts.get(&l) {
                Some(pl) => Some(pl),
                None => {
                    return Response::Error {
                        code: ErrorCode::UnknownResource,
                        message: format!("unknown PipelineLayoutId({})", l.raw()),
                    };
                }
            },
            None => None,
        };
        let vertex_module = match tables.shader_modules.get(&desc.vertex.module) {
            Some(m) => m,
            None => {
                return Response::Error {
                    code: ErrorCode::UnknownResource,
                    message: format!("unknown vertex ShaderModuleId({})", desc.vertex.module.raw()),
                };
            }
        };
        let fragment_module = match &desc.fragment {
            Some(f) => match tables.shader_modules.get(&f.module) {
                Some(m) => Some(m),
                None => {
                    return Response::Error {
                        code: ErrorCode::UnknownResource,
                        message: format!(
                            "unknown fragment ShaderModuleId({})",
                            f.module.raw()
                        ),
                    };
                }
            },
            None => None,
        };

        // Vertex buffer layouts need a stable backing for `&[VertexAttribute]`.
        // We collect into a Vec<VertexBufferLayout<'_>> first so the borrows in
        // the wgpu descriptor reference stable storage.
        let vertex_buffers: Vec<wgpu::VertexBufferLayout<'_>> = desc
            .vertex
            .buffers
            .iter()
            .map(|vbl| wgpu::VertexBufferLayout {
                array_stride: vbl.array_stride,
                step_mode: vbl.step_mode,
                attributes: &vbl.attributes,
            })
            .collect();

        let fragment_targets_storage: Option<Vec<Option<wgpu::ColorTargetState>>> = desc
            .fragment
            .as_ref()
            .map(|f| f.targets.clone());

        let fragment = match (&desc.fragment, fragment_module, fragment_targets_storage.as_ref()) {
            (Some(f), Some(module), Some(targets)) => Some(wgpu::FragmentState {
                module,
                entry_point: f.entry_point.as_deref(),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets,
            }),
            _ => None,
        };

        let pipeline = self
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: desc.label.as_deref(),
                layout,
                vertex: wgpu::VertexState {
                    module: vertex_module,
                    entry_point: desc.vertex.entry_point.as_deref(),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &vertex_buffers,
                },
                primitive: desc.primitive,
                depth_stencil: desc.depth_stencil,
                multisample: desc.multisample,
                fragment,
                multiview: desc.multiview.and_then(std::num::NonZeroU32::new),
                cache: None,
            });
        drop(tables);
        self.tables
            .lock()
            .unwrap()
            .render_pipelines
            .insert(id, pipeline);
        Response::Ok
    }
}

fn lookup_image_copy_buffer<'a>(
    tables: &'a ResourceTables,
    src: &ImageCopyBuffer,
) -> Result<wgpu::TexelCopyBufferInfo<'a>, Response> {
    let buffer = tables
        .buffers
        .get(&src.buffer)
        .ok_or_else(|| Response::Error {
            code: ErrorCode::UnknownResource,
            message: format!("unknown BufferId({})", src.buffer.raw()),
        })?;
    Ok(wgpu::TexelCopyBufferInfo {
        buffer,
        layout: src.layout,
    })
}

fn lookup_image_copy_texture<'a>(
    tables: &'a ResourceTables,
    src: &ImageCopyTexture,
) -> Result<wgpu::TexelCopyTextureInfo<'a>, Response> {
    let texture = tables
        .textures
        .get(&src.texture)
        .ok_or_else(|| Response::Error {
            code: ErrorCode::UnknownResource,
            message: format!("unknown TextureId({})", src.texture.raw()),
        })?;
    Ok(wgpu::TexelCopyTextureInfo {
        texture,
        mip_level: src.mip_level,
        origin: src.origin,
        aspect: src.aspect,
    })
}
