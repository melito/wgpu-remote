//! Replay engine. Owns a `wgpu::Instance` + `Device` + `Queue` and a set of
//! [`ResourceTables`](crate::tables::ResourceTables); receives [`Frame`]s,
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
    commands::{CommandBufferRecording, ComputeCommand, EncoderCommand},
    descriptors::{
        BindGroupDescriptor, BindGroupLayoutDescriptor, BindingResource, ComputePipelineDescriptor,
        PipelineLayoutDescriptor, ShaderModuleDescriptor, ShaderSource,
    },
    ids::{
        BindGroupId, BindGroupLayoutId, BufferId, ComputePipelineId, PipelineLayoutId, ResourceId,
        ShaderModuleId,
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

            // Still stubbed for v1.1 — return Ok to keep the protocol moving.
            Action::CreateTexture { .. }
            | Action::CreateTextureView { .. }
            | Action::CreateSampler { .. }
            | Action::CreateRenderPipeline { .. }
            | Action::CreateCommandEncoder { .. }
            | Action::Destroy(_) => Some(Response::Ok),
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
            // Render passes, image copies, etc. are stubbed for v1.1.
            EncoderCommand::BeginRenderPass { .. }
            | EncoderCommand::CopyBufferToTexture { .. }
            | EncoderCommand::CopyTextureToBuffer { .. } => Err(Response::Error {
                code: ErrorCode::InvalidArgument,
                message: "encoder command not yet supported in v1".into(),
            }),
            // The enum is non_exhaustive; a future variant added without a
            // matching arm should fail loudly rather than be misinterpreted.
            _ => Err(Response::Error {
                code: ErrorCode::InvalidArgument,
                message: "unknown encoder command variant — protocol/server version skew?".into(),
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
}
