//! Instance / Adapter / Device / Queue facade.
//!
//! v1: collapses Instance and Adapter into a single connection (matching the
//! server's "one engine = one device" assumption). The Adapter type is a
//! near-noop indirection so the user-side code reads the way wgpu code does:
//!
//! ```ignore
//! let instance = Instance::new(client);
//! let adapter = instance.request_adapter().await?;
//! let (device, queue) = adapter.request_device().await?;
//! ```

use std::sync::Arc;

use wgpu_remote_protocol::{
    Action, Response,
    descriptors::{
        BindGroupDescriptor, BindGroupLayoutDescriptor, BufferDescriptor,
        ComputePipelineDescriptor, PipelineLayoutDescriptor, RenderPipelineDescriptor,
        SamplerDescriptor, ShaderModuleDescriptor, TextureDescriptor,
    },
};
use wgpu_remote_transport::Connection;

use crate::{
    Client, ClientError,
    encoder::CommandEncoder,
    ids::IdMinter,
    resources::{
        BindGroup, BindGroupLayout, Buffer, ComputePipeline, PipelineLayout, RenderPipeline,
        Sampler, ShaderModule, Texture,
    },
};

pub struct Instance<C: Connection + Clone + 'static> {
    client: Arc<Client<C>>,
}

impl<C: Connection + Clone + 'static> Instance<C> {
    /// Wrap an already-built [`Client`]. For end-to-end use:
    /// `Instance::new(Client::new(connection))`.
    pub fn new(client: Client<C>) -> Self {
        Self {
            client: Arc::new(client),
        }
    }

    /// Send a Hello and request the single adapter the server exposes.
    pub async fn request_adapter(&self) -> Result<Adapter<C>, ClientError> {
        // The server collapses adapter selection in v1 — request_adapter is
        // an ack. We still send a Hello first so the protocol-version check
        // happens before any heavier work.
        match self
            .client
            .request(Action::Hello {
                protocol_version: wgpu_remote_protocol::PROTOCOL_VERSION,
            })
            .await?
        {
            Response::HelloAck { .. } => {}
            Response::Error { code, message } => {
                return Err(ClientError::ServerError(code, message));
            }
            other => return Err(unexpected("HelloAck", other)),
        }

        match self.client.request(Action::RequestAdapter {}).await? {
            Response::Ok => Ok(Adapter {
                client: Arc::clone(&self.client),
            }),
            Response::Error { code, message } => Err(ClientError::ServerError(code, message)),
            other => Err(unexpected("Ok", other)),
        }
    }

    pub fn client(&self) -> &Arc<Client<C>> {
        &self.client
    }
}

pub struct Adapter<C: Connection + Clone + 'static> {
    client: Arc<Client<C>>,
}

impl<C: Connection + Clone + 'static> Adapter<C> {
    pub async fn request_device(&self) -> Result<(Device<C>, Queue<C>), ClientError> {
        match self
            .client
            .request(Action::RequestDevice { label: None })
            .await?
        {
            Response::Ok => {}
            Response::Error { code, message } => {
                return Err(ClientError::ServerError(code, message));
            }
            other => return Err(unexpected("Ok", other)),
        }
        let inner = Arc::new(DeviceInner {
            client: Arc::clone(&self.client),
            ids: Arc::new(IdMinter::new()),
        });
        Ok((
            Device {
                inner: Arc::clone(&inner),
            },
            Queue { inner },
        ))
    }
}

/// Shared state for `Device` + `Queue`. Both need the client and to mint IDs;
/// neither is more authoritative than the other. The `IdMinter` is `Arc`'d
/// so resource handles that need to mint child IDs (e.g. `Texture::create_view`)
/// can clone it without going through the device.
pub(crate) struct DeviceInner<C: Connection + Clone + 'static> {
    pub(crate) client: Arc<Client<C>>,
    pub(crate) ids: Arc<IdMinter>,
}

pub struct Device<C: Connection + Clone + 'static> {
    inner: Arc<DeviceInner<C>>,
}

impl<C: Connection + Clone + 'static> Device<C> {
    // All `create_*` methods are sync. They mint an ID locally, ship the
    // creation Action via `Client::send` (fire-and-forget), and return the
    // typed handle. The multiplexed stream guarantees the server processes
    // creates before any subsequent reference to the ID — so the handle is
    // immediately usable in follow-up actions even though we didn't wait
    // for an Ok.
    //
    // Creation failures (validation errors, OOM) surface later as
    // `UnknownResource` on the next reference. Use `Client::request` with
    // a hand-built Action if you need explicit error handling.

    pub fn create_buffer(&self, desc: &BufferDescriptor) -> Buffer<C> {
        let id = self.inner.ids.mint_buffer();
        self.inner.client.send(Action::CreateBuffer {
            id,
            desc: desc.clone(),
        });
        Buffer::new(id, desc.size, Arc::clone(&self.inner.client))
    }

    pub fn create_shader_module(&self, desc: ShaderModuleDescriptor) -> ShaderModule<C> {
        let id = self.inner.ids.mint_shader_module();
        self.inner
            .client
            .send(Action::CreateShaderModule { id, desc });
        ShaderModule::new(id, Arc::clone(&self.inner.client))
    }

    pub fn create_bind_group_layout(
        &self,
        desc: BindGroupLayoutDescriptor,
    ) -> BindGroupLayout<C> {
        let id = self.inner.ids.mint_bind_group_layout();
        self.inner
            .client
            .send(Action::CreateBindGroupLayout { id, desc });
        BindGroupLayout::new(id, Arc::clone(&self.inner.client))
    }

    pub fn create_bind_group(&self, desc: BindGroupDescriptor) -> BindGroup<C> {
        let id = self.inner.ids.mint_bind_group();
        self.inner
            .client
            .send(Action::CreateBindGroup { id, desc });
        BindGroup::new(id, Arc::clone(&self.inner.client))
    }

    pub fn create_pipeline_layout(&self, desc: PipelineLayoutDescriptor) -> PipelineLayout<C> {
        let id = self.inner.ids.mint_pipeline_layout();
        self.inner
            .client
            .send(Action::CreatePipelineLayout { id, desc });
        PipelineLayout::new(id, Arc::clone(&self.inner.client))
    }

    pub fn create_compute_pipeline(&self, desc: ComputePipelineDescriptor) -> ComputePipeline<C> {
        let id = self.inner.ids.mint_compute_pipeline();
        self.inner
            .client
            .send(Action::CreateComputePipeline { id, desc });
        ComputePipeline::new(id, Arc::clone(&self.inner.client))
    }

    pub fn create_render_pipeline(&self, desc: RenderPipelineDescriptor) -> RenderPipeline<C> {
        let id = self.inner.ids.mint_render_pipeline();
        self.inner
            .client
            .send(Action::CreateRenderPipeline { id, desc });
        RenderPipeline::new(id, Arc::clone(&self.inner.client))
    }

    pub fn create_texture(&self, desc: &TextureDescriptor) -> Texture<C> {
        let id = self.inner.ids.mint_texture();
        self.inner.client.send(Action::CreateTexture {
            id,
            desc: desc.clone(),
        });
        Texture::new(
            id,
            Arc::clone(&self.inner.client),
            Arc::clone(&self.inner.ids),
        )
    }

    pub fn create_sampler(&self, desc: &SamplerDescriptor) -> Sampler<C> {
        let id = self.inner.ids.mint_sampler();
        self.inner.client.send(Action::CreateSampler {
            id,
            desc: desc.clone(),
        });
        Sampler::new(id, Arc::clone(&self.inner.client))
    }

    pub fn create_command_encoder(&self, label: Option<String>) -> CommandEncoder<C> {
        CommandEncoder::new(label, Arc::clone(&self.inner.client))
    }

    pub fn client(&self) -> &Arc<Client<C>> {
        &self.inner.client
    }
}

pub struct Queue<C: Connection + Clone + 'static> {
    inner: Arc<DeviceInner<C>>,
}

impl<C: Connection + Clone + 'static> Queue<C> {
    /// Schedule a buffer write. Sync — fire-and-forget on the multiplexed
    /// stream, ordered relative to subsequent actions.
    pub fn write_buffer(&self, buffer: &Buffer<C>, offset: u64, data: bytes::Bytes) {
        self.inner.client.send(Action::WriteBuffer {
            buffer: buffer.id(),
            offset,
            data,
        });
    }

    /// Submit one or more recorded command buffers. Sync — the actual
    /// `Action::Submit` is fired in order on the wire. Encoding errors at
    /// recording time are unrecoverable here: returns Err if any recording
    /// fails to serialize.
    pub fn submit<I: IntoIterator<Item = crate::encoder::CommandBuffer>>(
        &self,
        command_buffers: I,
    ) -> Result<(), ClientError> {
        let recordings: Result<Vec<bytes::Bytes>, bincode::error::EncodeError> = command_buffers
            .into_iter()
            .map(|cb| cb.into_recording().encode())
            .collect();
        let recordings = recordings?;
        self.inner.client.send(Action::Submit { recordings });
        Ok(())
    }

    pub fn client(&self) -> &Arc<Client<C>> {
        &self.inner.client
    }
}

// -- helpers ----------------------------------------------------------------

fn unexpected(want: &'static str, got: Response) -> ClientError {
    ClientError::ServerError(
        wgpu_remote_protocol::responses::ErrorCode::Internal,
        format!("expected {want}, got {got:?}"),
    )
}

