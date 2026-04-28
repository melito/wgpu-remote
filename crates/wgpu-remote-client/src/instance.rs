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
    /// Create a buffer. Async because v1 uses one stream per request, so
    /// follow-up actions that reference this buffer's ID could race the
    /// create on the server side. Awaiting the Ok here keeps the resource
    /// table populated before the user can use the handle. v1.1 will let us
    /// switch to fire-and-forget once the server multiplexes actions onto a
    /// single ordered stream per connection.
    pub async fn create_buffer(
        &self,
        desc: &BufferDescriptor,
    ) -> Result<Buffer<C>, ClientError> {
        let id = self.inner.ids.mint_buffer();
        ok(self
            .inner
            .client
            .request(Action::CreateBuffer {
                id,
                desc: desc.clone(),
            })
            .await?)?;
        Ok(Buffer::new(id, desc.size, Arc::clone(&self.inner.client)))
    }

    pub async fn create_shader_module(
        &self,
        desc: ShaderModuleDescriptor,
    ) -> Result<ShaderModule<C>, ClientError> {
        let id = self.inner.ids.mint_shader_module();
        ok(self.inner.client.request(Action::CreateShaderModule { id, desc }).await?)?;
        Ok(ShaderModule::new(id, Arc::clone(&self.inner.client)))
    }

    pub async fn create_bind_group_layout(
        &self,
        desc: BindGroupLayoutDescriptor,
    ) -> Result<BindGroupLayout<C>, ClientError> {
        let id = self.inner.ids.mint_bind_group_layout();
        ok(self
            .inner
            .client
            .request(Action::CreateBindGroupLayout { id, desc })
            .await?)?;
        Ok(BindGroupLayout::new(id, Arc::clone(&self.inner.client)))
    }

    pub async fn create_bind_group(
        &self,
        desc: BindGroupDescriptor,
    ) -> Result<BindGroup<C>, ClientError> {
        let id = self.inner.ids.mint_bind_group();
        ok(self
            .inner
            .client
            .request(Action::CreateBindGroup { id, desc })
            .await?)?;
        Ok(BindGroup::new(id, Arc::clone(&self.inner.client)))
    }

    pub async fn create_pipeline_layout(
        &self,
        desc: PipelineLayoutDescriptor,
    ) -> Result<PipelineLayout<C>, ClientError> {
        let id = self.inner.ids.mint_pipeline_layout();
        ok(self
            .inner
            .client
            .request(Action::CreatePipelineLayout { id, desc })
            .await?)?;
        Ok(PipelineLayout::new(id, Arc::clone(&self.inner.client)))
    }

    pub async fn create_compute_pipeline(
        &self,
        desc: ComputePipelineDescriptor,
    ) -> Result<ComputePipeline<C>, ClientError> {
        let id = self.inner.ids.mint_compute_pipeline();
        ok(self
            .inner
            .client
            .request(Action::CreateComputePipeline { id, desc })
            .await?)?;
        Ok(ComputePipeline::new(id, Arc::clone(&self.inner.client)))
    }

    pub async fn create_render_pipeline(
        &self,
        desc: RenderPipelineDescriptor,
    ) -> Result<RenderPipeline<C>, ClientError> {
        let id = self.inner.ids.mint_render_pipeline();
        ok(self
            .inner
            .client
            .request(Action::CreateRenderPipeline { id, desc })
            .await?)?;
        Ok(RenderPipeline::new(id, Arc::clone(&self.inner.client)))
    }

    pub async fn create_texture(
        &self,
        desc: &TextureDescriptor,
    ) -> Result<Texture<C>, ClientError> {
        let id = self.inner.ids.mint_texture();
        ok(self
            .inner
            .client
            .request(Action::CreateTexture {
                id,
                desc: desc.clone(),
            })
            .await?)?;
        Ok(Texture::new(
            id,
            Arc::clone(&self.inner.client),
            Arc::clone(&self.inner.ids),
        ))
    }

    pub async fn create_sampler(
        &self,
        desc: &SamplerDescriptor,
    ) -> Result<Sampler<C>, ClientError> {
        let id = self.inner.ids.mint_sampler();
        ok(self
            .inner
            .client
            .request(Action::CreateSampler {
                id,
                desc: desc.clone(),
            })
            .await?)?;
        Ok(Sampler::new(id, Arc::clone(&self.inner.client)))
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
    pub async fn write_buffer(
        &self,
        buffer: &Buffer<C>,
        offset: u64,
        data: bytes::Bytes,
    ) -> Result<(), ClientError> {
        ok(self
            .inner
            .client
            .request(Action::WriteBuffer {
                buffer: buffer.id(),
                offset,
                data,
            })
            .await?)?;
        Ok(())
    }

    pub async fn submit<I: IntoIterator<Item = crate::encoder::CommandBuffer>>(
        &self,
        command_buffers: I,
    ) -> Result<(), ClientError> {
        let recordings: Result<Vec<bytes::Bytes>, bincode::error::EncodeError> = command_buffers
            .into_iter()
            .map(|cb| cb.into_recording().encode())
            .collect();
        let recordings = recordings?;
        ok(self
            .inner
            .client
            .request(Action::Submit { recordings })
            .await?)?;
        Ok(())
    }

    pub fn client(&self) -> &Arc<Client<C>> {
        &self.inner.client
    }
}

// -- helpers ----------------------------------------------------------------

fn ok(r: Response) -> Result<(), ClientError> {
    match r {
        Response::Ok => Ok(()),
        Response::Error { code, message } => Err(ClientError::ServerError(code, message)),
        other => Err(unexpected("Ok", other)),
    }
}

fn unexpected(want: &'static str, got: Response) -> ClientError {
    ClientError::ServerError(
        wgpu_remote_protocol::responses::ErrorCode::Internal,
        format!("expected {want}, got {got:?}"),
    )
}

