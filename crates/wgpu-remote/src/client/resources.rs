//! Typed resource handles. Each holds an ID, a connection-handle pair (for
//! both fire-and-forget destroy and the Buffer mapping path), and on Drop
//! ships an [`Action::Destroy`] in the background.
//!
//! Drop is sync; sending a destroy is async. We bridge the two by spawning
//! a fire-and-forget tokio task: we don't wait for the destroy round-trip,
//! and any error is logged. This requires the drop to happen inside a tokio
//! runtime — true for normal use, panics if the runtime is gone.
//!
//! Resources are `Arc`-shared internally so cloning a handle is cheap and
//! shared ownership doesn't trigger destroys until the *last* clone drops.

use std::ops::{Bound, RangeBounds};
use std::sync::Arc;

use bytes::Bytes;
use crate::protocol::{
    Action, Response,
    descriptors::TextureViewDescriptor,
    ids::{
        BindGroupId, BindGroupLayoutId, BufferId, ComputePipelineId, PipelineLayoutId,
        RenderPipelineId, ResourceId, SamplerId, ShaderModuleId, TextureId, TextureViewId,
    },
};
use crate::transport::Connection;

use crate::client::{Client, ClientError, ids::IdMinter};

/// Ships the destroy action without waiting for the response. Drop is
/// infallible — any error surfaces later as `UnknownResource` if anyone
/// references the freed ID.
fn spawn_destroy<C>(client: Arc<Client<C>>, resource: ResourceId)
where
    C: Connection + Clone + 'static,
{
    client.send(Action::Destroy(resource));
}

/// Generates a typed resource: `pub struct $name { inner: Arc<$inner_name> }`,
/// `id()` accessor, `Clone` (cheap Arc clone), `Drop` that ships an
/// `Action::Destroy`. The inner struct name must be unique per invocation,
/// hence the explicit `$inner_name` parameter — macro hygiene means we can't
/// reuse `Inner` across invocations in the same module.
macro_rules! resource {
    ($name:ident, $inner_name:ident, $id_ty:ident, $resource_variant:ident) => {
        pub struct $name<C: Connection + Clone + 'static> {
            inner: Arc<$inner_name<C>>,
        }

        struct $inner_name<C: Connection + Clone + 'static> {
            id: $id_ty,
            client: Arc<Client<C>>,
        }

        impl<C: Connection + Clone + 'static> $name<C> {
            pub(crate) fn new(id: $id_ty, client: Arc<Client<C>>) -> Self {
                Self {
                    inner: Arc::new($inner_name { id, client }),
                }
            }

            pub fn id(&self) -> $id_ty {
                self.inner.id
            }

            #[allow(dead_code)]
            pub(crate) fn client(&self) -> &Arc<Client<C>> {
                &self.inner.client
            }
        }

        impl<C: Connection + Clone + 'static> Clone for $name<C> {
            fn clone(&self) -> Self {
                Self {
                    inner: Arc::clone(&self.inner),
                }
            }
        }

        impl<C: Connection + Clone + 'static> Drop for $inner_name<C> {
            fn drop(&mut self) {
                spawn_destroy(
                    Arc::clone(&self.client),
                    ResourceId::$resource_variant(self.id),
                );
            }
        }
    };
}

pub struct Buffer<C: Connection + Clone + 'static> {
    inner: Arc<BufferInner<C>>,
}

struct BufferInner<C: Connection + Clone + 'static> {
    id: BufferId,
    size: u64,
    client: Arc<Client<C>>,
}

impl<C: Connection + Clone + 'static> Buffer<C> {
    pub(crate) fn new(id: BufferId, size: u64, client: Arc<Client<C>>) -> Self {
        Self {
            inner: Arc::new(BufferInner { id, size, client }),
        }
    }

    pub fn id(&self) -> BufferId {
        self.inner.id
    }

    pub fn size(&self) -> u64 {
        self.inner.size
    }

    /// Read a range of the buffer back to the client. Requires the buffer
    /// was created with `BufferUsages::MAP_READ` (and the range must fall
    /// within `0..self.size()`).
    ///
    /// Simpler shape than wgpu's `slice(...).map_async(...)` — we collapse
    /// the async-ack-then-get-mapped-range dance into a single
    /// `read_range(...)`, returning the bytes directly. A `map_async` mirror
    /// can be layered on top later if users want the wgpu-faithful API.
    pub async fn read_range<R: RangeBounds<u64>>(
        &self,
        range: R,
    ) -> Result<Bytes, ClientError> {
        let (offset, size) = bounds_to_offset_size(&range, self.inner.size);
        match self
            .inner
            .client
            .request(Action::MapBufferForRead {
                buffer: self.inner.id,
                offset,
                size,
            })
            .await?
        {
            Response::BufferData { data } => Ok(data),
            Response::Error { code, message } => Err(ClientError::ServerError(code, message)),
            other => Err(ClientError::ServerError(
                crate::protocol::responses::ErrorCode::Internal,
                format!("expected BufferData, got {other:?}"),
            )),
        }
    }

    /// Convenience: read the entire buffer.
    pub async fn read_all(&self) -> Result<Bytes, ClientError> {
        self.read_range(..).await
    }

    /// Upload bytes into this buffer at `offset`. Fire-and-forget, ordered on
    /// the wire like other actions. The server buffer must allow `COPY_DST`;
    /// the drop-in's `mapped_at_creation` emulation guarantees that.
    pub fn write_range(&self, offset: u64, data: Bytes) {
        self.inner.client.send(Action::WriteBuffer {
            buffer: self.inner.id,
            offset,
            data,
        });
    }
}

fn bounds_to_offset_size<R: RangeBounds<u64>>(range: &R, total: u64) -> (u64, u64) {
    let start = match range.start_bound() {
        Bound::Included(&n) => n,
        Bound::Excluded(&n) => n + 1,
        Bound::Unbounded => 0,
    };
    let end = match range.end_bound() {
        Bound::Included(&n) => n + 1,
        Bound::Excluded(&n) => n,
        Bound::Unbounded => total,
    };
    (start, end.saturating_sub(start))
}

impl<C: Connection + Clone + 'static> Clone for Buffer<C> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<C: Connection + Clone + 'static> Drop for BufferInner<C> {
    fn drop(&mut self) {
        spawn_destroy(Arc::clone(&self.client), ResourceId::Buffer(self.id));
    }
}

resource!(ShaderModule, ShaderModuleInner, ShaderModuleId, ShaderModule);
resource!(BindGroupLayout, BindGroupLayoutInner, BindGroupLayoutId, BindGroupLayout);
resource!(BindGroup, BindGroupInner, BindGroupId, BindGroup);
resource!(PipelineLayout, PipelineLayoutInner, PipelineLayoutId, PipelineLayout);
resource!(ComputePipeline, ComputePipelineInner, ComputePipelineId, ComputePipeline);
resource!(Sampler, SamplerInner, SamplerId, Sampler);
resource!(TextureView, TextureViewInner, TextureViewId, TextureView);

// RenderPipeline is hand-written (like Texture) because `get_bind_group_layout`
// must mint a new BindGroupLayoutId via the shared `IdMinter` and ship a
// DeriveRenderPipelineBindGroupLayout action. The macro doesn't carry IdMinter.

pub struct RenderPipeline<C: Connection + Clone + 'static> {
    inner: Arc<RenderPipelineInner<C>>,
}

struct RenderPipelineInner<C: Connection + Clone + 'static> {
    id: RenderPipelineId,
    client: Arc<Client<C>>,
    ids: Arc<IdMinter>,
}

impl<C: Connection + Clone + 'static> RenderPipeline<C> {
    pub(crate) fn new(id: RenderPipelineId, client: Arc<Client<C>>, ids: Arc<IdMinter>) -> Self {
        Self {
            inner: Arc::new(RenderPipelineInner { id, client, ids }),
        }
    }

    pub fn id(&self) -> RenderPipelineId {
        self.inner.id
    }

    /// The pipeline's bind-group layout at `index`. Sync — mints a fresh id and
    /// ships a derive action; the server calls `get_bind_group_layout` on the
    /// real pipeline and stores the result under that id. Ordering on the
    /// multiplexed stream guarantees any later reference to the id resolves.
    pub fn get_bind_group_layout(&self, index: u32) -> BindGroupLayout<C> {
        let bgl_id = self.inner.ids.mint_bind_group_layout();
        self.inner.client.send(Action::DeriveRenderPipelineBindGroupLayout {
            pipeline: self.inner.id,
            index,
            id: bgl_id,
        });
        BindGroupLayout::new(bgl_id, Arc::clone(&self.inner.client))
    }
}

impl<C: Connection + Clone + 'static> Clone for RenderPipeline<C> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<C: Connection + Clone + 'static> Drop for RenderPipelineInner<C> {
    fn drop(&mut self) {
        spawn_destroy(Arc::clone(&self.client), ResourceId::RenderPipeline(self.id));
    }
}

// Texture is hand-written because it needs `create_view`, which must mint a
// new TextureViewId via the shared `IdMinter` and ship a CreateTextureView
// action. The macro doesn't know about IdMinter or the parent device.

pub struct Texture<C: Connection + Clone + 'static> {
    inner: Arc<TextureInner<C>>,
}

struct TextureInner<C: Connection + Clone + 'static> {
    id: TextureId,
    client: Arc<Client<C>>,
    ids: Arc<IdMinter>,
}

impl<C: Connection + Clone + 'static> Texture<C> {
    pub(crate) fn new(id: TextureId, client: Arc<Client<C>>, ids: Arc<IdMinter>) -> Self {
        Self {
            inner: Arc::new(TextureInner { id, client, ids }),
        }
    }

    pub fn id(&self) -> TextureId {
        self.inner.id
    }

    /// Create a view onto this texture. Sync — the action is enqueued in
    /// order on the multiplexed stream, so any subsequent reference to the
    /// returned view's ID will see the view created on the server.
    pub fn create_view(&self, desc: TextureViewDescriptor) -> TextureView<C> {
        let view_id = self.inner.ids.mint_texture_view();
        self.inner.client.send(Action::CreateTextureView {
            id: view_id,
            texture: self.inner.id,
            desc,
        });
        TextureView::new(view_id, Arc::clone(&self.inner.client))
    }
}

impl<C: Connection + Clone + 'static> Clone for Texture<C> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<C: Connection + Clone + 'static> Drop for TextureInner<C> {
    fn drop(&mut self) {
        spawn_destroy(Arc::clone(&self.client), ResourceId::Texture(self.id));
    }
}
