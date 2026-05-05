//! Implementations of `wgpu::custom::*Interface` traits.
//!
//! Each public wgpu type (Instance, Adapter, Device, Buffer, …) gets a small
//! adapter struct here. The struct holds the corresponding `wgpu-remote-client`
//! facade type plus any state the custom-backend protocol demands (callback
//! storage, error channels). Trait impls forward to the facade.
//!
//! Currently wired:
//! - Instance, Adapter, Device, Queue construction.
//! - All `Device::create_*` paths (buffer, texture, texture view, sampler,
//!   shader module, bind group [layout], pipeline layout, compute/render
//!   pipeline). Each maps a wgpu descriptor through [`crate::translate`]
//!   then calls the facade's create method, which is sync + fire-and-forget
//!   thanks to the multiplexed-stream guarantee.
//!
//! Still stubbed: encoder + passes (need lifetime-bridging via
//! `Arc<Mutex<facade::CommandEncoder<C>>>`), `Queue::submit` /
//! `write_buffer`, `Buffer::map_async`. Each `unimplemented!()` carries a
//! precise reason so partial-build users see a readable failure.

use std::sync::Arc;

use wgpu::custom::{
    AdapterInterface, BindGroupInterface, BindGroupLayoutInterface, BufferInterface,
    CommandBufferInterface, CommandEncoderInterface, ComputePassInterface,
    ComputePipelineInterface, DeviceInterface, DispatchAdapter, DispatchBindGroup,
    DispatchBindGroupLayout, DispatchBlas, DispatchBuffer, DispatchBufferMappedRange,
    DispatchCommandBuffer, DispatchCommandEncoder, DispatchComputePass, DispatchComputePipeline,
    DispatchDevice, DispatchExternalTexture, DispatchPipelineCache, DispatchPipelineLayout,
    DispatchQuerySet, DispatchQueue, DispatchQueueWriteBuffer, DispatchRenderBundle,
    DispatchRenderBundleEncoder, DispatchRenderPass, DispatchRenderPipeline, DispatchSampler,
    DispatchShaderModule, DispatchSurface, DispatchTexture, DispatchTextureView, DispatchTlas,
    InstanceInterface, PipelineLayoutInterface, PopErrorScopeFuture, QueueInterface,
    RenderPassInterface, RenderPipelineInterface, RequestAdapterFuture, RequestDeviceFuture,
    SamplerInterface, ShaderCompilationInfoFuture, ShaderModuleInterface, TextureInterface,
    TextureViewInterface,
};
use wgpu_remote_client::{
    Adapter as FacadeAdapter,
    BindGroup as FacadeBindGroup, BindGroupLayout as FacadeBindGroupLayout,
    Buffer as FacadeBuffer, Client, ComputePipeline as FacadeComputePipeline,
    Device as FacadeDevice, Instance as FacadeInstance,
    PipelineLayout as FacadePipelineLayout, Queue as FacadeQueue,
    RenderPipeline as FacadeRenderPipeline, Sampler as FacadeSampler,
    ShaderModule as FacadeShaderModule, Texture as FacadeTexture,
    TextureView as FacadeTextureView,
};
use wgpu_remote_protocol::ids::{
    BindGroupId, BindGroupLayoutId, BufferId, ComputePipelineId, PipelineLayoutId,
    RenderPipelineId, SamplerId, ShaderModuleId, TextureId, TextureViewId,
};
use wgpu_remote_transport::Connection;

use crate::translate;

// ---------------------------------------------------------------------------
// Instance
// ---------------------------------------------------------------------------

pub(crate) struct Instance<C: Connection + Clone + 'static> {
    facade: Arc<FacadeInstance<C>>,
}

impl<C: Connection + Clone + 'static> Instance<C> {
    pub(crate) fn new(connection: C) -> Self {
        Self {
            facade: Arc::new(FacadeInstance::new(Client::new(connection))),
        }
    }
}

impl<C: Connection + Clone + 'static> std::fmt::Debug for Instance<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("wgpu_remote_wgpu::Instance").finish_non_exhaustive()
    }
}

impl<C: Connection + Clone + 'static> InstanceInterface for Instance<C> {
    fn new(_desc: &wgpu::InstanceDescriptor) -> Self
    where
        Self: Sized,
    {
        // Reachable only via `wgpu::Instance::new(...)`, which constructs
        // *some* backend by descriptor. Users of this crate come in via
        // `crate::install(connection)` instead — there's no connection to
        // synthesize from a descriptor alone.
        panic!(
            "wgpu_remote_wgpu::Instance cannot be constructed via wgpu::Instance::new(); \
             use wgpu_remote_wgpu::install(connection) instead"
        );
    }

    unsafe fn create_surface(
        &self,
        _target: wgpu::SurfaceTargetUnsafe,
    ) -> Result<DispatchSurface, wgpu::CreateSurfaceError> {
        // Surfaces present into a window in the *client* process. There's no
        // path for a remote GPU to write into the client's window — that's
        // what video streaming is for, and it's explicitly out of scope (see
        // README "Probably never").
        panic!(
            "wgpu-remote does not support Surface; the remote GPU cannot present \
             into a window on the client. Render to a texture and read back instead."
        );
    }

    fn request_adapter(
        &self,
        _options: &wgpu::RequestAdapterOptions<'_, '_>,
    ) -> std::pin::Pin<Box<dyn RequestAdapterFuture>> {
        let facade = Arc::clone(&self.facade);
        Box::pin(async move {
            match facade.request_adapter().await {
                Ok(a) => Ok(DispatchAdapter::custom(Adapter {
                    facade: Arc::new(a),
                })),
                Err(_) => Err(wgpu::wgt::RequestAdapterError::NotFound {
                    active_backends: wgpu::Backends::empty(),
                    requested_backends: wgpu::Backends::empty(),
                    supported_backends: wgpu::Backends::empty(),
                    no_fallback_backends: wgpu::Backends::empty(),
                    no_adapter_backends: wgpu::Backends::empty(),
                    incompatible_surface_backends: wgpu::Backends::empty(),
                }),
            }
        })
    }

    fn poll_all_devices(&self, _force_wait: bool) -> bool {
        // The remote backend doesn't need polling — the server is its own
        // event loop, and the client's reader task drives responses. Returning
        // true reports "all work submitted up to this point is complete from
        // *our* perspective"; the server may still be working.
        true
    }

    fn wgsl_language_features(&self) -> wgpu::WgslLanguageFeatures {
        // None of WGSL's optional language features (e.g. shader-f16,
        // packed-4x8) are exposed yet — the protocol ferries WGSL source
        // verbatim to the server, which decides what its compiler accepts.
        // Apps that probe for language features on the *client* will see
        // none.
        wgpu::WgslLanguageFeatures::empty()
    }
}

// ---------------------------------------------------------------------------
// Adapter
// ---------------------------------------------------------------------------

pub(crate) struct Adapter<C: Connection + Clone + 'static> {
    facade: Arc<FacadeAdapter<C>>,
}

impl<C: Connection + Clone + 'static> std::fmt::Debug for Adapter<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("wgpu_remote_wgpu::Adapter").finish_non_exhaustive()
    }
}

impl<C: Connection + Clone + 'static> AdapterInterface for Adapter<C> {
    fn request_device(
        &self,
        _desc: &wgpu::DeviceDescriptor<'_>,
    ) -> std::pin::Pin<Box<dyn RequestDeviceFuture>> {
        let facade = Arc::clone(&self.facade);
        Box::pin(async move {
            // wgpu's RequestDeviceError has no public constructor reachable
            // from a custom backend (its inner kind is gated to the wgpu_core
            // / WebGPU feature flags). On failure we have nothing to return,
            // so we panic with the underlying transport error. A future
            // version of the wgpu API may expose a custom-backend
            // construction path; until then this is the only option.
            let (device, queue) = facade
                .request_device()
                .await
                .unwrap_or_else(|e| panic!("wgpu-remote: request_device failed: {e}"));

            let device = Arc::new(device);
            let queue = Arc::new(queue);
            let error_handler = ErrorReporter::new();
            Ok((
                DispatchDevice::custom(Device {
                    facade: Arc::clone(&device),
                    error_handler: error_handler.clone(),
                }),
                DispatchQueue::custom(Queue {
                    facade: Arc::clone(&queue),
                    error_handler,
                    next_submission: std::sync::atomic::AtomicU64::new(1),
                }),
            ))
        })
    }

    fn is_surface_supported(&self, _surface: &DispatchSurface) -> bool {
        false
    }

    fn features(&self) -> wgpu::Features {
        // Curated set of features we can actually deliver across the wire.
        // For now: empty — apps that probe `Adapter::features()` will skip
        // any feature-gated path. As the protocol grows we add bits here.
        wgpu::Features::empty()
    }

    fn limits(&self) -> wgpu::Limits {
        // The remote device's actual limits aren't exposed yet; advertise
        // the wgpu defaults. A future protocol revision can ferry the
        // server's adapter limits at handshake time.
        wgpu::Limits::default()
    }

    fn downlevel_capabilities(&self) -> wgpu::DownlevelCapabilities {
        wgpu::DownlevelCapabilities::default()
    }

    fn get_info(&self) -> wgpu::AdapterInfo {
        // wgpu 27 has no `Backend::Custom`. Noop is the closest fit —
        // signals "this isn't one of the built-in backends" to apps that
        // condition on backend identity.
        wgpu::AdapterInfo {
            name: "wgpu-remote".to_string(),
            vendor: 0,
            device: 0,
            device_type: wgpu::DeviceType::Other,
            driver: "wgpu-remote".to_string(),
            driver_info: env!("CARGO_PKG_VERSION").to_string(),
            backend: wgpu::Backend::Noop,
        }
    }

    fn get_texture_format_features(
        &self,
        _format: wgpu::TextureFormat,
    ) -> wgpu::TextureFormatFeatures {
        // Conservative: no usages allowed, no flags set. A future protocol
        // revision can ferry the server's per-format capabilities at
        // handshake time.
        wgpu::TextureFormatFeatures {
            allowed_usages: wgpu::TextureUsages::empty(),
            flags: wgpu::TextureFormatFeatureFlags::empty(),
        }
    }

    fn get_presentation_timestamp(&self) -> wgpu::PresentationTimestamp {
        wgpu::PresentationTimestamp::INVALID_TIMESTAMP
    }
}

// ---------------------------------------------------------------------------
// Resource adapter structs
// ---------------------------------------------------------------------------
//
// Each one wraps the corresponding facade type and implements the matching
// `wgpu::custom::*Interface`. Most interfaces are *empty* (just `CommonTraits`
// = `Send + Sync + Debug + 'static`), so the impls below are mostly trivial.
//
// Why one struct per resource type even when the trait is empty: wgpu's
// dispatch layer downcasts via type identity. Two resources of different
// kinds must be distinct concrete types so `Buffer::as_custom::<Buffer<C>>()`
// can succeed without aliasing with, say, `Sampler::as_custom::<Sampler<C>>()`.

macro_rules! resource_adapter {
    (
        $name:ident, $facade:ident, $id_ty:ty, $interface:ident
    ) => {
        pub(crate) struct $name<C: Connection + Clone + 'static> {
            pub(crate) facade: $facade<C>,
        }

        impl<C: Connection + Clone + 'static> $name<C> {
            pub(crate) fn id(&self) -> $id_ty {
                self.facade.id()
            }
        }

        impl<C: Connection + Clone + 'static> std::fmt::Debug for $name<C> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.debug_struct(concat!("wgpu_remote_wgpu::", stringify!($name)))
                    .finish_non_exhaustive()
            }
        }

        impl<C: Connection + Clone + 'static> $interface for $name<C> {}
    };
}

resource_adapter!(BindGroupLayout, FacadeBindGroupLayout, BindGroupLayoutId, BindGroupLayoutInterface);
resource_adapter!(BindGroup, FacadeBindGroup, BindGroupId, BindGroupInterface);
resource_adapter!(Sampler, FacadeSampler, SamplerId, SamplerInterface);
resource_adapter!(TextureView, FacadeTextureView, TextureViewId, TextureViewInterface);
resource_adapter!(PipelineLayout, FacadePipelineLayout, PipelineLayoutId, PipelineLayoutInterface);

// Buffer, ShaderModule, ComputePipeline, RenderPipeline, Texture have
// non-empty interfaces — written out below.

pub(crate) struct Buffer<C: Connection + Clone + 'static> {
    pub(crate) facade: FacadeBuffer<C>,
    /// Stores the bytes returned by `map_async`, so subsequent
    /// `get_mapped_range` calls can hand them to the user. wgpu's lifecycle
    /// is map → get_mapped_range (potentially repeatedly) → unmap; we
    /// load the bytes once on map and keep them until unmap.
    mapped: Arc<StdMutex<Option<bytes::Bytes>>>,
}

impl<C: Connection + Clone + 'static> Buffer<C> {
    pub(crate) fn id(&self) -> BufferId {
        self.facade.id()
    }

    fn new(facade: FacadeBuffer<C>) -> Self {
        Self {
            facade,
            mapped: Arc::new(StdMutex::new(None)),
        }
    }
}

impl<C: Connection + Clone + 'static> std::fmt::Debug for Buffer<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("wgpu_remote_wgpu::Buffer").finish_non_exhaustive()
    }
}

impl<C: Connection + Clone + 'static> BufferInterface for Buffer<C> {
    fn map_async(
        &self,
        _mode: wgpu::MapMode,
        range: std::ops::Range<wgpu::wgt::BufferAddress>,
        callback: wgpu::custom::BufferMapCallback,
    ) {
        // Spawn a tokio task to do the readback, then invoke the
        // callback. The callback is `FnOnce + Send + 'static` — wgpu's
        // contract.
        //
        // MapMode is ignored: the protocol's MapBufferForRead only
        // supports read mode. Write-mapped buffers aren't bridged yet —
        // call write_buffer instead.
        let facade = self.facade.clone();
        let mapped = Arc::clone(&self.mapped);
        tokio::spawn(async move {
            let result = facade.read_range(range).await;
            match result {
                Ok(bytes) => {
                    *mapped.lock().unwrap() = Some(bytes);
                    callback(Ok(()));
                }
                Err(_) => {
                    callback(Err(wgpu::BufferAsyncError));
                }
            }
        });
    }

    fn get_mapped_range(
        &self,
        sub_range: std::ops::Range<wgpu::wgt::BufferAddress>,
    ) -> DispatchBufferMappedRange {
        let bytes = self
            .mapped
            .lock()
            .unwrap()
            .clone()
            .expect("get_mapped_range called before map_async completed");
        let start = sub_range.start as usize;
        let end = sub_range.end as usize;
        let slice = bytes.slice(start..end);
        DispatchBufferMappedRange::custom(BufferMappedRange { bytes: slice })
    }

    fn unmap(&self) {
        *self.mapped.lock().unwrap() = None;
    }

    fn destroy(&self) {
        // Facade Drop will ship Action::Destroy.
    }
}

/// Handed back from `Buffer::get_mapped_range`. Owns its slice so the
/// user can hold it across awaits without lifetime headaches.
pub(crate) struct BufferMappedRange {
    bytes: bytes::Bytes,
}

impl std::fmt::Debug for BufferMappedRange {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("wgpu_remote_wgpu::BufferMappedRange")
            .finish_non_exhaustive()
    }
}

impl wgpu::custom::BufferMappedRangeInterface for BufferMappedRange {
    fn slice(&self) -> &[u8] {
        &self.bytes
    }

    fn slice_mut(&mut self) -> &mut [u8] {
        // The wire-format readback is a `Bytes` (read-only). Writes
        // through the mapped range aren't supported yet — that's the
        // write-mapped buffer story.
        unimplemented!("BufferMappedRange is read-only in the current build")
    }
}

pub(crate) struct ShaderModule<C: Connection + Clone + 'static> {
    pub(crate) facade: FacadeShaderModule<C>,
}

impl<C: Connection + Clone + 'static> ShaderModule<C> {
    pub(crate) fn id(&self) -> ShaderModuleId {
        self.facade.id()
    }
}

impl<C: Connection + Clone + 'static> std::fmt::Debug for ShaderModule<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("wgpu_remote_wgpu::ShaderModule").finish_non_exhaustive()
    }
}

impl<C: Connection + Clone + 'static> ShaderModuleInterface for ShaderModule<C> {
    fn get_compilation_info(&self) -> std::pin::Pin<Box<dyn ShaderCompilationInfoFuture>> {
        // No shader compilation diagnostics are ferried over the wire yet.
        // Report success-with-no-messages.
        Box::pin(async {
            wgpu::CompilationInfo {
                messages: Vec::new(),
            }
        })
    }
}

pub(crate) struct ComputePipeline<C: Connection + Clone + 'static> {
    pub(crate) facade: FacadeComputePipeline<C>,
}

impl<C: Connection + Clone + 'static> ComputePipeline<C> {
    pub(crate) fn id(&self) -> ComputePipelineId {
        self.facade.id()
    }
}

impl<C: Connection + Clone + 'static> std::fmt::Debug for ComputePipeline<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("wgpu_remote_wgpu::ComputePipeline").finish_non_exhaustive()
    }
}

impl<C: Connection + Clone + 'static> ComputePipelineInterface for ComputePipeline<C> {
    fn get_bind_group_layout(&self, _index: u32) -> DispatchBindGroupLayout {
        // Pipeline-derived bind group layout introspection isn't supported
        // yet — the protocol doesn't ferry the per-pipeline layout slots
        // back to the client. Apps that build their own layouts up-front
        // (the common case) won't hit this path.
        unimplemented!(
            "ComputePipeline::get_bind_group_layout: pipeline-derived layouts \
             are not yet ferried back from the server"
        )
    }
}

pub(crate) struct RenderPipeline<C: Connection + Clone + 'static> {
    pub(crate) facade: FacadeRenderPipeline<C>,
}

impl<C: Connection + Clone + 'static> RenderPipeline<C> {
    pub(crate) fn id(&self) -> RenderPipelineId {
        self.facade.id()
    }
}

impl<C: Connection + Clone + 'static> std::fmt::Debug for RenderPipeline<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("wgpu_remote_wgpu::RenderPipeline").finish_non_exhaustive()
    }
}

impl<C: Connection + Clone + 'static> RenderPipelineInterface for RenderPipeline<C> {
    fn get_bind_group_layout(&self, _index: u32) -> DispatchBindGroupLayout {
        unimplemented!(
            "RenderPipeline::get_bind_group_layout: pipeline-derived layouts \
             are not yet ferried back from the server"
        )
    }
}

pub(crate) struct Texture<C: Connection + Clone + 'static> {
    pub(crate) facade: FacadeTexture<C>,
}

impl<C: Connection + Clone + 'static> Texture<C> {
    pub(crate) fn id(&self) -> TextureId {
        self.facade.id()
    }
}

impl<C: Connection + Clone + 'static> std::fmt::Debug for Texture<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("wgpu_remote_wgpu::Texture").finish_non_exhaustive()
    }
}

impl<C: Connection + Clone + 'static> TextureInterface for Texture<C> {
    fn create_view(&self, desc: &wgpu::TextureViewDescriptor<'_>) -> DispatchTextureView {
        let view = self.facade.create_view(translate::texture_view_descriptor(desc));
        DispatchTextureView::custom(TextureView { facade: view })
    }

    fn destroy(&self) {
        // Facade Drop ships destroy. See Buffer::destroy.
    }
}

// ---------------------------------------------------------------------------
// Device + Queue
// ---------------------------------------------------------------------------

pub(crate) struct Device<C: Connection + Clone + 'static> {
    pub(crate) facade: Arc<FacadeDevice<C>>,
    /// Where uncaptured validation errors go. Updated by
    /// `Device::on_uncaptured_error`. Default behavior matches wgpu's
    /// documented contract: errors translate into panics.
    pub(crate) error_handler: ErrorReporter,
}

/// Shared handle to the user-installed uncaptured-error handler. Cloned
/// into anything that needs to report unsupported-feature errors —
/// command encoders, passes — so they don't all need a back-reference to
/// the device.
#[derive(Clone)]
pub(crate) struct ErrorReporter {
    handler: Arc<StdMutex<Option<Arc<dyn wgpu::UncapturedErrorHandler>>>>,
}

impl ErrorReporter {
    pub(crate) fn new() -> Self {
        Self {
            handler: Arc::new(StdMutex::new(None)),
        }
    }

    fn install(&self, handler: Arc<dyn wgpu::UncapturedErrorHandler>) {
        *self.handler.lock().unwrap() = Some(handler);
    }

    /// Report an unsupported-feature use to the installed handler. If no
    /// handler is installed, panic — that's wgpu's documented default for
    /// uncaptured errors.
    pub(crate) fn report_unsupported(&self, what: &str) {
        let description = format!("wgpu-remote does not support {what}");
        self.report(wgpu::Error::Validation {
            source: Box::new(UnsupportedError(description.clone())),
            description,
        });
    }

    /// Route a fully-formed wgpu::Error through the installed handler,
    /// or panic if none is installed.
    pub(crate) fn report(&self, err: wgpu::Error) {
        let handler = self.handler.lock().unwrap().clone();
        match handler {
            Some(h) => h(err),
            None => panic!("uncaptured wgpu error: {err}"),
        }
    }
}

#[derive(Debug)]
struct UnsupportedError(String);

impl std::fmt::Display for UnsupportedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for UnsupportedError {}

#[derive(Debug)]
struct SubmitError(String);

impl std::fmt::Display for SubmitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for SubmitError {}

impl<C: Connection + Clone + 'static> std::fmt::Debug for Device<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("wgpu_remote_wgpu::Device").finish_non_exhaustive()
    }
}

impl<C: Connection + Clone + 'static> DeviceInterface for Device<C> {
    fn features(&self) -> wgpu::Features {
        wgpu::Features::empty()
    }

    fn limits(&self) -> wgpu::Limits {
        wgpu::Limits::default()
    }

    fn create_shader_module(
        &self,
        desc: wgpu::ShaderModuleDescriptor<'_>,
        _shader_bound_checks: wgpu::ShaderRuntimeChecks,
    ) -> DispatchShaderModule {
        // ShaderRuntimeChecks (bounds checks) is a per-build wgpu setting
        // that controls validation in the local backend. It's irrelevant
        // here — bounds checks happen on the *server* against the same
        // module, governed by the server's wgpu build.
        let module = self
            .facade
            .create_shader_module(translate::shader_module_descriptor(&desc));
        DispatchShaderModule::custom(ShaderModule { facade: module })
    }

    unsafe fn create_shader_module_passthrough(
        &self,
        _desc: &wgpu::ShaderModuleDescriptorPassthrough<'_>,
    ) -> DispatchShaderModule {
        // Shader passthrough hands wgpu a backend-native blob (SPIR-V via
        // VK_KHR_portability_subset, MSL, etc.). The remote backend has no
        // concept of "the local GPU's native shader format" — there's no
        // local GPU. Use the standard `create_shader_module` path with WGSL
        // or pre-compiled SPIR-V instead.
        panic!(
            "wgpu-remote does not support create_shader_module_passthrough; \
             use create_shader_module with WGSL or pre-compiled SPIR-V"
        );
    }

    fn create_bind_group_layout(
        &self,
        desc: &wgpu::BindGroupLayoutDescriptor<'_>,
    ) -> DispatchBindGroupLayout {
        let layout = self
            .facade
            .create_bind_group_layout(translate::bind_group_layout_descriptor(desc));
        DispatchBindGroupLayout::custom(BindGroupLayout { facade: layout })
    }

    fn create_bind_group(&self, desc: &wgpu::BindGroupDescriptor<'_>) -> DispatchBindGroup {
        // Translation can fail if a referenced resource didn't come from
        // this backend. A future pass routes that to `on_uncaptured_error`;
        // for now we panic with the precise diagnostic.
        let proto = translate::bind_group_descriptor::<C>(desc)
            .unwrap_or_else(|e| panic!("create_bind_group: {e}"));
        let bg = self.facade.create_bind_group(proto);
        DispatchBindGroup::custom(BindGroup { facade: bg })
    }

    fn create_pipeline_layout(
        &self,
        desc: &wgpu::PipelineLayoutDescriptor<'_>,
    ) -> DispatchPipelineLayout {
        let proto = translate::pipeline_layout_descriptor::<C>(desc)
            .unwrap_or_else(|e| panic!("create_pipeline_layout: {e}"));
        let layout = self.facade.create_pipeline_layout(proto);
        DispatchPipelineLayout::custom(PipelineLayout { facade: layout })
    }

    fn create_render_pipeline(
        &self,
        desc: &wgpu::RenderPipelineDescriptor<'_>,
    ) -> DispatchRenderPipeline {
        let proto = translate::render_pipeline_descriptor::<C>(desc)
            .unwrap_or_else(|e| panic!("create_render_pipeline: {e}"));
        let pipeline = self.facade.create_render_pipeline(proto);
        DispatchRenderPipeline::custom(RenderPipeline { facade: pipeline })
    }

    fn create_mesh_pipeline(
        &self,
        _desc: &wgpu::MeshPipelineDescriptor<'_>,
    ) -> DispatchRenderPipeline {
        unimplemented!("mesh shading not supported by wgpu-remote")
    }

    fn create_compute_pipeline(
        &self,
        desc: &wgpu::ComputePipelineDescriptor<'_>,
    ) -> DispatchComputePipeline {
        let proto = translate::compute_pipeline_descriptor::<C>(desc)
            .unwrap_or_else(|e| panic!("create_compute_pipeline: {e}"));
        let pipeline = self.facade.create_compute_pipeline(proto);
        DispatchComputePipeline::custom(ComputePipeline { facade: pipeline })
    }

    unsafe fn create_pipeline_cache(
        &self,
        _desc: &wgpu::PipelineCacheDescriptor<'_>,
    ) -> DispatchPipelineCache {
        unimplemented!("pipeline cache not supported by wgpu-remote")
    }

    fn create_buffer(&self, desc: &wgpu::BufferDescriptor<'_>) -> DispatchBuffer {
        let buf = self
            .facade
            .create_buffer(&translate::buffer_descriptor(desc));
        DispatchBuffer::custom(Buffer::new(buf))
    }

    fn create_texture(&self, desc: &wgpu::TextureDescriptor<'_>) -> DispatchTexture {
        let tex = self
            .facade
            .create_texture(&translate::texture_descriptor(desc));
        DispatchTexture::custom(Texture { facade: tex })
    }

    fn create_external_texture(
        &self,
        _desc: &wgpu::ExternalTextureDescriptor<'_>,
        _planes: &[&wgpu::TextureView],
    ) -> DispatchExternalTexture {
        unimplemented!("external textures not supported by wgpu-remote")
    }

    fn create_blas(
        &self,
        _desc: &wgpu::CreateBlasDescriptor<'_>,
        _sizes: wgpu::BlasGeometrySizeDescriptors,
    ) -> (Option<u64>, DispatchBlas) {
        unimplemented!("acceleration structures not supported by wgpu-remote")
    }

    fn create_tlas(&self, _desc: &wgpu::CreateTlasDescriptor<'_>) -> DispatchTlas {
        unimplemented!("acceleration structures not supported by wgpu-remote")
    }

    fn create_sampler(&self, desc: &wgpu::SamplerDescriptor<'_>) -> DispatchSampler {
        let sampler = self
            .facade
            .create_sampler(&translate::sampler_descriptor(desc));
        DispatchSampler::custom(Sampler { facade: sampler })
    }

    fn create_query_set(&self, _desc: &wgpu::QuerySetDescriptor<'_>) -> DispatchQuerySet {
        unimplemented!("query sets not supported by wgpu-remote")
    }

    fn create_command_encoder(
        &self,
        desc: &wgpu::CommandEncoderDescriptor<'_>,
    ) -> DispatchCommandEncoder {
        DispatchCommandEncoder::custom(CommandEncoder::<C>::new(
            desc.label.map(str::to_owned),
            self.error_handler.clone(),
        ))
    }

    fn create_render_bundle_encoder(
        &self,
        _desc: &wgpu::RenderBundleEncoderDescriptor<'_>,
    ) -> DispatchRenderBundleEncoder {
        unimplemented!("render bundles not supported by wgpu-remote")
    }

    fn set_device_lost_callback(&self, _device_lost_callback: wgpu::custom::BoxDeviceLostCallback) {
        // No-op until we wire the connection-loss path through to wgpu.
    }

    fn on_uncaptured_error(&self, handler: Arc<dyn wgpu::UncapturedErrorHandler>) {
        self.error_handler.install(handler);
    }

    fn push_error_scope(&self, _filter: wgpu::ErrorFilter) {
        // No-op until error scope propagation lands.
    }

    fn pop_error_scope(&self) -> std::pin::Pin<Box<dyn PopErrorScopeFuture>> {
        Box::pin(async { None })
    }

    unsafe fn start_graphics_debugger_capture(&self) {
        panic!("graphics debugger capture is server-side; not supported in wgpu-remote");
    }

    unsafe fn stop_graphics_debugger_capture(&self) {
        panic!("graphics debugger capture is server-side; not supported in wgpu-remote");
    }

    fn poll(
        &self,
        _poll_type: wgpu::wgt::PollType<u64>,
    ) -> Result<wgpu::PollStatus, wgpu::PollError> {
        // We don't track real submission status yet — claim "queue empty"
        // so callers don't loop.
        Ok(wgpu::PollStatus::QueueEmpty)
    }

    fn get_internal_counters(&self) -> wgpu::InternalCounters {
        wgpu::InternalCounters::default()
    }

    fn generate_allocator_report(&self) -> Option<wgpu::AllocatorReport> {
        None
    }

    fn destroy(&self) {
        // Facade Drop will tear down. wgpu calls destroy() when the user
        // explicitly destroys the device — for us that's a no-op signal,
        // since holding the handle alive is the only control we expose.
    }
}

// ---------------------------------------------------------------------------
// Queue
// ---------------------------------------------------------------------------

pub(crate) struct Queue<C: Connection + Clone + 'static> {
    pub(crate) facade: Arc<FacadeQueue<C>>,
    pub(crate) error_handler: ErrorReporter,
    /// Monotonic submission counter — wgpu's `Queue::submit` returns this
    /// as a `u64` opaque "submission index" that callers can later poll
    /// against. We don't track real completion; this is enough for apps
    /// that just want a unique increment per submit.
    pub(crate) next_submission: std::sync::atomic::AtomicU64,
}

impl<C: Connection + Clone + 'static> std::fmt::Debug for Queue<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("wgpu_remote_wgpu::Queue").finish_non_exhaustive()
    }
}

impl<C: Connection + Clone + 'static> QueueInterface for Queue<C> {
    fn write_buffer(
        &self,
        buffer: &DispatchBuffer,
        offset: wgpu::wgt::BufferAddress,
        data: &[u8],
    ) {
        let buf = buffer
            .as_custom::<Buffer<C>>()
            .expect("Queue::write_buffer received a buffer from a different backend");
        // The facade's write_buffer takes ownership of the bytes (via
        // Bytes::copy_from_slice this round-trips through an Arc<Vec>).
        // wgpu's interface gives us a borrow, so we have to copy.
        self.facade
            .write_buffer(&buf.facade, offset, bytes::Bytes::copy_from_slice(data));
    }

    fn create_staging_buffer(
        &self,
        _size: wgpu::wgt::BufferSize,
    ) -> Option<DispatchQueueWriteBuffer> {
        None
    }

    fn validate_write_buffer(
        &self,
        _buffer: &DispatchBuffer,
        _offset: wgpu::wgt::BufferAddress,
        _size: wgpu::wgt::BufferSize,
    ) -> Option<()> {
        Some(())
    }

    fn write_staging_buffer(
        &self,
        _buffer: &DispatchBuffer,
        _offset: wgpu::wgt::BufferAddress,
        _staging_buffer: &DispatchQueueWriteBuffer,
    ) {
        // create_staging_buffer always returns None, so no caller should
        // ever hand us a real DispatchQueueWriteBuffer. If they do, that's
        // a wgpu-remote misuse rather than an unsupported-feature error.
        unimplemented!(
            "write_staging_buffer is unreachable: create_staging_buffer always returns None"
        )
    }

    fn write_texture(
        &self,
        _texture: wgpu::wgt::TexelCopyTextureInfo<&wgpu::Texture>,
        _data: &[u8],
        _data_layout: wgpu::wgt::TexelCopyBufferLayout,
        _size: wgpu::wgt::Extent3d,
    ) {
        // Queue::write_texture is in WebGPU's standard surface — apps
        // can't degrade around its absence. Until the protocol carries
        // texture writes, leave as a not-yet-wired marker (distinct
        // from "unsupported" which routes through the error handler).
        unimplemented!("Queue::write_texture not yet wired through wgpu-remote-wgpu")
    }

    fn submit(
        &self,
        command_buffers: &mut dyn Iterator<Item = DispatchCommandBuffer>,
    ) -> u64 {
        // Drain the iterator into facade `CommandBuffer`s (the per-CB
        // recordings live in our adapter; we hand them to the facade
        // unchanged). The facade's `submit` is sync + fire-and-forget; the
        // encoded recording is shipped on the multiplexed stream.
        let recordings: Vec<wgpu_remote_client::CommandBuffer> = command_buffers
            .map(|cb| {
                let recorded = cb
                    .as_custom::<CommandBuffer>()
                    .expect("Queue::submit received a command buffer from a different backend");
                wgpu_remote_client::CommandBuffer::from_recording(recorded.recording.clone())
            })
            .collect();

        // submit() can fail at the bincode encode step. wgpu's
        // `Queue::submit` is sync and returns u64 unconditionally — we
        // can't surface the error in the return value. Route through the
        // installed uncaptured-error handler instead. The submission has
        // not occurred; the caller will likely dispatch follow-on work
        // referencing the (uncreated) submission index, which will then
        // also surface as a server-side `UnknownResource` and propagate
        // back through the error channel.
        if let Err(e) = self.facade.submit(recordings) {
            let description = format!("Queue::submit failed: {e}");
            self.error_handler.report(wgpu::Error::Internal {
                source: Box::new(SubmitError(description.clone())),
                description,
            });
        }

        // Synthesize a monotonic submission counter. wgpu uses this as a
        // poll-target index; with no submission tracking on our side we
        // keep an incrementing counter scoped to this queue.
        self.next_submission
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    }

    fn get_timestamp_period(&self) -> f32 {
        // No timestamp queries supported — periodic of 0 is the conventional
        // signal for "timestamps unavailable".
        0.0
    }

    fn on_submitted_work_done(&self, _callback: wgpu::custom::BoxSubmittedWorkDoneCallback) {
        // No-op until submission tracking lands. wgpu calls back when the
        // submission with this id completes; we'd ferry that off the wire.
    }

    fn compact_blas(&self, _blas: &DispatchBlas) -> (Option<u64>, DispatchBlas) {
        // BLAS adapters are unconstructible in this build (`create_blas`
        // panics), so this is unreachable in practice. Keep the panic.
        unimplemented!("acceleration structures not supported by wgpu-remote")
    }
}

// ---------------------------------------------------------------------------
// CommandEncoder + RenderPass + ComputePass + CommandBuffer
// ---------------------------------------------------------------------------
//
// We deliberately *don't* delegate encoding to the facade's own
// `CommandEncoder<C>`. The facade's encoder loans `&mut self` into its
// pass types, which doesn't fit through wgpu's owned `DispatchRenderPass` /
// `DispatchComputePass` model. Building a parallel accumulator using the
// protocol's `CommandBufferRecording` directly is structurally simpler and
// avoids needing an `Arc<Mutex<facade::CommandEncoder>>` around a type the
// facade itself never wraps in one.
//
// On submission we still go through the facade's `Queue::submit`, which
// handles encoding + the multiplexed-stream send.

use std::sync::Mutex as StdMutex;
use wgpu_remote_protocol::commands::{
    CommandBufferRecording, ComputeCommand, EncoderCommand, RenderCommand,
    RenderPassColorAttachment as ProtoColorAttachment,
    RenderPassDepthStencilAttachment as ProtoDepthAttachment,
};

/// Generic over `<C>` even though the encoder itself doesn't hold a
/// connection — the parameter is what lets the `CommandEncoderInterface`
/// impl resolve `Buffer<C>`, `BindGroup<C>`, etc. for its `as_custom`
/// extractions, and what makes wgpu's `as_custom::<CommandEncoder<C>>()`
/// disambiguate per-connection-type.
///
/// State lives behind an `Arc<Inner>` so pass adapters can hold a
/// back-reference and flush their accumulated commands on Drop.
pub(crate) struct CommandEncoder<C: Connection + Clone + 'static> {
    inner: Arc<EncoderInner<C>>,
}

struct EncoderInner<C: Connection + Clone + 'static> {
    /// Mutex-wrapped because wgpu's `CommandEncoderInterface::finish`
    /// takes `&mut self` but the rest of its methods are `&self`. Recording
    /// into a `Vec` from `&self` requires interior mutability; we use the
    /// same mutex for `finish` to take the recording out.
    recording: StdMutex<CommandBufferRecording>,
    /// Cloned from the parent device. Pass adapters reach this via their
    /// `Arc<EncoderInner<C>>` parent reference and use it to report
    /// unsupported-feature uses.
    error_handler: ErrorReporter,
    _phantom: std::marker::PhantomData<fn() -> C>,
}

impl<C: Connection + Clone + 'static> CommandEncoder<C> {
    fn new(label: Option<String>, error_handler: ErrorReporter) -> Self {
        Self {
            inner: Arc::new(EncoderInner {
                recording: StdMutex::new(CommandBufferRecording {
                    label,
                    commands: Vec::new(),
                }),
                error_handler,
                _phantom: std::marker::PhantomData,
            }),
        }
    }
}

impl<C: Connection + Clone + 'static> EncoderInner<C> {
    fn push(&self, cmd: EncoderCommand) {
        self.recording.lock().unwrap().commands.push(cmd);
    }
}

impl<C: Connection + Clone + 'static> std::fmt::Debug for CommandEncoder<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("wgpu_remote_wgpu::CommandEncoder").finish_non_exhaustive()
    }
}

impl<C: Connection + Clone + 'static> CommandEncoderInterface for CommandEncoder<C> {
    fn copy_buffer_to_buffer(
        &self,
        source: &DispatchBuffer,
        source_offset: wgpu::wgt::BufferAddress,
        destination: &DispatchBuffer,
        destination_offset: wgpu::wgt::BufferAddress,
        copy_size: Option<wgpu::wgt::BufferAddress>,
    ) {
        let src = source
            .as_custom::<Buffer<C>>()
            .expect("copy_buffer_to_buffer: foreign source");
        let dst = destination
            .as_custom::<Buffer<C>>()
            .expect("copy_buffer_to_buffer: foreign destination");
        // wgpu allows None to mean "to end of source". The protocol wants
        // an explicit size — fall back to source.size - source_offset.
        let size = copy_size.unwrap_or_else(|| src.facade.size().saturating_sub(source_offset));
        self.inner.push(EncoderCommand::CopyBufferToBuffer {
            source: src.id(),
            source_offset,
            destination: dst.id(),
            destination_offset,
            size,
        });
    }

    fn copy_buffer_to_texture(
        &self,
        _source: wgpu::TexelCopyBufferInfo<'_>,
        _destination: wgpu::TexelCopyTextureInfo<'_>,
        _copy_size: wgpu::wgt::Extent3d,
    ) {
        unimplemented!("copy_buffer_to_texture not yet wired through the wgpu drop-in")
    }

    fn copy_texture_to_buffer(
        &self,
        _source: wgpu::TexelCopyTextureInfo<'_>,
        _destination: wgpu::TexelCopyBufferInfo<'_>,
        _copy_size: wgpu::wgt::Extent3d,
    ) {
        unimplemented!("copy_texture_to_buffer not yet wired through the wgpu drop-in")
    }

    fn copy_texture_to_texture(
        &self,
        _source: wgpu::TexelCopyTextureInfo<'_>,
        _destination: wgpu::TexelCopyTextureInfo<'_>,
        _copy_size: wgpu::wgt::Extent3d,
    ) {
        unimplemented!("copy_texture_to_texture not yet wired through the wgpu drop-in")
    }

    fn begin_compute_pass(&self, desc: &wgpu::ComputePassDescriptor<'_>) -> DispatchComputePass {
        DispatchComputePass::custom(ComputePass::<C>::new(
            desc.label.map(str::to_owned),
            Arc::clone(&self.inner),
        ))
    }

    fn begin_render_pass(&self, desc: &wgpu::RenderPassDescriptor<'_>) -> DispatchRenderPass {
        let color_attachments = desc
            .color_attachments
            .iter()
            .map(|maybe| {
                maybe.as_ref().map(|att| ProtoColorAttachment {
                    view: att
                        .view
                        .as_custom::<TextureView<C>>()
                        .expect("render pass color attachment: foreign view")
                        .id(),
                    depth_slice: att.depth_slice,
                    resolve_target: att.resolve_target.as_ref().map(|v| {
                        v.as_custom::<TextureView<C>>()
                            .expect("render pass resolve target: foreign view")
                            .id()
                    }),
                    ops: att.ops,
                })
            })
            .collect();
        let depth_stencil_attachment =
            desc.depth_stencil_attachment.as_ref().map(|att| ProtoDepthAttachment {
                view: att
                    .view
                    .as_custom::<TextureView<C>>()
                    .expect("render pass depth attachment: foreign view")
                    .id(),
                depth_ops: att.depth_ops,
                stencil_ops: att.stencil_ops,
            });
        DispatchRenderPass::custom(RenderPass::<C>::new(
            desc.label.map(str::to_owned),
            color_attachments,
            depth_stencil_attachment,
            Arc::clone(&self.inner),
        ))
    }

    fn finish(&mut self) -> DispatchCommandBuffer {
        // Take the recording out by swapping in an empty one. The encoder
        // shouldn't be reused after finish, but if it is, recording into
        // the empty replacement is a no-op as far as observable behavior.
        let recording = std::mem::take(&mut *self.inner.recording.lock().unwrap());
        DispatchCommandBuffer::custom(CommandBuffer { recording })
    }

    fn clear_texture(
        &self,
        _texture: &DispatchTexture,
        _subresource_range: &wgpu::ImageSubresourceRange,
    ) {
        unimplemented!("clear_texture not yet wired through the wgpu drop-in")
    }

    fn clear_buffer(
        &self,
        buffer: &DispatchBuffer,
        offset: wgpu::wgt::BufferAddress,
        size: Option<wgpu::wgt::BufferAddress>,
    ) {
        let buf = buffer
            .as_custom::<Buffer<C>>()
            .expect("clear_buffer: foreign buffer");
        self.inner.push(EncoderCommand::ClearBuffer {
            buffer: buf.id(),
            offset,
            size,
        });
    }

    fn insert_debug_marker(&self, _label: &str) {
        // Debug markers are server-side observability; not bridged yet.
    }

    fn push_debug_group(&self, _label: &str) {}
    fn pop_debug_group(&self) {}

    fn write_timestamp(&self, _query_set: &DispatchQuerySet, _query_index: u32) {
        unimplemented!("query sets not supported by wgpu-remote")
    }

    fn resolve_query_set(
        &self,
        _query_set: &DispatchQuerySet,
        _first_query: u32,
        _query_count: u32,
        _destination: &DispatchBuffer,
        _destination_offset: wgpu::wgt::BufferAddress,
    ) {
        unimplemented!("query sets not supported by wgpu-remote")
    }

    fn mark_acceleration_structures_built<'a>(
        &self,
        _blas: &mut dyn Iterator<Item = &'a wgpu::Blas>,
        _tlas: &mut dyn Iterator<Item = &'a wgpu::Tlas>,
    ) {
        self.inner
            .error_handler
            .report_unsupported("acceleration structures");
    }

    fn build_acceleration_structures<'a>(
        &self,
        _blas: &mut dyn Iterator<Item = &'a wgpu::BlasBuildEntry<'a>>,
        _tlas: &mut dyn Iterator<Item = &'a wgpu::Tlas>,
    ) {
        self.inner
            .error_handler
            .report_unsupported("acceleration structures");
    }

    fn transition_resources<'a>(
        &mut self,
        _buffer_transitions: &mut dyn Iterator<
            Item = wgpu::wgt::BufferTransition<&'a DispatchBuffer>,
        >,
        _texture_transitions: &mut dyn Iterator<
            Item = wgpu::wgt::TextureTransition<&'a DispatchTexture>,
        >,
    ) {
        // Resource transitions are an explicit-control feature. The remote
        // backend tracks transitions on the server; client-supplied hints
        // are advisory. No-op for now.
    }
}

// ---------------------------------------------------------------------------
// ComputePass adapter
// ---------------------------------------------------------------------------
//
// Each pass holds its own command vec plus an `Arc<CommandEncoder<C>>`
// pointing at the parent encoder. The wgpu public `RenderPass<'encoder>`
// borrow lifetime guarantees the encoder outlives the pass on the
// user-facing side, but at the dispatch layer we hold only owned types,
// so we keep an Arc reference. Drop of the pass adapter pushes the
// accumulated commands into the encoder's recording.

pub(crate) struct ComputePass<C: Connection + Clone + 'static> {
    label: Option<String>,
    commands: Vec<ComputeCommand>,
    parent: Arc<EncoderInner<C>>,
}

impl<C: Connection + Clone + 'static> ComputePass<C> {
    fn new(label: Option<String>, parent: Arc<EncoderInner<C>>) -> Self {
        Self {
            label,
            commands: Vec::new(),
            parent,
        }
    }
}

impl<C: Connection + Clone + 'static> std::fmt::Debug for ComputePass<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("wgpu_remote_wgpu::ComputePass").finish_non_exhaustive()
    }
}

impl<C: Connection + Clone + 'static> Drop for ComputePass<C> {
    fn drop(&mut self) {
        let label = self.label.take();
        let commands = std::mem::take(&mut self.commands);
        self.parent
            .push(EncoderCommand::BeginComputePass { label, commands });
    }
}

impl<C: Connection + Clone + 'static> ComputePassInterface for ComputePass<C> {
    fn set_pipeline(&mut self, pipeline: &DispatchComputePipeline) {
        let p = pipeline
            .as_custom::<ComputePipeline<C>>()
            .expect("ComputePass::set_pipeline: foreign pipeline");
        self.commands.push(ComputeCommand::SetPipeline(p.id()));
    }

    fn set_bind_group(
        &mut self,
        index: u32,
        bind_group: Option<&DispatchBindGroup>,
        offsets: &[wgpu::wgt::DynamicOffset],
    ) {
        let bg = bind_group
            .expect("ComputePass::set_bind_group: optional bind groups not yet supported")
            .as_custom::<BindGroup<C>>()
            .expect("ComputePass::set_bind_group: foreign bind group");
        self.commands.push(ComputeCommand::SetBindGroup {
            index,
            group: bg.id(),
            offsets: offsets.to_vec(),
        });
    }

    fn set_push_constants(&mut self, _offset: u32, _data: &[u8]) {
        unimplemented!("compute push constants not yet wired")
    }

    fn insert_debug_marker(&mut self, _label: &str) {}
    fn push_debug_group(&mut self, _group_label: &str) {}
    fn pop_debug_group(&mut self) {}

    fn write_timestamp(&mut self, _query_set: &DispatchQuerySet, _query_index: u32) {
        // QuerySet adapters are unconstructible (create_query_set panics),
        // so this is unreachable in practice.
        self.parent.error_handler.report_unsupported("query sets");
    }

    fn begin_pipeline_statistics_query(
        &mut self,
        _query_set: &DispatchQuerySet,
        _query_index: u32,
    ) {
        self.parent.error_handler.report_unsupported("query sets");
    }

    fn end_pipeline_statistics_query(&mut self) {
        self.parent.error_handler.report_unsupported("query sets");
    }

    fn dispatch_workgroups(&mut self, x: u32, y: u32, z: u32) {
        self.commands
            .push(ComputeCommand::DispatchWorkgroups { x, y, z });
    }

    fn dispatch_workgroups_indirect(
        &mut self,
        indirect_buffer: &DispatchBuffer,
        indirect_offset: wgpu::wgt::BufferAddress,
    ) {
        let buf = indirect_buffer
            .as_custom::<Buffer<C>>()
            .expect("dispatch_workgroups_indirect: foreign buffer");
        self.commands
            .push(ComputeCommand::DispatchWorkgroupsIndirect {
                indirect_buffer: buf.id(),
                indirect_offset,
            });
    }

    fn end(&mut self) {
        // No-op: pass commands flush back to the parent encoder via Drop.
        // Calling `end()` early is a hint, not a requirement.
    }
}

// ---------------------------------------------------------------------------
// RenderPass adapter
// ---------------------------------------------------------------------------

pub(crate) struct RenderPass<C: Connection + Clone + 'static> {
    label: Option<String>,
    color_attachments: Vec<Option<ProtoColorAttachment>>,
    depth_stencil_attachment: Option<ProtoDepthAttachment>,
    commands: Vec<RenderCommand>,
    parent: Arc<EncoderInner<C>>,
}

impl<C: Connection + Clone + 'static> RenderPass<C> {
    fn new(
        label: Option<String>,
        color_attachments: Vec<Option<ProtoColorAttachment>>,
        depth_stencil_attachment: Option<ProtoDepthAttachment>,
        parent: Arc<EncoderInner<C>>,
    ) -> Self {
        Self {
            label,
            color_attachments,
            depth_stencil_attachment,
            commands: Vec::new(),
            parent,
        }
    }
}

impl<C: Connection + Clone + 'static> std::fmt::Debug for RenderPass<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("wgpu_remote_wgpu::RenderPass").finish_non_exhaustive()
    }
}

impl<C: Connection + Clone + 'static> Drop for RenderPass<C> {
    fn drop(&mut self) {
        let label = self.label.take();
        let color_attachments = std::mem::take(&mut self.color_attachments);
        let depth_stencil_attachment = self.depth_stencil_attachment.take();
        let commands = std::mem::take(&mut self.commands);
        self.parent.push(EncoderCommand::BeginRenderPass {
            label,
            color_attachments,
            depth_stencil_attachment,
            commands,
        });
    }
}

impl<C: Connection + Clone + 'static> RenderPassInterface for RenderPass<C> {
    fn set_pipeline(&mut self, pipeline: &DispatchRenderPipeline) {
        let p = pipeline
            .as_custom::<RenderPipeline<C>>()
            .expect("RenderPass::set_pipeline: foreign pipeline");
        self.commands.push(RenderCommand::SetPipeline(p.id()));
    }

    fn set_bind_group(
        &mut self,
        index: u32,
        bind_group: Option<&DispatchBindGroup>,
        offsets: &[wgpu::wgt::DynamicOffset],
    ) {
        let bg = bind_group
            .expect("RenderPass::set_bind_group: optional bind groups not yet supported")
            .as_custom::<BindGroup<C>>()
            .expect("RenderPass::set_bind_group: foreign bind group");
        self.commands.push(RenderCommand::SetBindGroup {
            index,
            group: bg.id(),
            offsets: offsets.to_vec(),
        });
    }

    fn set_index_buffer(
        &mut self,
        buffer: &DispatchBuffer,
        index_format: wgpu::IndexFormat,
        offset: wgpu::wgt::BufferAddress,
        size: Option<wgpu::wgt::BufferSize>,
    ) {
        let buf = buffer
            .as_custom::<Buffer<C>>()
            .expect("set_index_buffer: foreign buffer");
        self.commands.push(RenderCommand::SetIndexBuffer {
            buffer: buf.id(),
            format: index_format,
            offset,
            size: size.map(|s| s.get()),
        });
    }

    fn set_vertex_buffer(
        &mut self,
        slot: u32,
        buffer: &DispatchBuffer,
        offset: wgpu::wgt::BufferAddress,
        size: Option<wgpu::wgt::BufferSize>,
    ) {
        let buf = buffer
            .as_custom::<Buffer<C>>()
            .expect("set_vertex_buffer: foreign buffer");
        self.commands.push(RenderCommand::SetVertexBuffer {
            slot,
            buffer: buf.id(),
            offset,
            size: size.map(|s| s.get()),
        });
    }

    fn set_push_constants(&mut self, _stages: wgpu::ShaderStages, _offset: u32, _data: &[u8]) {
        unimplemented!("render push constants not yet wired")
    }

    fn set_blend_constant(&mut self, _color: wgpu::Color) {
        unimplemented!("set_blend_constant not yet wired")
    }

    fn set_scissor_rect(&mut self, _x: u32, _y: u32, _width: u32, _height: u32) {
        unimplemented!("set_scissor_rect not yet wired")
    }

    fn set_viewport(
        &mut self,
        _x: f32,
        _y: f32,
        _width: f32,
        _height: f32,
        _min_depth: f32,
        _max_depth: f32,
    ) {
        unimplemented!("set_viewport not yet wired")
    }

    fn set_stencil_reference(&mut self, _reference: u32) {
        unimplemented!("set_stencil_reference not yet wired")
    }

    fn draw(&mut self, vertices: std::ops::Range<u32>, instances: std::ops::Range<u32>) {
        self.commands.push(RenderCommand::Draw {
            vertices,
            instances,
        });
    }

    fn draw_indexed(
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

    fn draw_mesh_tasks(&mut self, _x: u32, _y: u32, _z: u32) {
        self.parent.error_handler.report_unsupported("mesh shading");
    }

    fn draw_indirect(
        &mut self,
        _indirect_buffer: &DispatchBuffer,
        _indirect_offset: wgpu::wgt::BufferAddress,
    ) {
        // Indirect draws are part of stock WebGPU. Not unsupported in
        // principle — just not yet wired through the protocol.
        unimplemented!("draw_indirect not yet wired through wgpu-remote-wgpu")
    }

    fn draw_indexed_indirect(
        &mut self,
        _indirect_buffer: &DispatchBuffer,
        _indirect_offset: wgpu::wgt::BufferAddress,
    ) {
        unimplemented!("draw_indexed_indirect not yet wired through wgpu-remote-wgpu")
    }

    fn draw_mesh_tasks_indirect(
        &mut self,
        _indirect_buffer: &DispatchBuffer,
        _indirect_offset: wgpu::wgt::BufferAddress,
    ) {
        self.parent.error_handler.report_unsupported("mesh shading");
    }

    fn multi_draw_indirect(
        &mut self,
        _indirect_buffer: &DispatchBuffer,
        _indirect_offset: wgpu::wgt::BufferAddress,
        _count: u32,
    ) {
        self.parent.error_handler.report_unsupported("multi-draw");
    }

    fn multi_draw_indexed_indirect(
        &mut self,
        _indirect_buffer: &DispatchBuffer,
        _indirect_offset: wgpu::wgt::BufferAddress,
        _count: u32,
    ) {
        self.parent.error_handler.report_unsupported("multi-draw");
    }

    fn multi_draw_indirect_count(
        &mut self,
        _indirect_buffer: &DispatchBuffer,
        _indirect_offset: wgpu::wgt::BufferAddress,
        _count_buffer: &DispatchBuffer,
        _count_buffer_offset: wgpu::wgt::BufferAddress,
        _max_count: u32,
    ) {
        self.parent
            .error_handler
            .report_unsupported("multi-draw-count");
    }

    fn multi_draw_mesh_tasks_indirect(
        &mut self,
        _indirect_buffer: &DispatchBuffer,
        _indirect_offset: wgpu::wgt::BufferAddress,
        _count: u32,
    ) {
        self.parent.error_handler.report_unsupported("mesh shading");
    }

    fn multi_draw_indexed_indirect_count(
        &mut self,
        _indirect_buffer: &DispatchBuffer,
        _indirect_offset: wgpu::wgt::BufferAddress,
        _count_buffer: &DispatchBuffer,
        _count_buffer_offset: wgpu::wgt::BufferAddress,
        _max_count: u32,
    ) {
        self.parent
            .error_handler
            .report_unsupported("multi-draw-count");
    }

    fn multi_draw_mesh_tasks_indirect_count(
        &mut self,
        _indirect_buffer: &DispatchBuffer,
        _indirect_offset: wgpu::wgt::BufferAddress,
        _count_buffer: &DispatchBuffer,
        _count_buffer_offset: wgpu::wgt::BufferAddress,
        _max_count: u32,
    ) {
        self.parent.error_handler.report_unsupported("mesh shading");
    }

    fn insert_debug_marker(&mut self, _label: &str) {}
    fn push_debug_group(&mut self, _group_label: &str) {}
    fn pop_debug_group(&mut self) {}

    fn write_timestamp(&mut self, _query_set: &DispatchQuerySet, _query_index: u32) {
        self.parent.error_handler.report_unsupported("query sets");
    }

    fn begin_occlusion_query(&mut self, _query_index: u32) {
        self.parent
            .error_handler
            .report_unsupported("occlusion queries");
    }

    fn end_occlusion_query(&mut self) {
        self.parent
            .error_handler
            .report_unsupported("occlusion queries");
    }

    fn begin_pipeline_statistics_query(
        &mut self,
        _query_set: &DispatchQuerySet,
        _query_index: u32,
    ) {
        self.parent
            .error_handler
            .report_unsupported("pipeline statistics queries");
    }

    fn end_pipeline_statistics_query(&mut self) {
        self.parent
            .error_handler
            .report_unsupported("pipeline statistics queries");
    }

    fn execute_bundles(
        &mut self,
        _render_bundles: &mut dyn Iterator<Item = &DispatchRenderBundle>,
    ) {
        // RenderBundle adapters are unconstructible (create_render_bundle_encoder
        // panics), so this is unreachable in practice.
        self.parent
            .error_handler
            .report_unsupported("render bundles");
    }

    fn end(&mut self) {
        // No-op: see ComputePass::end.
    }
}

// ---------------------------------------------------------------------------
// CommandBuffer adapter (just a recording owner)
// ---------------------------------------------------------------------------

pub(crate) struct CommandBuffer {
    pub(crate) recording: CommandBufferRecording,
}

impl std::fmt::Debug for CommandBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("wgpu_remote_wgpu::CommandBuffer").finish_non_exhaustive()
    }
}

impl CommandBufferInterface for CommandBuffer {}
