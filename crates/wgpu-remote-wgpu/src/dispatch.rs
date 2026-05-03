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
    ComputePipelineInterface, DeviceInterface, DispatchAdapter, DispatchBindGroup,
    DispatchBindGroupLayout, DispatchBlas, DispatchBuffer, DispatchBufferMappedRange,
    DispatchCommandBuffer, DispatchCommandEncoder, DispatchComputePipeline, DispatchDevice,
    DispatchExternalTexture, DispatchPipelineCache, DispatchPipelineLayout, DispatchQuerySet,
    DispatchQueue, DispatchQueueWriteBuffer, DispatchRenderBundleEncoder,
    DispatchRenderPipeline, DispatchSampler, DispatchShaderModule, DispatchSurface,
    DispatchTexture, DispatchTextureView, DispatchTlas, InstanceInterface,
    PipelineLayoutInterface, PopErrorScopeFuture, QueueInterface, RenderPipelineInterface,
    RequestAdapterFuture, RequestDeviceFuture, SamplerInterface, ShaderCompilationInfoFuture,
    ShaderModuleInterface, TextureInterface, TextureViewInterface,
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
            Ok((
                DispatchDevice::custom(Device {
                    facade: Arc::clone(&device),
                }),
                DispatchQueue::custom(Queue {
                    facade: Arc::clone(&queue),
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
}

impl<C: Connection + Clone + 'static> Buffer<C> {
    pub(crate) fn id(&self) -> BufferId {
        self.facade.id()
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
        _range: std::ops::Range<wgpu::wgt::BufferAddress>,
        _callback: wgpu::custom::BufferMapCallback,
    ) {
        unimplemented!("Buffer::map_async — callback dispatcher lands in a follow-up commit")
    }

    fn get_mapped_range(
        &self,
        _sub_range: std::ops::Range<wgpu::wgt::BufferAddress>,
    ) -> DispatchBufferMappedRange {
        unimplemented!("Buffer::get_mapped_range — depends on map_async")
    }

    fn unmap(&self) {
        unimplemented!("Buffer::unmap — depends on map_async")
    }

    fn destroy(&self) {
        // Facade Drop will ship Action::Destroy. Explicit destroy() from
        // wgpu is a hint we don't need — keeping the facade handle alive
        // is the only control surface we expose.
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
    #[allow(dead_code)]
    pub(crate) facade: Arc<FacadeDevice<C>>,
}

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
        DispatchBuffer::custom(Buffer { facade: buf })
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
        _desc: &wgpu::CommandEncoderDescriptor<'_>,
    ) -> DispatchCommandEncoder {
        unimplemented!("Device::create_command_encoder not yet implemented")
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

    fn on_uncaptured_error(&self, _handler: Arc<dyn wgpu::UncapturedErrorHandler>) {
        // No-op until NotSupported error routing lands.
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
    #[allow(dead_code)]
    pub(crate) facade: Arc<FacadeQueue<C>>,
}

impl<C: Connection + Clone + 'static> std::fmt::Debug for Queue<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("wgpu_remote_wgpu::Queue").finish_non_exhaustive()
    }
}

impl<C: Connection + Clone + 'static> QueueInterface for Queue<C> {
    fn write_buffer(
        &self,
        _buffer: &DispatchBuffer,
        _offset: wgpu::wgt::BufferAddress,
        _data: &[u8],
    ) {
        unimplemented!("Queue::write_buffer not yet implemented")
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
        unimplemented!("staging buffer writes not yet implemented")
    }

    fn write_texture(
        &self,
        _texture: wgpu::wgt::TexelCopyTextureInfo<&wgpu::Texture>,
        _data: &[u8],
        _data_layout: wgpu::wgt::TexelCopyBufferLayout,
        _size: wgpu::wgt::Extent3d,
    ) {
        unimplemented!("Queue::write_texture not yet implemented")
    }

    fn submit(
        &self,
        _command_buffers: &mut dyn Iterator<Item = DispatchCommandBuffer>,
    ) -> u64 {
        unimplemented!("Queue::submit not yet implemented")
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
        unimplemented!("acceleration structures not supported by wgpu-remote")
    }
}
