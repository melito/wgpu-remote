//! Implementations of `wgpu::custom::*Interface` traits.
//!
//! Each public wgpu type (Instance, Adapter, Device, Buffer, …) gets a small
//! adapter struct here. The struct holds the corresponding `wgpu-remote-client`
//! facade type plus any state the custom-backend protocol demands (callback
//! storage, error channels). Trait impls forward to the facade.
//!
//! Currently scaffolded down to `request_device` (returning a stub Device +
//! Queue). Resource creates, encoders, passes, and `map_async` land in
//! follow-up commits — each unimplemented method is marked with a
//! `unimplemented!()` carrying a precise reason so partial-build users see a
//! readable failure rather than a mystery panic.

use std::sync::Arc;

use wgpu::custom::{
    AdapterInterface, DeviceInterface, DispatchAdapter, DispatchBindGroup,
    DispatchBindGroupLayout, DispatchBlas, DispatchBuffer, DispatchCommandBuffer,
    DispatchCommandEncoder, DispatchComputePipeline, DispatchDevice, DispatchExternalTexture,
    DispatchPipelineCache, DispatchPipelineLayout, DispatchQuerySet, DispatchQueue,
    DispatchQueueWriteBuffer, DispatchRenderBundleEncoder, DispatchRenderPipeline,
    DispatchSampler, DispatchShaderModule, DispatchSurface, DispatchTexture, DispatchTlas,
    InstanceInterface, PopErrorScopeFuture, QueueInterface, RequestAdapterFuture,
    RequestDeviceFuture,
};
use wgpu_remote_client::{
    Adapter as FacadeAdapter, Client, Device as FacadeDevice, Instance as FacadeInstance,
    Queue as FacadeQueue,
};
use wgpu_remote_transport::Connection;

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
// Device + Queue (stub: methods unimplemented)
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
        _desc: wgpu::ShaderModuleDescriptor<'_>,
        _shader_bound_checks: wgpu::ShaderRuntimeChecks,
    ) -> DispatchShaderModule {
        unimplemented!("Device::create_shader_module not yet implemented")
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
        _desc: &wgpu::BindGroupLayoutDescriptor<'_>,
    ) -> DispatchBindGroupLayout {
        unimplemented!("Device::create_bind_group_layout not yet implemented")
    }

    fn create_bind_group(
        &self,
        _desc: &wgpu::BindGroupDescriptor<'_>,
    ) -> DispatchBindGroup {
        unimplemented!("Device::create_bind_group not yet implemented")
    }

    fn create_pipeline_layout(
        &self,
        _desc: &wgpu::PipelineLayoutDescriptor<'_>,
    ) -> DispatchPipelineLayout {
        unimplemented!("Device::create_pipeline_layout not yet implemented")
    }

    fn create_render_pipeline(
        &self,
        _desc: &wgpu::RenderPipelineDescriptor<'_>,
    ) -> DispatchRenderPipeline {
        unimplemented!("Device::create_render_pipeline not yet implemented")
    }

    fn create_mesh_pipeline(
        &self,
        _desc: &wgpu::MeshPipelineDescriptor<'_>,
    ) -> DispatchRenderPipeline {
        unimplemented!("mesh shading not supported by wgpu-remote")
    }

    fn create_compute_pipeline(
        &self,
        _desc: &wgpu::ComputePipelineDescriptor<'_>,
    ) -> DispatchComputePipeline {
        unimplemented!("Device::create_compute_pipeline not yet implemented")
    }

    unsafe fn create_pipeline_cache(
        &self,
        _desc: &wgpu::PipelineCacheDescriptor<'_>,
    ) -> DispatchPipelineCache {
        unimplemented!("pipeline cache not supported by wgpu-remote")
    }

    fn create_buffer(&self, _desc: &wgpu::BufferDescriptor<'_>) -> DispatchBuffer {
        unimplemented!("Device::create_buffer not yet implemented")
    }

    fn create_texture(&self, _desc: &wgpu::TextureDescriptor<'_>) -> DispatchTexture {
        unimplemented!("Device::create_texture not yet implemented")
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

    fn create_sampler(&self, _desc: &wgpu::SamplerDescriptor<'_>) -> DispatchSampler {
        unimplemented!("Device::create_sampler not yet implemented")
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
