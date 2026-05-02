//! True wgpu drop-in for `wgpu-remote`.
//!
//! Hand a [`Connection`] to [`install`] and you get back a real
//! [`wgpu::Instance`]. From there your code looks identical to a stock wgpu
//! app — `Buffer`, `Device`, `Queue`, etc. are wgpu's own public types,
//! backed by a remote GPU through the same protocol the
//! [`wgpu_remote_client`] facade speaks.
//!
//! Built on wgpu 27's `custom` feature: this crate implements the
//! [`wgpu::custom`] interface traits and hands the resulting
//! `InstanceInterface` to [`wgpu::Instance::from_custom`].
//!
//! ## Status
//!
//! Adapter / Device / Queue *construction* is wired up. Resource creates,
//! encoders, render/compute passes, and `map_async` are not yet implemented
//! — calling them panics with a clear message. The end-to-end compute test
//! will land alongside those impls.
//!
//! ## Unsupported features
//!
//! - **Surface / present** — the remote GPU can't write into a window in the
//!   client process. Render to a texture and read back instead.
//! - **Acceleration structures, mesh shading, render bundles, query sets,
//!   external textures, pipeline cache** — not yet bridged across the wire.
//!   `Adapter::features()` advertises them as absent.
//! - **Shader passthrough** (`create_shader_module_passthrough`) — there's
//!   no local GPU whose native format we'd need.
//! - **Graphics debugger capture** — debugging the remote GPU is a
//!   server-side concern.
//!
//! [`Connection`]: wgpu_remote_transport::Connection

mod dispatch;

use wgpu_remote_transport::Connection;

/// Build a stock [`wgpu::Instance`] backed by the remote GPU on the far end
/// of `connection`.
pub fn install<C>(connection: C) -> wgpu::Instance
where
    C: Connection + Clone + 'static,
{
    wgpu::Instance::from_custom(dispatch::Instance::new(connection))
}
