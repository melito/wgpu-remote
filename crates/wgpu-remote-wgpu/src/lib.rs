//! True wgpu drop-in for `wgpu-remote`. **Work in progress** — the public
//! [`install`] entry point is not yet implemented.
//!
//! Goal: hand a [`Connection`] to [`install`] and you get back a real
//! [`wgpu::Instance`]. From there your code looks identical to a stock wgpu
//! app — `Buffer`, `Device`, `Queue`, etc. are wgpu's own public types,
//! backed by a remote GPU through the same protocol the
//! [`wgpu_remote_client`] facade speaks.
//!
//! Built on wgpu 27's `custom` feature (which exposes
//! [`wgpu::custom::*`] interface traits and
//! [`wgpu::Instance::from_custom`]).
//!
//! [`Connection`]: wgpu_remote_transport::Connection

use wgpu_remote_transport::Connection;

/// Build a stock [`wgpu::Instance`] backed by the remote GPU on the far end
/// of `connection`.
///
/// Not yet implemented — see crate-level docs.
pub fn install<C>(_connection: C) -> wgpu::Instance
where
    C: Connection + Clone + 'static,
{
    unimplemented!(
        "wgpu-remote-wgpu install() is in progress; implement the wgpu::custom::*Interface \
         traits and wire them up via wgpu::Instance::from_custom"
    )
}
