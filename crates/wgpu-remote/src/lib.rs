//! Run [wgpu](https://github.com/gfx-rs/wgpu) workloads on a GPU that lives
//! on another machine.
//!
//! This is the convenience entry point. Add `wgpu-remote` and `wgpu` to your
//! `Cargo.toml` and connect in two lines:
//!
//! ## QUIC (direct connection with CA cert)
//!
//! ```rust,no_run
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let instance = wgpu_remote::connect_quic("127.0.0.1:4433", "ca-cert.der").await?;
//!
//! // From here, everything is plain wgpu.
//! let adapter = instance.request_adapter(&Default::default()).await.unwrap();
//! let (device, queue) = adapter.request_device(&Default::default()).await?;
//! # Ok(())
//! # }
//! ```
//!
//! ## iroh (NAT-traversed, no certs needed)
//!
//! ```rust,no_run
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let instance = wgpu_remote::connect_iroh("6c190b769dd5...").await?;
//!
//! let adapter = instance.request_adapter(&Default::default()).await.unwrap();
//! let (device, queue) = adapter.request_device(&Default::default()).await?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Lower-level access
//!
//! If you need more control over the connection (custom TLS, secret keys,
//! relay configuration), build the transport yourself and hand it to
//! [`install`]:
//!
//! ```rust,no_run
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! use wgpu_remote::transport::quic::QuicEndpoint;
//! use rustls::pki_types::CertificateDer;
//!
//! let cert = CertificateDer::from(std::fs::read("ca-cert.der")?);
//! let endpoint = QuicEndpoint::client(cert)?;
//! let connection = endpoint.connect("127.0.0.1:4433".parse()?, "localhost").await?;
//! let instance = wgpu_remote::install(connection);
//! # Ok(())
//! # }
//! ```

use std::net::SocketAddr;
use std::path::Path;

/// Re-export the transport layer for lower-level use.
pub use wgpu_remote_transport as transport;

/// Re-export the iroh transport for lower-level use.
#[cfg(feature = "iroh")]
pub use wgpu_remote_transport_iroh as iroh;

/// Build a `wgpu::Instance` from any [`transport::Connection`].
///
/// This is the low-level entry point — use [`connect_quic`] or
/// [`connect_iroh`] for the common cases.
pub use wgpu_remote_wgpu::install;

/// Connect to a wgpu-remote server over QUIC and return a `wgpu::Instance`.
///
/// `addr` is the server's `IP:port` (e.g. `"127.0.0.1:4433"`).
/// `ca_cert_path` is the path to the CA certificate (DER format).
///
/// ```rust,no_run
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let instance = wgpu_remote::connect_quic("127.0.0.1:4433", "ca-cert.der").await?;
/// # Ok(())
/// # }
/// ```
#[cfg(feature = "quic")]
pub async fn connect_quic(
    addr: &str,
    ca_cert_path: impl AsRef<Path>,
) -> Result<wgpu::Instance, Error> {
    use rustls::pki_types::CertificateDer;
    use wgpu_remote_transport::quic::QuicEndpoint;

    let _ = rustls::crypto::ring::default_provider().install_default();

    let cert_bytes = std::fs::read(ca_cert_path.as_ref()).map_err(|e| Error::Io(e))?;
    let endpoint = QuicEndpoint::client(CertificateDer::from(cert_bytes))
        .map_err(|e| Error::Transport(e.to_string()))?;

    let socket_addr: SocketAddr = addr
        .parse()
        .map_err(|e| Error::Transport(format!("invalid address: {e}")))?;

    let connection = endpoint
        .connect(socket_addr, "localhost")
        .await
        .map_err(|e| Error::Transport(e.to_string()))?;

    // Keep the endpoint alive for the connection's lifetime.
    std::mem::forget(endpoint);

    Ok(install(connection))
}

/// Connect to a wgpu-remote server over QUIC, specifying the server name
/// for TLS verification.
///
/// Use this when the server cert has a SAN other than `localhost`.
#[cfg(feature = "quic")]
pub async fn connect_quic_with_server_name(
    addr: &str,
    ca_cert_path: impl AsRef<Path>,
    server_name: &str,
) -> Result<wgpu::Instance, Error> {
    use rustls::pki_types::CertificateDer;
    use wgpu_remote_transport::quic::QuicEndpoint;

    let _ = rustls::crypto::ring::default_provider().install_default();

    let cert_bytes = std::fs::read(ca_cert_path.as_ref()).map_err(|e| Error::Io(e))?;
    let endpoint = QuicEndpoint::client(CertificateDer::from(cert_bytes))
        .map_err(|e| Error::Transport(e.to_string()))?;

    let socket_addr: SocketAddr = addr
        .parse()
        .map_err(|e| Error::Transport(format!("invalid address: {e}")))?;

    let connection = endpoint
        .connect(socket_addr, server_name)
        .await
        .map_err(|e| Error::Transport(e.to_string()))?;

    std::mem::forget(endpoint);

    Ok(install(connection))
}

/// Connect to a wgpu-remote server over iroh and return a `wgpu::Instance`.
///
/// `endpoint_id` is the server's iroh endpoint ID (a hex string printed by
/// the server on startup). No certificates or IP addresses needed — iroh
/// handles discovery and NAT traversal automatically.
///
/// ```rust,no_run
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let instance = wgpu_remote::connect_iroh("6c190b769dd5...").await?;
/// # Ok(())
/// # }
/// ```
#[cfg(feature = "iroh")]
pub async fn connect_iroh(endpoint_id: &str) -> Result<wgpu::Instance, Error> {
    use wgpu_remote_transport::Transport;
    use wgpu_remote_transport_iroh::{EndpointAddr, IrohEndpoint};

    let _ = rustls::crypto::ring::default_provider().install_default();

    let remote_id = endpoint_id
        .parse()
        .map_err(|e| Error::Transport(format!("invalid endpoint ID: {e}")))?;

    let ep = IrohEndpoint::with_discovery()
        .await
        .map_err(|e| Error::Transport(e.to_string()))?;
    ep.endpoint().online().await;

    let conn = ep
        .dial(EndpointAddr::new(remote_id))
        .await
        .map_err(|e| Error::Transport(e.to_string()))?;

    std::mem::forget(ep);

    Ok(install(conn))
}

/// Errors returned by the convenience connect functions.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("transport: {0}")]
    Transport(String),
}
