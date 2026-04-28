//! Pluggable transport for wgpu-remote.
//!
//! The trait surface is shaped to fit WebTransport's capability floor (bidi
//! streams + optional datagrams), so quinn, iroh, and a future browser-side
//! `wtransport` impl all slot in without redesign.

use std::future::Future;

use bytes::Bytes;
use thiserror::Error;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite};

#[cfg(feature = "quic")]
pub mod quic;

/// Read one length-prefixed frame from `r` and return the *full* bytes
/// (length prefix + body) so the caller can run their decoder over them.
///
/// Symmetric with the encoder in `wgpu-remote-protocol::codec`. Lives here so
/// both client and server can reuse it; the protocol crate stays I/O-free.
pub async fn read_frame<R: AsyncRead + Unpin>(r: &mut R) -> std::io::Result<Vec<u8>> {
    let mut len_buf = [0u8; 4];
    r.read_exact(&mut len_buf).await?;
    let len = u32::from_be_bytes(len_buf) as usize;
    let mut body = vec![0u8; len];
    r.read_exact(&mut body).await?;
    let mut full = Vec::with_capacity(4 + len);
    full.extend_from_slice(&len_buf);
    full.extend_from_slice(&body);
    Ok(full)
}

#[derive(Debug, Error)]
pub enum TransportError {
    #[error("connection closed")]
    Closed,
    #[error("address resolution failed: {0}")]
    Resolve(String),
    #[error("tls / handshake failed: {0}")]
    Handshake(String),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("transport-specific: {0}")]
    Other(String),
}

/// Establishes connections. Each impl picks its own address type — `SocketAddr`
/// for raw QUIC, `NodeAddr` for iroh, `Url` for WebTransport.
pub trait Transport: Send + Sync + 'static {
    type Address: Send + Sync + 'static;
    type Connection: Connection;

    fn dial(
        &self,
        addr: Self::Address,
    ) -> impl Future<Output = Result<Self::Connection, TransportError>> + Send;

    fn accept(&self) -> impl Future<Output = Result<Self::Connection, TransportError>> + Send;
}

/// A live connection. Multiplexes any number of bidirectional streams.
pub trait Connection: Send + Sync + 'static {
    type SendStream: AsyncWrite + Send + Unpin + 'static;
    type RecvStream: AsyncRead + Send + Unpin + 'static;

    fn open_bi(
        &self,
    ) -> impl Future<Output = Result<(Self::SendStream, Self::RecvStream), TransportError>> + Send;

    fn accept_bi(
        &self,
    ) -> impl Future<Output = Result<(Self::SendStream, Self::RecvStream), TransportError>> + Send;

    fn close(&self, code: u32, reason: &[u8]);
}

/// Optional capability: unreliable datagrams. WebTransport, QUIC (RFC 9221),
/// and iroh support this; libp2p's story is weaker. Transports that lack it
/// simply don't implement the trait.
pub trait Datagrams: Connection {
    fn send_datagram(
        &self,
        data: Bytes,
    ) -> impl Future<Output = Result<(), TransportError>> + Send;

    fn recv_datagram(&self) -> impl Future<Output = Result<Bytes, TransportError>> + Send;
}
