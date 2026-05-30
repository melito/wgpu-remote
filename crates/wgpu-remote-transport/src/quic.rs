//! quinn-based QUIC transport.
//!
//! Two entry points:
//! - [`QuicEndpoint::server`] binds a UDP socket with a self-signed cert and
//!   listens for incoming connections. The cert chain is returned alongside
//!   so the test client can pin it.
//! - [`QuicEndpoint::client`] makes a client endpoint that trusts the supplied
//!   cert chain.
//!
//! Production deployments will replace the self-signed flow with real cert
//! provisioning, but the surface here doesn't change.

use std::net::SocketAddr;
use std::sync::Arc;

use quinn::crypto::rustls::{QuicClientConfig, QuicServerConfig};
use quinn::{ClientConfig, ConnectError, ConnectionError, Endpoint, ServerConfig};
use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer};

use crate::Connection;
use crate::TransportError;
use crate::pki::ServerCert;

/// ALPN identifier for the wgpu-remote protocol.
pub const ALPN: &[u8] = b"wgpu-remote/1";

#[derive(Debug, thiserror::Error)]
pub enum QuicError {
    #[error("rustls config: {0}")]
    Rustls(#[from] rustls::Error),
    #[error("rcgen: {0}")]
    Rcgen(#[from] rcgen::Error),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("connect: {0}")]
    Connect(#[from] ConnectError),
    #[error("connection: {0}")]
    Connection(#[from] ConnectionError),
    #[error("no incoming connection")]
    NoIncoming,
}

/// Wraps a `quinn::Endpoint` plus a record of what role it was opened in.
pub struct QuicEndpoint {
    endpoint: Endpoint,
}

impl QuicEndpoint {
    /// Bind a server endpoint to `addr` (use `0.0.0.0:0` to let the OS pick).
    /// Returns the endpoint and the DER-encoded server cert so a test client
    /// can pin it directly.
    pub fn server(addr: SocketAddr) -> Result<(Self, CertificateDer<'static>), QuicError> {
        let cert = rcgen::generate_simple_self_signed(vec!["localhost".to_string()])?;
        let key = PrivatePkcs8KeyDer::from(cert.key_pair.serialize_der());
        let cert_der = cert.cert.der().clone();

        let mut server_crypto = rustls::ServerConfig::builder()
            .with_no_client_auth()
            .with_single_cert(vec![cert_der.clone()], PrivateKeyDer::Pkcs8(key))?;
        server_crypto.alpn_protocols = vec![ALPN.to_vec()];

        let quic_server_crypto = QuicServerConfig::try_from(server_crypto)
            .map_err(|e| QuicError::Rustls(rustls::Error::General(format!("{e:?}"))))?;
        let server_config = ServerConfig::with_crypto(Arc::new(quic_server_crypto));
        let endpoint = Endpoint::server(server_config, addr)?;
        Ok((Self { endpoint }, cert_der))
    }

    /// Bind a server endpoint using a CA-issued [`ServerCert`].
    ///
    /// The cert chain and private key come from `CertAuthority::issue_server_cert`.
    pub fn server_with_cert(
        addr: SocketAddr,
        server_cert: ServerCert,
    ) -> Result<Self, QuicError> {
        let mut server_crypto = rustls::ServerConfig::builder()
            .with_no_client_auth()
            .with_single_cert(server_cert.cert_chain, server_cert.private_key)?;
        server_crypto.alpn_protocols = vec![ALPN.to_vec()];

        let quic_server_crypto = QuicServerConfig::try_from(server_crypto)
            .map_err(|e| QuicError::Rustls(rustls::Error::General(format!("{e:?}"))))?;
        let server_config = ServerConfig::with_crypto(Arc::new(quic_server_crypto));
        let endpoint = Endpoint::server(server_config, addr)?;
        Ok(Self { endpoint })
    }

    /// Build a client endpoint that pins exactly the supplied server cert.
    pub fn client(server_cert: CertificateDer<'static>) -> Result<Self, QuicError> {
        let mut roots = rustls::RootCertStore::empty();
        roots.add(server_cert)?;

        let mut client_crypto = rustls::ClientConfig::builder()
            .with_root_certificates(roots)
            .with_no_client_auth();
        client_crypto.alpn_protocols = vec![ALPN.to_vec()];

        let quic_client_crypto = QuicClientConfig::try_from(client_crypto)
            .map_err(|e| QuicError::Rustls(rustls::Error::General(format!("{e:?}"))))?;
        let client_config = ClientConfig::new(Arc::new(quic_client_crypto));

        let mut endpoint = Endpoint::client("0.0.0.0:0".parse().unwrap())?;
        endpoint.set_default_client_config(client_config);
        Ok(Self { endpoint })
    }

    pub fn local_addr(&self) -> std::io::Result<SocketAddr> {
        self.endpoint.local_addr()
    }

    /// Dial a server. `server_name` must match a SAN on the server cert
    /// (we used `"localhost"`).
    pub async fn connect(
        &self,
        addr: SocketAddr,
        server_name: &str,
    ) -> Result<QuicConnection, QuicError> {
        let connecting = self.endpoint.connect(addr, server_name)?;
        let conn = connecting.await?;
        Ok(QuicConnection::new(conn))
    }

    /// Accept the next incoming connection.
    pub async fn accept(&self) -> Result<QuicConnection, QuicError> {
        let incoming = self.endpoint.accept().await.ok_or(QuicError::NoIncoming)?;
        let conn = incoming.await?;
        Ok(QuicConnection::new(conn))
    }

    /// Idle timeout shutdown.
    pub fn close(&self) {
        self.endpoint.close(0u32.into(), b"shutdown");
    }
}

#[derive(Clone)]
pub struct QuicConnection {
    inner: quinn::Connection,
}

impl QuicConnection {
    fn new(inner: quinn::Connection) -> Self {
        Self { inner }
    }

    pub fn remote_address(&self) -> SocketAddr {
        self.inner.remote_address()
    }
}

impl Connection for QuicConnection {
    type SendStream = quinn::SendStream;
    type RecvStream = quinn::RecvStream;

    async fn open_bi(&self) -> Result<(Self::SendStream, Self::RecvStream), TransportError> {
        self.inner
            .open_bi()
            .await
            .map_err(|e| TransportError::Other(e.to_string()))
    }

    async fn accept_bi(&self) -> Result<(Self::SendStream, Self::RecvStream), TransportError> {
        self.inner
            .accept_bi()
            .await
            .map_err(|e| TransportError::Other(e.to_string()))
    }

    fn close(&self, code: u32, reason: &[u8]) {
        self.inner.close(code.into(), reason);
    }
}
