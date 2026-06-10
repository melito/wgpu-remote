//! iroh transport adapter for [`wgpu_remote_transport`].
//!
//! Uses iroh's NAT-traversing QUIC stack to implement the [`Transport`] and
//! [`Connection`] traits. Peers are addressed by [`EndpointAddr`] (an
//! `EndpointId` plus optional relay/direct addresses).

use bytes::Bytes;
use iroh::endpoint::presets::Minimal;
use iroh::endpoint::{RecvStream, SendStream, VarInt};
use iroh::{Endpoint, SecretKey};
use wgpu_remote_transport::{Connection, Datagrams, Transport, TransportError};

// Re-export iroh types that callers need for addressing.
pub use iroh::{EndpointAddr, EndpointId};

/// ALPN identifier for wgpu-remote over iroh.
pub const ALPN: &[u8] = b"wgpu-remote/1";

/// An iroh-backed transport endpoint.
///
/// Wraps [`iroh::Endpoint`] and implements [`Transport`] with
/// `Address = EndpointAddr`.
pub struct IrohEndpoint {
    endpoint: Endpoint,
}

impl IrohEndpoint {
    /// Create a new iroh endpoint with a random secret key.
    ///
    /// Uses the `Minimal` preset (ring TLS crypto, no relay/discovery).
    /// For endpoints that need to connect across networks, use
    /// [`IrohEndpoint::with_discovery`] instead.
    pub async fn new() -> Result<Self, IrohError> {
        Self::with_secret_key(SecretKey::generate()).await
    }

    /// Create a new iroh endpoint with the given secret key.
    ///
    /// Uses the `Minimal` preset (no relay/discovery).
    pub async fn with_secret_key(secret_key: SecretKey) -> Result<Self, IrohError> {
        let endpoint = Endpoint::builder(Minimal)
            .secret_key(secret_key)
            .alpns(vec![ALPN.to_vec()])
            .bind()
            .await?;
        Ok(Self { endpoint })
    }

    /// Create an endpoint with n0's public relay + DNS discovery.
    ///
    /// This is the recommended constructor for real-world use — endpoints
    /// can find each other by ID alone, with NAT traversal via relay.
    pub async fn with_discovery() -> Result<Self, IrohError> {
        use iroh::endpoint::presets::N0;
        let endpoint = Endpoint::builder(N0)
            .alpns(vec![ALPN.to_vec()])
            .bind()
            .await?;
        Ok(Self { endpoint })
    }

    /// Create an endpoint from a pre-configured [`iroh::Endpoint`].
    ///
    /// The caller is responsible for setting ALPN protocols.
    pub fn from_endpoint(endpoint: Endpoint) -> Self {
        Self { endpoint }
    }

    /// The endpoint's public identifier.
    pub fn id(&self) -> iroh::EndpointId {
        self.endpoint.id()
    }

    /// The endpoint's address (id + known addresses).
    pub fn endpoint_addr(&self) -> EndpointAddr {
        EndpointAddr {
            id: self.endpoint.id(),
            addrs: Default::default(),
        }
    }

    /// Access the underlying iroh endpoint.
    pub fn endpoint(&self) -> &Endpoint {
        &self.endpoint
    }
}

impl Transport for IrohEndpoint {
    type Address = EndpointAddr;
    type Connection = IrohConnection;

    async fn dial(&self, addr: Self::Address) -> Result<Self::Connection, TransportError> {
        let conn = self
            .endpoint
            .connect(addr, ALPN)
            .await
            .map_err(|e| TransportError::Other(e.to_string()))?;
        Ok(IrohConnection { inner: conn })
    }

    async fn accept(&self) -> Result<Self::Connection, TransportError> {
        let incoming = self
            .endpoint
            .accept()
            .await
            .ok_or(TransportError::Closed)?;
        let conn = incoming
            .accept()
            .map_err(|e| TransportError::Other(e.to_string()))?
            .await
            .map_err(|e| TransportError::Other(e.to_string()))?;
        Ok(IrohConnection { inner: conn })
    }
}

/// A live iroh connection implementing [`Connection`].
#[derive(Clone)]
pub struct IrohConnection {
    inner: iroh::endpoint::Connection,
}

impl IrohConnection {
    /// Wrap an existing iroh connection.
    pub fn new(inner: iroh::endpoint::Connection) -> Self {
        Self { inner }
    }

    /// The remote peer's endpoint ID.
    pub fn remote_id(&self) -> iroh::EndpointId {
        self.inner.remote_id()
    }

    /// Wait until the connection is closed by the peer or an error occurs.
    pub async fn closed(&self) -> iroh::endpoint::ConnectionError {
        self.inner.closed().await
    }
}

impl Connection for IrohConnection {
    type SendStream = SendStream;
    type RecvStream = RecvStream;

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
        self.inner.close(VarInt::from_u32(code), reason);
    }
}

impl Datagrams for IrohConnection {
    async fn send_datagram(&self, data: Bytes) -> Result<(), TransportError> {
        self.inner
            .send_datagram(data)
            .map_err(|e| TransportError::Other(e.to_string()))
    }

    async fn recv_datagram(&self) -> Result<Bytes, TransportError> {
        self.inner
            .read_datagram()
            .await
            .map_err(|e| TransportError::Other(e.to_string()))
    }
}

#[derive(Debug, thiserror::Error)]
pub enum IrohError {
    #[error("bind failed: {0}")]
    Bind(String),
}

impl From<iroh::endpoint::BindError> for IrohError {
    fn from(e: iroh::endpoint::BindError) -> Self {
        IrohError::Bind(e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::Ipv4Addr;
    use iroh::{RelayMap, RelayMode};
    use iroh_relay::server::{
        CertConfig, QuicConfig, RelayConfig as RelayServerConfig, Server, ServerConfig, TlsConfig,
    };
    use tokio::io::AsyncWriteExt;
    use wgpu_remote_transport::Connection;

    /// Spin up a local relay server so endpoints can discover each other.
    async fn local_relay() -> (RelayMap, iroh::RelayUrl, Server) {
        let (_certs, server_config) =
            iroh_relay::server::testing::self_signed_tls_certs_and_config();
        let tls = TlsConfig::new(
            (Ipv4Addr::LOCALHOST, 0),
            CertConfig::Manual { server_config },
        );
        let mut relay = RelayServerConfig::new((Ipv4Addr::LOCALHOST, 0));
        relay.tls = Some(tls);
        let mut config = ServerConfig::default();
        config.relay = Some(relay);
        config.quic = Some(QuicConfig::new((Ipv4Addr::LOCALHOST, 0)));

        let server = Server::spawn(config).await.unwrap();
        let url: iroh::RelayUrl = format!("https://{}", server.https_addr().unwrap())
            .parse()
            .unwrap();
        (RelayMap::from(url.clone()), url, server)
    }

    /// Create a pair of local iroh endpoints connected via a local relay.
    async fn test_pair() -> (IrohEndpoint, IrohEndpoint, EndpointAddr, Server) {
        use iroh::tls::CaRootsConfig;

        let (relay_map, relay_url, server) = local_relay().await;

        let raw1 = Endpoint::builder(Minimal)
            .relay_mode(RelayMode::Custom(relay_map.clone()))
            .ca_roots_config(CaRootsConfig::insecure_skip_verify())
            .alpns(vec![ALPN.to_vec()])
            .bind()
            .await
            .unwrap();
        let raw2 = Endpoint::builder(Minimal)
            .relay_mode(RelayMode::Custom(relay_map.clone()))
            .ca_roots_config(CaRootsConfig::insecure_skip_verify())
            .alpns(vec![ALPN.to_vec()])
            .bind()
            .await
            .unwrap();

        raw1.online().await;
        raw2.online().await;

        let ep2_addr = EndpointAddr::new(raw2.id()).with_relay_url(relay_url);
        (
            IrohEndpoint::from_endpoint(raw1),
            IrohEndpoint::from_endpoint(raw2),
            ep2_addr,
            server,
        )
    }

    /// Two local iroh endpoints exchange data over a bidi stream,
    /// exercising both the `Transport::dial`/`accept` and `Connection`
    /// trait methods.
    #[tokio::test]
    async fn bidi_stream_round_trip() {
        let (ep1, ep2, ep2_addr, _relay) = test_pair().await;

        let server = tokio::spawn(async move {
            let conn = ep2.accept().await.unwrap();
            let (mut tx, mut rx) = conn.accept_bi().await.unwrap();
            let mut buf = [0u8; 5];
            rx.read_exact(&mut buf).await.unwrap();
            assert_eq!(&buf, b"hello");
            tx.write_all(b"world").await.unwrap();
            tx.shutdown().await.unwrap();
            // Keep the connection alive until the client closes it,
            // otherwise data may not be fully flushed over the relay.
            conn.closed().await;
        });

        let conn = ep1.dial(ep2_addr).await.unwrap();
        let (mut tx, mut rx) = conn.open_bi().await.unwrap();
        tx.write_all(b"hello").await.unwrap();
        tx.shutdown().await.unwrap();

        let response = rx.read_to_end(1024).await.unwrap();
        assert_eq!(response, b"world");
        conn.close(0, b"done");

        server.await.unwrap();
    }

    /// Datagrams round-trip between two local iroh endpoints.
    #[tokio::test]
    async fn datagram_round_trip() {
        let (ep1, ep2, ep2_addr, _relay) = test_pair().await;

        let server = tokio::spawn(async move {
            let conn = ep2.accept().await.unwrap();
            let data = conn.recv_datagram().await.unwrap();
            assert_eq!(&data[..], b"ping");
            conn.send_datagram(Bytes::from_static(b"pong"))
                .await
                .unwrap();
            // Keep connection alive until client closes.
            conn.closed().await;
        });

        let client_conn = ep1.dial(ep2_addr).await.unwrap();
        client_conn
            .send_datagram(Bytes::from_static(b"ping"))
            .await
            .unwrap();

        let reply = client_conn.recv_datagram().await.unwrap();
        assert_eq!(&reply[..], b"pong");

        client_conn.close(0, b"done");
        server.await.unwrap();
    }
}
