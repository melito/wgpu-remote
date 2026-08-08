//! In-memory transport. Two `Connection`s connected by `tokio::io::duplex`
//! pairs — same process, no sockets, ideal for unit tests.
//!
//! Each `open_bi` allocates a fresh duplex pair; the local half is returned to
//! the caller, the remote half is shipped to the peer's `accept_bi` queue via
//! an `mpsc` channel. Closure propagates by dropping the senders.

use std::io;
use std::sync::Arc;

use tokio::io::{DuplexStream, ReadHalf, WriteHalf, duplex, split};
use tokio::sync::{Mutex, mpsc};
use wgpu_remote::transport::{Connection, TransportError};

const STREAM_BUF_BYTES: usize = 64 * 1024;
const QUEUE_DEPTH: usize = 64;

type StreamPair = (WriteHalf<DuplexStream>, ReadHalf<DuplexStream>);

pub struct InMemoryTransport {
    /// `dial` takes from this — the matching half is `accept`ed by the peer.
    /// In-process tests usually skip `dial`/`accept` and call [`pair`].
    _placeholder: (),
}

/// Build a connected client/server pair directly. Bypasses dial/accept since
/// in-memory tests rarely need to exercise those.
pub fn pair() -> (InMemoryConnection, InMemoryConnection) {
    // a → b channel: a's outgoing feeds b's incoming.
    let (a_to_b_tx, a_to_b_rx) = mpsc::channel(QUEUE_DEPTH);
    // b → a channel: b's outgoing feeds a's incoming.
    let (b_to_a_tx, b_to_a_rx) = mpsc::channel(QUEUE_DEPTH);
    let a = InMemoryConnection {
        outgoing: a_to_b_tx,
        incoming: Arc::new(Mutex::new(b_to_a_rx)),
    };
    let b = InMemoryConnection {
        outgoing: b_to_a_tx,
        incoming: Arc::new(Mutex::new(a_to_b_rx)),
    };
    (a, b)
}

#[derive(Clone)]
pub struct InMemoryConnection {
    outgoing: mpsc::Sender<StreamPair>,
    incoming: Arc<Mutex<mpsc::Receiver<StreamPair>>>,
}

impl Connection for InMemoryConnection {
    type SendStream = WriteHalf<DuplexStream>;
    type RecvStream = ReadHalf<DuplexStream>;

    async fn open_bi(&self) -> Result<(Self::SendStream, Self::RecvStream), TransportError> {
        let (local, remote) = duplex(STREAM_BUF_BYTES);
        let (local_r, local_w) = split(local);
        let (remote_r, remote_w) = split(remote);
        // Ship the *remote* half (peer reads what we write, peer writes what we read)
        // to the peer's accept_bi queue.
        self.outgoing
            .send((remote_w, remote_r))
            .await
            .map_err(|_| TransportError::Closed)?;
        Ok((local_w, local_r))
    }

    async fn accept_bi(&self) -> Result<(Self::SendStream, Self::RecvStream), TransportError> {
        let mut guard = self.incoming.lock().await;
        guard.recv().await.ok_or(TransportError::Closed)
    }

    fn close(&self, _code: u32, _reason: &[u8]) {
        // Dropping `outgoing` would close the half-connection; do that on Drop.
    }
}

// We don't currently expose dial/accept on InMemoryTransport — `pair()` is
// simpler for tests. Implement Transport later if a test needs the full path.
#[allow(dead_code)]
impl InMemoryTransport {
    pub fn new() -> Self {
        Self { _placeholder: () }
    }
}

impl Default for InMemoryTransport {
    fn default() -> Self {
        Self::new()
    }
}

// Sanity: WriteHalf + ReadHalf of a DuplexStream are AsyncWrite/AsyncRead.
// This `as` cast verifies at compile time without runtime cost.
#[allow(dead_code)]
fn _assert_traits() {
    fn want_async_write<T: tokio::io::AsyncWrite>() {}
    fn want_async_read<T: tokio::io::AsyncRead>() {}
    want_async_write::<WriteHalf<DuplexStream>>();
    want_async_read::<ReadHalf<DuplexStream>>();
    let _ = io::Error::other("unused");
}
