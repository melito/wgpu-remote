//! Low-level client. The wgpu-style facade lives in sibling modules and
//! delegates to this layer.
//!
//! v1.2: requests share a single bidirectional stream per [`Client`], with
//! both writer and reader running on dedicated background tasks.
//!
//! - **Writer task**: drains an mpsc inbox of pre-encoded frames and writes
//!   them to the send stream in mpsc-send order. Both `send` and `request`
//!   push to it. This is the layer that guarantees actions arrive at the
//!   server in the order they were issued — critical for fire-and-forget
//!   resource creates to be safe.
//! - **Reader task**: demuxes incoming response frames by [`RequestId`] via
//!   `oneshot` channels. Failures (transport errors, decode errors) cause
//!   all pending requests to fail with `ConnectionClosed`.

use std::collections::HashMap;
use std::sync::Mutex as StdMutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use thiserror::Error;
use tokio::io::AsyncWriteExt;
use tokio::sync::{OnceCell, mpsc, oneshot};
use tokio::task::JoinHandle;
use crate::protocol::{
    Action, Response,
    actions::{Frame, RequestId},
    codec::{decode_frame, encode_frame},
    responses::ResponseFrame,
};
use crate::transport::{Connection, read_frame};

#[derive(Debug, Error)]
pub enum ClientError {
    #[error("transport: {0}")]
    Transport(#[from] crate::transport::TransportError),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("encode: {0}")]
    Encode(#[from] bincode::error::EncodeError),
    #[error("decode: {0}")]
    Decode(#[from] bincode::error::DecodeError),
    #[error("codec: {0}")]
    Codec(#[from] crate::protocol::codec::CodecError),
    #[error("server returned an error response: {0:?}")]
    ServerError(crate::protocol::responses::ErrorCode, String),
    #[error("connection closed before response arrived")]
    ConnectionClosed,
}

pub struct Client<C: Connection + Clone> {
    connection: C,
    next_request_id: AtomicU64,
    /// Lazy-initialized on the first `request`/`send` call so `Client::new`
    /// can stay synchronous (constructing a Client shouldn't open streams).
    mux: OnceCell<Arc<Mux>>,
    _phantom: std::marker::PhantomData<C>,
}

/// One write task carries a pre-encoded frame; the writer task pulls them
/// in mpsc order and writes them to the send stream.
struct WriteJob {
    bytes: Vec<u8>,
}

struct Mux {
    /// Push-side of the writer task's inbox. Cloning is fine — concurrent
    /// senders just interleave by mpsc fairness.
    inbox: mpsc::UnboundedSender<WriteJob>,
    /// Shared with the reader task so it can deliver responses by RequestId.
    pending: Arc<StdMutex<HashMap<RequestId, oneshot::Sender<Response>>>>,
    /// Reader + writer; both aborted on Mux drop.
    reader: StdMutex<Option<JoinHandle<()>>>,
    writer: StdMutex<Option<JoinHandle<()>>>,
}

impl Mux {
    async fn open<C: Connection + Clone + 'static>(
        connection: &C,
    ) -> Result<Arc<Self>, ClientError> {
        let (mut tx_stream, mut rx_stream) = connection.open_bi().await?;
        let pending: Arc<StdMutex<HashMap<RequestId, oneshot::Sender<Response>>>> =
            Arc::new(StdMutex::new(HashMap::new()));
        let pending_for_reader = Arc::clone(&pending);

        // Writer task: drain the inbox and write to the stream in mpsc order.
        let (inbox_tx, mut inbox_rx) = mpsc::unbounded_channel::<WriteJob>();
        let writer = tokio::spawn(async move {
            while let Some(job) = inbox_rx.recv().await {
                if tx_stream.write_all(&job.bytes).await.is_err() {
                    break;
                }
                if tx_stream.flush().await.is_err() {
                    break;
                }
            }
        });

        // Reader task: decode incoming response frames, dispatch to oneshots.
        let reader = tokio::spawn(async move {
            loop {
                let raw = match read_frame(&mut rx_stream).await {
                    Ok(b) => b,
                    Err(_) => break,
                };
                let frame: ResponseFrame = match decode_frame(&raw) {
                    Ok((f, _)) => f,
                    Err(_) => break,
                };
                let sender = pending_for_reader
                    .lock()
                    .unwrap()
                    .remove(&frame.request_id);
                if let Some(s) = sender {
                    let _ = s.send(frame.response);
                }
                // Unknown id: server bug or response for a `send` whose
                // receiver was dropped. Discard.
            }
            // Reader exited. Fail all pending by clearing the map (dropping
            // each oneshot::Sender triggers RecvError on the awaiter).
            pending_for_reader.lock().unwrap().clear();
        });

        Ok(Arc::new(Mux {
            inbox: inbox_tx,
            pending,
            reader: StdMutex::new(Some(reader)),
            writer: StdMutex::new(Some(writer)),
        }))
    }
}

impl Drop for Mux {
    fn drop(&mut self) {
        if let Some(h) = self.reader.lock().unwrap().take() {
            h.abort();
        }
        if let Some(h) = self.writer.lock().unwrap().take() {
            h.abort();
        }
    }
}

impl<C: Connection + Clone + 'static> Client<C> {
    pub fn new(connection: C) -> Self {
        Self {
            connection,
            next_request_id: AtomicU64::new(1),
            mux: OnceCell::new(),
            _phantom: std::marker::PhantomData,
        }
    }

    fn mint_request_id(&self) -> RequestId {
        RequestId(self.next_request_id.fetch_add(1, Ordering::Relaxed))
    }

    async fn mux(&self) -> Result<&Arc<Mux>, ClientError> {
        self.mux
            .get_or_try_init(|| async { Mux::open::<C>(&self.connection).await })
            .await
    }

    /// Issue one action, await the response.
    pub async fn request(&self, action: Action) -> Result<Response, ClientError> {
        let request_id = self.mint_request_id();
        let frame = Frame {
            request_id: Some(request_id),
            action,
        };
        let bytes = encode_frame(&frame)?;
        let mux = self.mux().await?;

        let (resp_tx, resp_rx) = oneshot::channel();
        mux.pending.lock().unwrap().insert(request_id, resp_tx);

        if mux.inbox.send(WriteJob { bytes }).is_err() {
            mux.pending.lock().unwrap().remove(&request_id);
            return Err(ClientError::ConnectionClosed);
        }

        match resp_rx.await {
            Ok(response) => Ok(response),
            Err(_) => Err(ClientError::ConnectionClosed),
        }
    }

    /// Fire-and-forget: enqueue an [`Action`] for transmission. Returns
    /// immediately. The action is sent in mpsc-issuance order, so any
    /// follow-up `request`/`send` that references resources implicitly
    /// created here will see them.
    ///
    /// Errors (encoding, transport, server-side) are silently dropped.
    /// Resource-creation failures surface as `UnknownResource` on the next
    /// reference. Use [`request`](Self::request) if you need explicit error
    /// handling.
    pub fn send(&self, action: Action) {
        let request_id = self.mint_request_id();
        let frame = Frame {
            request_id: Some(request_id),
            action,
        };
        let Ok(bytes) = encode_frame(&frame) else { return };

        // Mux init is async, but the only thing we need from it after init is
        // the inbox sender. If init hasn't happened yet, fast-path it: spawn
        // a tiny task to do the init and queue the job.
        if let Some(mux) = self.mux.get() {
            // Hot path: mux already initialized.
            let _ = mux.inbox.send(WriteJob { bytes });
        } else {
            // Cold path: connection not yet opened. Initialize it on a task,
            // then queue the bytes. We can't borrow self past this scope, so
            // we have to clone the connection (cheap — Arc inside).
            let connection = self.connection.clone();
            let mux_cell = self.mux.clone();
            tokio::spawn(async move {
                let mux_init = mux_cell
                    .get_or_try_init(|| async { Mux::open::<C>(&connection).await })
                    .await;
                if let Ok(mux) = mux_init {
                    let _ = mux.inbox.send(WriteJob { bytes });
                }
            });
        }
    }

    pub fn connection(&self) -> &C {
        &self.connection
    }
}
