//! Low-level client. The wgpu-style facade lives in sibling modules and
//! delegates to this layer.

use std::sync::atomic::{AtomicU64, Ordering};

use thiserror::Error;
use tokio::io::AsyncWriteExt;
use wgpu_remote_protocol::{
    Action, Response,
    actions::{Frame, RequestId},
    codec::{decode_frame, encode_frame},
    responses::ResponseFrame,
};
use wgpu_remote_transport::{Connection, read_frame};

#[derive(Debug, Error)]
pub enum ClientError {
    #[error("transport: {0}")]
    Transport(#[from] wgpu_remote_transport::TransportError),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("encode: {0}")]
    Encode(#[from] bincode::error::EncodeError),
    #[error("decode: {0}")]
    Decode(#[from] bincode::error::DecodeError),
    #[error("codec: {0}")]
    Codec(#[from] wgpu_remote_protocol::codec::CodecError),
    #[error("server returned an error response: {0:?}")]
    ServerError(wgpu_remote_protocol::responses::ErrorCode, String),
    #[error("response request_id mismatch: sent {sent}, got {got}")]
    RequestIdMismatch { sent: u64, got: u64 },
}

pub struct Client<C> {
    connection: C,
    next_request_id: AtomicU64,
}

impl<C: Connection + Clone> Client<C> {
    pub fn new(connection: C) -> Self {
        Self {
            connection,
            next_request_id: AtomicU64::new(1),
        }
    }

    fn mint_request_id(&self) -> RequestId {
        RequestId(self.next_request_id.fetch_add(1, Ordering::Relaxed))
    }

    /// Issue one action, await the response. Opens a fresh stream.
    pub async fn request(&self, action: Action) -> Result<Response, ClientError> {
        let request_id = self.mint_request_id();
        let frame = Frame {
            request_id: Some(request_id),
            action,
        };
        let bytes = encode_frame(&frame)?;

        let (mut tx, mut rx) = self.connection.open_bi().await?;
        tx.write_all(&bytes).await?;
        // Half-close the send side: tells the server "I'm done writing." On
        // QUIC this is `SendStream::finish()` via Drop; on tokio duplex it's
        // an EOF on read once we drop. We drop after flushing.
        tx.flush().await?;
        drop(tx);

        let raw = read_frame(&mut rx).await?;
        let (response_frame, _) = decode_frame::<ResponseFrame>(&raw)?;
        if response_frame.request_id != request_id {
            return Err(ClientError::RequestIdMismatch {
                sent: request_id.0,
                got: response_frame.request_id.0,
            });
        }
        Ok(response_frame.response)
    }

    pub fn connection(&self) -> &C {
        &self.connection
    }
}
