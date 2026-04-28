//! Per-connection / per-stream handlers. Generic over [`Connection`] so the
//! same code runs over QUIC and the in-memory transport.
//!
//! Wire pattern (v1): one stream per request. Client opens a bidi stream,
//! writes the [`Frame`] of an `Action`, half-closes the send side, reads the
//! [`ResponseFrame`], closes. Server mirrors this.
//!
//! "One stream per request" trades a bit of overhead for simplicity — no
//! request-id multiplexer, no out-of-order reply handling. QUIC streams are
//! cheap so this is fine for v1.

use std::sync::Arc;

use tokio::io::{AsyncRead, AsyncWrite, AsyncWriteExt};
use wgpu_remote_protocol::{
    actions::Frame,
    codec::{decode_frame, encode_frame},
};
use wgpu_remote_transport::{Connection, read_frame};

use crate::Engine;

/// Handle a single stream: read one frame, dispatch, write one response,
/// flush. Errors are logged via the returned `Result`; the caller (the
/// connection loop) decides whether to keep the connection alive.
pub async fn handle_stream<S, R>(engine: &Engine, mut tx: S, mut rx: R) -> anyhow::Result<()>
where
    S: AsyncWrite + Unpin,
    R: AsyncRead + Unpin,
{
    let raw = read_frame(&mut rx).await?;
    let (frame, _) = decode_frame::<Frame>(&raw)?;
    if let Some(response) = engine.dispatch(frame).await {
        let bytes = encode_frame(&response)?;
        tx.write_all(&bytes).await?;
        tx.flush().await?;
    }
    // Drop tx → finishes the send side of the QUIC stream cleanly.
    Ok(())
}

/// Run the accept loop on a connection: spawn a stream handler per `accept_bi`.
/// Returns when the connection ends.
pub async fn run_connection<C>(engine: Arc<Engine>, connection: C) -> anyhow::Result<()>
where
    C: Connection + Clone,
    C::SendStream: 'static,
    C::RecvStream: 'static,
{
    loop {
        match connection.accept_bi().await {
            Ok((tx, rx)) => {
                let engine = engine.clone();
                tokio::spawn(async move {
                    if let Err(e) = handle_stream(&engine, tx, rx).await {
                        eprintln!("stream handler error: {e}");
                    }
                });
            }
            Err(_) => return Ok(()),
        }
    }
}
