//! Per-connection handler.
//!
//! Wire pattern (v1.1): one bidirectional stream per connection. The client
//! writes framed actions; the server reads them in order, dispatches each
//! sequentially against the [`Engine`], and writes the framed response back
//! over the same send stream.
//!
//! "Sequential dispatch" preserves action ordering on the wire. If a future
//! workload needs concurrent dispatch (e.g. parallelizing a slow
//! `MapBufferForRead` against subsequent buffer writes), that becomes an
//! opt-in decision on a per-action basis — but it does *not* hold for v1.1.
//!
//! Generic over [`Connection`] so the same handler runs over QUIC and the
//! in-memory transport.

use std::sync::Arc;

use tokio::io::{AsyncRead, AsyncWrite, AsyncWriteExt};
use wgpu_remote_protocol::{
    actions::Frame,
    codec::{decode_frame, encode_frame},
};
use wgpu_remote_transport::{Connection, read_frame};

use crate::Engine;

/// Run the action loop on a single bidi stream: read framed actions in a
/// loop, dispatch sequentially, write framed responses back to the same
/// send stream. Returns when the client closes the stream (clean EOF) or a
/// codec/transport error occurs.
pub async fn handle_stream<S, R>(engine: &Engine, mut tx: S, mut rx: R) -> anyhow::Result<()>
where
    S: AsyncWrite + Unpin,
    R: AsyncRead + Unpin,
{
    loop {
        let raw = match read_frame(&mut rx).await {
            Ok(b) => b,
            // Clean EOF or transport error → end the loop. The connection
            // loop will decide whether to accept another stream.
            Err(_) => return Ok(()),
        };
        let (frame, _) = decode_frame::<Frame>(&raw)?;
        if let Some(response) = engine.dispatch(frame).await {
            let bytes = encode_frame(&response)?;
            tx.write_all(&bytes).await?;
            tx.flush().await?;
        }
    }
}

/// Accept loop on a connection. v1.1 expects one bidi stream per connection
/// (the client opens it via the multiplexed Client). We still loop in case a
/// connection re-opens its stream after a crash, or a future workload uses
/// auxiliary streams.
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
