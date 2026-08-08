//! End-to-end: protocol frames over the in-memory transport.

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use wgpu_remote::protocol::{
    Action, PROTOCOL_VERSION, Response,
    actions::{Frame, RequestId},
    codec::{decode_frame, encode_frame},
    responses::ResponseFrame,
};
use wgpu_remote_tests::pair;
use wgpu_remote::transport::Connection;

/// Reads bytes for one length-prefixed frame. Returns `(len_bytes, body_bytes)`
/// concatenated so [`decode_frame`] can parse the whole thing.
async fn read_one_frame<R: tokio::io::AsyncRead + Unpin>(mut r: R) -> anyhow::Result<Vec<u8>> {
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

#[tokio::test]
async fn hello_handshake_round_trip() -> anyhow::Result<()> {
    let (client_conn, server_conn) = pair();

    // Server task: accept one stream, read a Hello, reply with HelloAck.
    let server = tokio::spawn(async move {
        let (mut tx, rx) = server_conn.accept_bi().await.unwrap();
        let raw = read_one_frame(rx).await.unwrap();
        let (frame, _) = decode_frame::<Frame>(&raw).unwrap();
        let request_id = frame.request_id.expect("client must include a request id");
        let protocol_version = match frame.action {
            Action::Hello { protocol_version } => protocol_version,
            other => panic!("expected Hello, got {other:?}"),
        };
        assert_eq!(protocol_version, PROTOCOL_VERSION);

        let response = ResponseFrame {
            request_id,
            response: Response::HelloAck {
                protocol_version: PROTOCOL_VERSION,
            },
        };
        let bytes = encode_frame(&response).unwrap();
        tx.write_all(&bytes).await.unwrap();
        tx.flush().await.unwrap();
    });

    // Client: open a stream, send Hello, await HelloAck.
    let (mut tx, rx) = client_conn.open_bi().await?;
    let request_id = RequestId(1);
    let bytes = encode_frame(&Frame {
        request_id: Some(request_id),
        action: Action::Hello {
            protocol_version: PROTOCOL_VERSION,
        },
    })?;
    tx.write_all(&bytes).await?;
    tx.flush().await?;

    let raw = read_one_frame(rx).await?;
    let (response, _) = decode_frame::<ResponseFrame>(&raw)?;
    assert_eq!(response.request_id, request_id);
    match response.response {
        Response::HelloAck { protocol_version } => {
            assert_eq!(protocol_version, PROTOCOL_VERSION);
        }
        other => panic!("expected HelloAck, got {other:?}"),
    }

    server.await?;
    Ok(())
}
