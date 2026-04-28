//! Drive the server [`Engine`] directly: create a buffer, write bytes, read
//! them back, assert they match. No transport involved — this exercises the
//! GPU dispatch path. Requires a working wgpu backend (Metal on macOS).

use bytes::Bytes;
use wgpu_remote_protocol::{
    Response,
    actions::{Action, Frame, RequestId},
    descriptors::BufferDescriptor,
    ids::BufferId,
    responses::ErrorCode,
};
use wgpu_remote_server::Engine;
use wgpu_types::BufferUsages;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn buffer_write_then_read_round_trip() -> anyhow::Result<()> {
    let engine = Engine::new().await?;

    let buffer_id = BufferId::new(1);
    let payload: Vec<u8> = (0..256u32).map(|i| (i & 0xFF) as u8).collect();

    // 1. CreateBuffer with COPY_DST | MAP_READ — minimum to write+map.
    let resp = engine
        .dispatch(Frame {
            request_id: Some(RequestId(1)),
            action: Action::CreateBuffer {
                id: buffer_id,
                desc: BufferDescriptor {
                    label: Some("test-buffer".into()),
                    size: payload.len() as u64,
                    usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                },
            },
        })
        .await
        .expect("CreateBuffer should produce a response");
    assert!(matches!(resp.response, Response::Ok), "got {:?}", resp.response);

    // 2. WriteBuffer.
    let resp = engine
        .dispatch(Frame {
            request_id: Some(RequestId(2)),
            action: Action::WriteBuffer {
                buffer: buffer_id,
                offset: 0,
                data: Bytes::from(payload.clone()),
            },
        })
        .await
        .expect("WriteBuffer should produce a response");
    assert!(matches!(resp.response, Response::Ok), "got {:?}", resp.response);

    // 3. MapBufferForRead — the latency-sensitive path.
    let resp = engine
        .dispatch(Frame {
            request_id: Some(RequestId(3)),
            action: Action::MapBufferForRead {
                buffer: buffer_id,
                offset: 0,
                size: payload.len() as u64,
            },
        })
        .await
        .expect("MapBufferForRead should produce a response");
    let bytes = match resp.response {
        Response::BufferData { data } => data,
        other => panic!("expected BufferData, got {other:?}"),
    };
    assert_eq!(bytes.as_ref(), payload.as_slice());

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unknown_buffer_id_errors() -> anyhow::Result<()> {
    let engine = Engine::new().await?;

    let resp = engine
        .dispatch(Frame {
            request_id: Some(RequestId(1)),
            action: Action::MapBufferForRead {
                buffer: BufferId::new(999),
                offset: 0,
                size: 4,
            },
        })
        .await
        .unwrap();
    match resp.response {
        Response::Error { code, .. } => assert_eq!(code, ErrorCode::UnknownResource),
        other => panic!("expected Error, got {other:?}"),
    }
    Ok(())
}
