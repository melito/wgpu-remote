//! Stress-test the multiplexed client: fire N concurrent `MapBufferForRead`
//! requests that each expect a *distinct* byte pattern back, and assert the
//! responses are correctly correlated.
//!
//! If the per-RequestId demuxer is buggy (e.g. returns the wrong oneshot's
//! response, or mixes up request_ids on retries), this test will fail with a
//! byte mismatch rather than a timeout.

use bytes::Bytes;
use std::sync::Arc;

use wgpu_remote_client::{BufferUsages, Client, Instance, descriptors::BufferDescriptor};
use wgpu_remote_server::{Engine, run_connection};
use wgpu_remote_tests::pair;

const N: usize = 32;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_map_for_read() -> anyhow::Result<()> {
    let (client_conn, server_conn) = pair();
    let engine = Arc::new(Engine::new().await?);
    let server_handle = tokio::spawn(async move {
        run_connection(engine, server_conn).await.unwrap();
    });

    let instance = Instance::new(Client::new(client_conn));
    let adapter = instance.request_adapter().await?;
    let (device, queue) = adapter.request_device().await?;

    // Create N buffers, each with size 4*i bytes (i in 1..=N) and write a
    // pattern unique to that buffer: bytes are all `i as u8`.
    let mut buffers = Vec::with_capacity(N);
    for i in 1..=N {
        let size = (i * 4) as u64;
        let buffer = device
            .create_buffer(&BufferDescriptor {
                label: Some(format!("b{i}")),
                size,
                usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
        let payload: Vec<u8> = vec![i as u8; size as usize];
        queue.write_buffer(&buffer, 0, Bytes::from(payload.clone()));
        buffers.push((buffer, payload));
    }

    // Fire N reads in parallel from the same Client. The multiplexed reader
    // task has to deliver the right bytes to the right awaiter.
    let read_futures: Vec<_> = buffers
        .iter()
        .map(|(buf, _expected)| {
            let buf = buf.clone();
            async move { buf.read_all().await }
        })
        .collect();
    let results = futures_util::future::try_join_all(read_futures).await?;

    // Assert each response carries the bytes for *its* buffer, not someone
    // else's.
    for (i, (got, (_buf, expected))) in results.iter().zip(buffers.iter()).enumerate() {
        assert_eq!(
            got.as_ref(),
            expected.as_slice(),
            "buffer index {i}: bytes don't match — demuxer correlated the wrong RequestId"
        );
    }

    drop(buffers);
    drop(queue);
    drop(device);
    drop(adapter);
    drop(instance);
    let _ = tokio::time::timeout(std::time::Duration::from_secs(2), server_handle).await;
    Ok(())
}
