//! Smallest possible end-to-end check: `install()` produces a `wgpu::Instance`
//! whose `request_adapter` and `request_device` round-trip successfully
//! through the in-memory transport.
//!
//! No resource creates or shader work yet — those land alongside their
//! interface impls. The point of this test is to fail loudly if the dispatch
//! plumbing (custom-backend construction, future return shapes, error
//! conversions) regresses.

use wgpu_remote_server::{Engine, run_connection};
use wgpu_remote_tests::pair;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn install_request_adapter_and_device() -> anyhow::Result<()> {
    let (client_conn, server_conn) = pair();

    let engine = std::sync::Arc::new(Engine::new().await?);
    let server = tokio::spawn(async move {
        run_connection(engine, server_conn).await.unwrap();
    });

    let instance = wgpu_remote_wgpu::install(client_conn);

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await?;

    let info = adapter.get_info();
    assert_eq!(info.name, "wgpu-remote");

    let (_device, _queue) = adapter
        .request_device(&wgpu::DeviceDescriptor::default())
        .await?;

    drop(server);
    Ok(())
}
