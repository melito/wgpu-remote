//! End-to-end over QUIC: server + engine on one task, client on another,
//! same compute-double workload as `engine_compute.rs` but every action
//! crosses a real UDP socket and TLS handshake.
//!
//! This is the moment the architecture either stands up or falls over:
//! the protocol, transport, server engine, and client all have to compose.

use std::num::NonZeroU64;
use std::sync::Arc;

use bytes::Bytes;
use wgpu_remote_client::Client;
use wgpu_remote_protocol::{
    Action, Response,
    commands::{CommandBufferRecording, ComputeCommand, EncoderCommand},
    descriptors::{
        BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindingResource,
        BufferDescriptor, ComputePipelineDescriptor, PipelineLayoutDescriptor,
        ShaderModuleDescriptor, ShaderSource,
    },
    ids::{
        BindGroupId, BindGroupLayoutId, BufferId, ComputePipelineId, PipelineLayoutId,
        ShaderModuleId,
    },
};
use wgpu_remote_server::{Engine, run_connection};
use wgpu_remote_transport::quic::QuicEndpoint;
use wgpu_types::{
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferUsages, ShaderStages,
};

const DOUBLE_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read_write> data: array<u32>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    data[gid.x] = data[gid.x] * 2u;
}
"#;

async fn ok(client: &Client<impl wgpu_remote_transport::Connection + Clone>, action: Action) {
    match client.request(action).await.expect("client request") {
        Response::Ok => {}
        other => panic!("expected Ok, got {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn compute_double_over_quic() -> anyhow::Result<()> {
    // Install the rustls default crypto provider for the process. Tests in
    // the same process race on this; using a OnceCell-style guard keeps it
    // safe even when run with --test-threads > 1.
    let _ = rustls::crypto::ring::default_provider().install_default();

    // 1. Spin up server endpoint.
    let (server_endpoint, server_cert) =
        QuicEndpoint::server("127.0.0.1:0".parse().unwrap()).expect("server endpoint");
    let server_addr = server_endpoint.local_addr().expect("local_addr");

    // 2. Spawn engine + accept loop.
    let engine = Arc::new(Engine::new().await?);
    let engine_for_loop = engine.clone();
    let accept_handle = tokio::spawn(async move {
        // We expect exactly one connection in this test.
        let conn = server_endpoint.accept().await.expect("accept");
        run_connection(engine_for_loop, conn).await.unwrap();
    });

    // 3. Client endpoint that pins the server's self-signed cert.
    let client_endpoint = QuicEndpoint::client(server_cert).expect("client endpoint");
    let connection = client_endpoint
        .connect(server_addr, "localhost")
        .await
        .expect("connect");
    let client = Client::new(connection);

    // 4. Resource IDs (chosen by hand for the test).
    let storage_id = BufferId::new(1);
    let staging_id = BufferId::new(2);
    let shader_id = ShaderModuleId::new(10);
    let bgl_id = BindGroupLayoutId::new(20);
    let layout_id = PipelineLayoutId::new(21);
    let bg_id = BindGroupId::new(22);
    let pipeline_id = ComputePipelineId::new(30);

    let input: Vec<u32> = (1..=8).collect();
    let input_bytes: Vec<u8> = input.iter().flat_map(|n| n.to_ne_bytes()).collect();
    let buffer_size = input_bytes.len() as u64;

    // 5. Storage + staging buffers.
    ok(
        &client,
        Action::CreateBuffer {
            id: storage_id,
            desc: BufferDescriptor {
                label: Some("storage".into()),
                size: buffer_size,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            },
        },
    )
    .await;
    ok(
        &client,
        Action::CreateBuffer {
            id: staging_id,
            desc: BufferDescriptor {
                label: Some("staging".into()),
                size: buffer_size,
                usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
                mapped_at_creation: false,
            },
        },
    )
    .await;

    // 6. Upload input.
    ok(
        &client,
        Action::WriteBuffer {
            buffer: storage_id,
            offset: 0,
            data: Bytes::from(input_bytes.clone()),
        },
    )
    .await;

    // 7. Shader → BGL → BG → pipeline layout → pipeline.
    ok(
        &client,
        Action::CreateShaderModule {
            id: shader_id,
            desc: ShaderModuleDescriptor {
                label: Some("double".into()),
                source: ShaderSource::Wgsl(DOUBLE_SHADER.into()),
            },
        },
    )
    .await;
    ok(
        &client,
        Action::CreateBindGroupLayout {
            id: bgl_id,
            desc: BindGroupLayoutDescriptor {
                label: Some("storage-bgl".into()),
                entries: vec![BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(buffer_size),
                    },
                    count: None,
                }],
            },
        },
    )
    .await;
    ok(
        &client,
        Action::CreateBindGroup {
            id: bg_id,
            desc: BindGroupDescriptor {
                label: Some("storage-bg".into()),
                layout: bgl_id,
                entries: vec![BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Buffer {
                        buffer: storage_id,
                        offset: 0,
                        size: None,
                    },
                }],
            },
        },
    )
    .await;
    ok(
        &client,
        Action::CreatePipelineLayout {
            id: layout_id,
            desc: PipelineLayoutDescriptor {
                label: Some("compute-layout".into()),
                bind_group_layouts: vec![bgl_id],
                push_constant_ranges: vec![],
            },
        },
    )
    .await;
    ok(
        &client,
        Action::CreateComputePipeline {
            id: pipeline_id,
            desc: ComputePipelineDescriptor {
                label: Some("double-pipeline".into()),
                layout: Some(layout_id),
                module: shader_id,
                entry_point: Some("main".into()),
                constants: vec![],
            },
        },
    )
    .await;

    // 8. Submit compute pass + copy.
    let recording = CommandBufferRecording {
        label: Some("double-cmds".into()),
        commands: vec![
            EncoderCommand::BeginComputePass {
                label: Some("double-pass".into()),
                commands: vec![
                    ComputeCommand::SetPipeline(pipeline_id),
                    ComputeCommand::SetBindGroup {
                        index: 0,
                        group: bg_id,
                        offsets: vec![],
                    },
                    ComputeCommand::DispatchWorkgroups {
                        x: input.len() as u32,
                        y: 1,
                        z: 1,
                    },
                ],
            },
            EncoderCommand::CopyBufferToBuffer {
                source: storage_id,
                source_offset: 0,
                destination: staging_id,
                destination_offset: 0,
                size: buffer_size,
            },
        ],
    };
    ok(
        &client,
        Action::Submit {
            recordings: vec![recording.encode()?],
        },
    )
    .await;

    // 9. Map for read.
    let response = client
        .request(Action::MapBufferForRead {
            buffer: staging_id,
            offset: 0,
            size: buffer_size,
        })
        .await?;
    let bytes = match response {
        Response::BufferData { data } => data,
        other => panic!("expected BufferData, got {other:?}"),
    };
    let output: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|c| u32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let expected: Vec<u32> = input.iter().map(|n| n * 2).collect();
    assert_eq!(output, expected);

    // 10. Clean up: drop the client connection so the server's accept loop
    //     observes the close and returns.
    drop(client);
    drop(client_endpoint);
    let _ = tokio::time::timeout(std::time::Duration::from_secs(2), accept_handle).await;

    Ok(())
}
