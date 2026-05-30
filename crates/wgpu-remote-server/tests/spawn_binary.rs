//! Spawn the `wgpu-remote-server` binary as a child process, connect a
//! client over QUIC, run round-trip workloads. Proves the binary is actually
//! usable end-to-end (not just the in-process `run_connection` helper).

use std::num::NonZeroU64;
use std::process::Stdio;
use std::time::Duration;

use bytes::Bytes;
use rustls::pki_types::CertificateDer;
use tokio::process::{Child, Command};
use tokio::time::sleep;
use wgpu_remote_client::Client;
use wgpu_remote_protocol::{
    Action, PROTOCOL_VERSION, Response,
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
use wgpu_remote_transport::{
    Connection,
    quic::{QuicConnection, QuicEndpoint},
};
use wgpu_types::{
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferUsages, ShaderStages,
};

const SERVER_BIN: &str = env!("CARGO_BIN_EXE_wgpu-remote-server");
const DOUBLE_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read_write> data: array<u32>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    data[gid.x] = data[gid.x] * 2u;
}
"#;

/// Spin up the server binary in self-signed mode. Returns the live child
/// process, a connected QUIC client endpoint, and a [`Client`] talking to it.
async fn spawn_server_self_signed() -> anyhow::Result<(Child, QuicEndpoint, Client<QuicConnection>)>
{
    let _ = rustls::crypto::ring::default_provider().install_default();

    let tmp = tempdir()?;
    let cert_path = tmp.join("server-cert.der");
    let port_path = tmp.join("port");

    let child = Command::new(SERVER_BIN)
        .arg("--self-signed")
        .arg("--bind")
        .arg("127.0.0.1:0")
        .arg("--cert-out")
        .arg(&cert_path)
        .arg("--port-file")
        .arg(&port_path)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .kill_on_drop(true)
        .spawn()?;

    let port = wait_for_file(&port_path, Duration::from_secs(10))
        .await?
        .trim()
        .parse::<u16>()?;
    let cert_der = wait_for_file_bytes(&cert_path, Duration::from_secs(10)).await?;

    let endpoint = QuicEndpoint::client(CertificateDer::from(cert_der))?;
    let connection = endpoint
        .connect(format!("127.0.0.1:{port}").parse()?, "localhost")
        .await?;
    let client = Client::new(connection);
    Ok((child, endpoint, client))
}

/// Spin up the server binary in CA mode. Returns the live child process,
/// a connected QUIC client endpoint, and a [`Client`] talking to it.
async fn spawn_server_ca() -> anyhow::Result<(Child, QuicEndpoint, Client<QuicConnection>)> {
    let _ = rustls::crypto::ring::default_provider().install_default();

    let tmp = tempdir()?;
    let ca_cert_path = tmp.join("ca-cert.der");
    let ca_key_path = tmp.join("ca-key.der");
    let port_path = tmp.join("port");

    let child = Command::new(SERVER_BIN)
        .arg("--bind")
        .arg("127.0.0.1:0")
        .arg("--ca-cert")
        .arg(&ca_cert_path)
        .arg("--ca-key")
        .arg(&ca_key_path)
        .arg("--port-file")
        .arg(&port_path)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .kill_on_drop(true)
        .spawn()?;

    let port = wait_for_file(&port_path, Duration::from_secs(10))
        .await?
        .trim()
        .parse::<u16>()?;
    let ca_cert_der = wait_for_file_bytes(&ca_cert_path, Duration::from_secs(10)).await?;

    let endpoint = QuicEndpoint::client(CertificateDer::from(ca_cert_der))?;
    let connection = endpoint
        .connect(format!("127.0.0.1:{port}").parse()?, "localhost")
        .await?;
    let client = Client::new(connection);
    Ok((child, endpoint, client))
}

async fn ok<C: Connection + Clone>(client: &Client<C>, action: Action) {
    match client.request(action).await.expect("client request") {
        Response::Ok => {}
        other => panic!("expected Ok, got {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn binary_handshake() -> anyhow::Result<()> {
    let (mut child, endpoint, client) = spawn_server_self_signed().await?;

    let resp = client
        .request(Action::Hello {
            protocol_version: PROTOCOL_VERSION,
        })
        .await?;
    match resp {
        Response::HelloAck { protocol_version } => {
            assert_eq!(protocol_version, PROTOCOL_VERSION);
        }
        other => panic!("expected HelloAck, got {other:?}"),
    }

    drop(client);
    drop(endpoint);
    child.kill().await?;
    Ok(())
}

/// Handshake using the CA-mode server. Proves the CA → server cert → client
/// trust chain works end-to-end across a real process boundary.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn binary_handshake_ca() -> anyhow::Result<()> {
    let (mut child, endpoint, client) = spawn_server_ca().await?;

    let resp = client
        .request(Action::Hello {
            protocol_version: PROTOCOL_VERSION,
        })
        .await?;
    match resp {
        Response::HelloAck { protocol_version } => {
            assert_eq!(protocol_version, PROTOCOL_VERSION);
        }
        other => panic!("expected HelloAck, got {other:?}"),
    }

    drop(client);
    drop(endpoint);
    child.kill().await?;
    Ok(())
}

/// The full compute_double scenario, but the server is a *separate process*
/// reached over a real UDP socket. Same workload as the in-process
/// `quic_compute` test in `wgpu-remote-tests`, exercised here against a
/// freshly-built `wgpu-remote-server` binary.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn binary_compute_double() -> anyhow::Result<()> {
    let (mut child, endpoint, client) = spawn_server_self_signed().await?;

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
    ok(
        &client,
        Action::WriteBuffer {
            buffer: storage_id,
            offset: 0,
            data: Bytes::from(input_bytes.clone()),
        },
    )
    .await;
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

    drop(client);
    drop(endpoint);
    child.kill().await?;
    Ok(())
}

// Tiny helpers — tempfile crate isn't worth a dep for one test.

fn tempdir() -> std::io::Result<std::path::PathBuf> {
    let base = std::env::temp_dir();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let pid = std::process::id();
    let dir = base.join(format!("wgpu-remote-test-{pid}-{nanos}"));
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

async fn wait_for_file(path: &std::path::Path, timeout: Duration) -> anyhow::Result<String> {
    let start = std::time::Instant::now();
    loop {
        match std::fs::read_to_string(path) {
            Ok(s) if !s.is_empty() => return Ok(s),
            _ => {}
        }
        if start.elapsed() > timeout {
            anyhow::bail!("timed out waiting for {}", path.display());
        }
        sleep(Duration::from_millis(50)).await;
    }
}

async fn wait_for_file_bytes(
    path: &std::path::Path,
    timeout: Duration,
) -> anyhow::Result<Vec<u8>> {
    let start = std::time::Instant::now();
    loop {
        match std::fs::read(path) {
            Ok(b) if !b.is_empty() => return Ok(b),
            _ => {}
        }
        if start.elapsed() > timeout {
            anyhow::bail!("timed out waiting for {}", path.display());
        }
        sleep(Duration::from_millis(50)).await;
    }
}
