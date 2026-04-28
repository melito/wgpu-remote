//! `wgpu-remote-cli` — connect to a wgpu-remote-server and run a workload.
//!
//! Two subcommands:
//!   - `ping`: open a connection and exchange the protocol handshake. Smoke
//!     test for connectivity.
//!   - `compute-double`: run a WGSL compute shader that doubles each u32 in
//!     a buffer of N elements, print input and output. Validates the full
//!     GPU dispatch + readback path.
//!
//! In v1 the server uses a self-signed cert that the client must pin via
//! `--cert <PATH>`. The DER file is whatever `wgpu-remote-server` writes via
//! `--cert-out`.

use std::net::SocketAddr;
use std::num::NonZeroU64;
use std::path::PathBuf;
use std::process::ExitCode;

use bytes::Bytes;
use rustls::pki_types::CertificateDer;
use wgpu_remote_client::{
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferUsages, Client, Instance,
    ShaderStages,
    descriptors::{
        BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindingResource,
        BufferDescriptor, ComputePipelineDescriptor, PipelineLayoutDescriptor,
        ShaderModuleDescriptor, ShaderSource,
    },
};
use wgpu_remote_protocol::{Action, PROTOCOL_VERSION, Response};
use wgpu_remote_transport::quic::QuicEndpoint;

const USAGE: &str = "\
wgpu-remote-cli — demo client for wgpu-remote-server

USAGE:
    wgpu-remote-cli <SUBCOMMAND>

SUBCOMMANDS:
    ping              Connect, exchange protocol handshake, disconnect.
    compute-double    Run a doubling compute shader on integers 1..=N.

COMMON OPTIONS:
    --server <ADDR>    Server address  [default: 127.0.0.1:4433]
    --cert <PATH>      Path to the server's DER cert  [default: ./server-cert.der]
    --server-name <S>  Server name to verify against the cert  [default: localhost]
    -h, --help         Print this help

`compute-double` extra options:
    --count <N>        Number of u32 values to double  [default: 8]
";

#[derive(Debug)]
struct Common {
    server: SocketAddr,
    cert: PathBuf,
    server_name: String,
}

#[derive(Debug)]
enum Cmd {
    Ping(Common),
    ComputeDouble(Common, u32),
}

fn parse_args() -> Result<Cmd, String> {
    let mut argv = std::env::args().skip(1);
    let sub = argv.next().ok_or_else(|| "missing subcommand".to_string())?;
    match sub.as_str() {
        "-h" | "--help" => {
            print!("{USAGE}");
            std::process::exit(0);
        }
        "ping" | "compute-double" => {}
        other => return Err(format!("unknown subcommand: {other}")),
    }

    let mut server: SocketAddr = "127.0.0.1:4433".parse().unwrap();
    let mut cert = PathBuf::from("./server-cert.der");
    let mut server_name = "localhost".to_string();
    let mut count: u32 = 8;

    while let Some(arg) = argv.next() {
        match arg.as_str() {
            "--server" => server = argv.next().ok_or("--server needs a value")?.parse().map_err(|e| format!("invalid --server: {e}"))?,
            "--cert" => cert = PathBuf::from(argv.next().ok_or("--cert needs a value")?),
            "--server-name" => server_name = argv.next().ok_or("--server-name needs a value")?,
            "--count" => count = argv.next().ok_or("--count needs a value")?.parse().map_err(|e| format!("invalid --count: {e}"))?,
            "-h" | "--help" => {
                print!("{USAGE}");
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument: {other}")),
        }
    }

    let common = Common {
        server,
        cert,
        server_name,
    };
    Ok(match sub.as_str() {
        "ping" => Cmd::Ping(common),
        "compute-double" => Cmd::ComputeDouble(common, count),
        _ => unreachable!(),
    })
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> ExitCode {
    if let Err(e) = real_main().await {
        eprintln!("error: {e}");
        ExitCode::from(1)
    } else {
        ExitCode::SUCCESS
    }
}

async fn real_main() -> anyhow::Result<()> {
    let cmd = match parse_args() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: {e}\n");
            eprint!("{USAGE}");
            std::process::exit(2);
        }
    };

    let _ = rustls::crypto::ring::default_provider().install_default();

    match cmd {
        Cmd::Ping(c) => ping(c).await,
        Cmd::ComputeDouble(c, n) => compute_double(c, n).await,
    }
}

async fn connect(c: &Common) -> anyhow::Result<wgpu_remote_transport::quic::QuicConnection> {
    let cert_bytes = std::fs::read(&c.cert)
        .map_err(|e| anyhow::anyhow!("read cert {}: {e}", c.cert.display()))?;
    let endpoint = QuicEndpoint::client(CertificateDer::from(cert_bytes))?;
    let connection = endpoint.connect(c.server, &c.server_name).await?;
    // Leak the endpoint so the connection stays alive past this fn — we hand
    // back the connection to the caller, which owns the rest of the lifetime.
    std::mem::forget(endpoint);
    Ok(connection)
}

async fn ping(c: Common) -> anyhow::Result<()> {
    println!("connecting to {}…", c.server);
    let connection = connect(&c).await?;
    let client = Client::new(connection);

    let response = client
        .request(Action::Hello {
            protocol_version: PROTOCOL_VERSION,
        })
        .await?;
    match response {
        Response::HelloAck { protocol_version } => {
            println!(
                "ok — server speaks protocol v{protocol_version} (client v{PROTOCOL_VERSION})"
            );
            Ok(())
        }
        Response::Error { code, message } => {
            anyhow::bail!("server returned error {:?}: {}", code, message);
        }
        other => anyhow::bail!("expected HelloAck, got {other:?}"),
    }
}

const DOUBLE_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read_write> data: array<u32>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    data[gid.x] = data[gid.x] * 2u;
}
"#;

async fn compute_double(c: Common, count: u32) -> anyhow::Result<()> {
    if count == 0 {
        anyhow::bail!("--count must be at least 1");
    }
    println!("connecting to {}…", c.server);
    let connection = connect(&c).await?;

    let instance = Instance::new(Client::new(connection));
    let adapter = instance.request_adapter().await?;
    let (device, queue) = adapter.request_device().await?;
    println!("connected and got device — building workload");

    let input: Vec<u32> = (1..=count).collect();
    let input_bytes: Bytes = input
        .iter()
        .flat_map(|n| n.to_ne_bytes())
        .collect::<Vec<u8>>()
        .into();
    let buffer_size = input_bytes.len() as u64;

    let storage = device
        .create_buffer(&BufferDescriptor {
            label: Some("storage".into()),
            size: buffer_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
        .await?;
    let staging = device
        .create_buffer(&BufferDescriptor {
            label: Some("staging".into()),
            size: buffer_size,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        })
        .await?;

    queue.write_buffer(&storage, 0, input_bytes.clone()).await?;

    let shader = device
        .create_shader_module(ShaderModuleDescriptor {
            label: Some("double".into()),
            source: ShaderSource::Wgsl(DOUBLE_SHADER.into()),
        })
        .await?;

    let bgl = device
        .create_bind_group_layout(BindGroupLayoutDescriptor {
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
        })
        .await?;

    let bind_group = device
        .create_bind_group(BindGroupDescriptor {
            label: Some("storage-bg".into()),
            layout: bgl.id(),
            entries: vec![BindGroupEntry {
                binding: 0,
                resource: BindingResource::Buffer {
                    buffer: storage.id(),
                    offset: 0,
                    size: None,
                },
            }],
        })
        .await?;

    let pipeline_layout = device
        .create_pipeline_layout(PipelineLayoutDescriptor {
            label: Some("compute-layout".into()),
            bind_group_layouts: vec![bgl.id()],
            push_constant_ranges: vec![],
        })
        .await?;

    let pipeline = device
        .create_compute_pipeline(ComputePipelineDescriptor {
            label: Some("double-pipeline".into()),
            layout: Some(pipeline_layout.id()),
            module: shader.id(),
            entry_point: Some("main".into()),
            constants: vec![],
        })
        .await?;

    let mut encoder = device.create_command_encoder(Some("double-cmds".into()));
    {
        let mut pass = encoder.begin_compute_pass(Some("double-pass".into()));
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(count, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&storage, 0, &staging, 0, buffer_size);
    queue.submit([encoder.finish()]).await?;

    let bytes = staging.read_all().await?;
    let output: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|c| u32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    println!(" input: {:?}", input);
    println!("output: {:?}", output);
    let expected: Vec<u32> = input.iter().map(|n| n * 2).collect();
    if output == expected {
        println!("ok — every value was doubled");
    } else {
        println!("MISMATCH — expected {:?}", expected);
        anyhow::bail!("output did not match expected");
    }

    Ok(())
}
