//! `wgpu-remote-cli` — connect to a wgpu-remote-server and run a workload.
//!
//! Subcommands:
//!   - `ping`: connection + handshake smoke test.
//!   - `compute-double`: WGSL compute shader doubling each u32 in a buffer.
//!   - `render-checkerboard`: render an N×M checkerboard into a texture, copy
//!     to a staging buffer, write out as a PPM (P6) image. The marquee
//!     visual demo for the render path.
//!
//! Supports two transports:
//!   - QUIC (default): connects to a server by IP:port with a CA cert.
//!   - iroh (`--iroh`): connects by endpoint ID — no certs, NAT-traversed.

use std::net::SocketAddr;
use std::num::NonZeroU64;
use std::path::PathBuf;
use std::process::ExitCode;

// Binaries are their own compilation unit, so they don't inherit the lib's
// `extern crate … as wgpu_types` alias — re-establish it here.
#[cfg(feature = "wgpu-27")]
extern crate wgpu_types_27 as wgpu_types;
#[cfg(feature = "wgpu-29")]
extern crate wgpu_types_29 as wgpu_types;

use bytes::Bytes;
use rustls::pki_types::CertificateDer;
use wgpu_remote::protocol::{Action, PROTOCOL_VERSION, Response};
use wgpu_remote::transport::Connection;
use wgpu_types::{
    Color, ColorTargetState, ColorWrites, Extent3d, LoadOp, MultisampleState, Operations,
    Origin3d, PrimitiveState, StoreOp, TexelCopyBufferLayout, TextureAspect, TextureDimension,
    TextureFormat, TextureUsages,
};

const USAGE: &str = "\
wgpu-remote-cli — demo client for wgpu-remote-server

USAGE:
    wgpu-remote-cli [OPTIONS] <SUBCOMMAND>

SUBCOMMANDS:
    ping                  Connect, exchange protocol handshake, disconnect.
    compute-double        Run a doubling compute shader on integers 1..=N.
    render-checkerboard   Render a checkerboard pattern, write the result as a
                          PPM image. Visual demo of the render path.

TRANSPORT:
    --iroh                 Use iroh transport (NAT-traversed QUIC).
    --endpoint-id <ID>     Server's iroh endpoint ID (required with --iroh).

QUIC OPTIONS (default transport):
    --server <ADDR>    Server address  [default: 127.0.0.1:4433]
    --ca-cert <PATH>   Path to the CA cert (DER) — or a self-signed server
                       cert for legacy mode  [default: ./ca-cert.der]
    --cert <PATH>      Alias for --ca-cert
    --server-name <S>  Server name to verify against the cert  [default: localhost]

SUBCOMMAND OPTIONS:
    --count <N>        (compute-double) Number of u32 values  [default: 8]
    --width <N>        (render-checkerboard) Image width      [default: 256]
    --height <N>       (render-checkerboard) Image height     [default: 256]
    --tile <N>         (render-checkerboard) Tile size         [default: 32]
    --output <PATH>    (render-checkerboard) PPM output path   [default: ./checkerboard.ppm]
    -h, --help         Print this help
";

#[derive(Debug)]
struct QuicOpts {
    server: SocketAddr,
    ca_cert: PathBuf,
    server_name: String,
}

#[derive(Debug)]
struct IrohOpts {
    endpoint_id: String,
}

#[derive(Debug)]
enum TransportOpts {
    Quic(QuicOpts),
    Iroh(IrohOpts),
}

#[derive(Debug)]
struct CheckerboardArgs {
    width: u32,
    height: u32,
    tile: u32,
    output: PathBuf,
}

#[derive(Debug)]
enum Workload {
    Ping,
    ComputeDouble(u32),
    RenderCheckerboard(CheckerboardArgs),
}

#[derive(Debug)]
struct Cmd {
    transport: TransportOpts,
    workload: Workload,
}

fn parse_args() -> Result<Cmd, String> {
    let mut argv = std::env::args().skip(1);

    let mut sub: Option<String> = None;
    let mut server: SocketAddr = "127.0.0.1:4433".parse().unwrap();
    let mut ca_cert = PathBuf::from("./ca-cert.der");
    let mut server_name = "localhost".to_string();
    let mut iroh = false;
    let mut endpoint_id: Option<String> = None;
    let mut count: u32 = 8;
    let mut width: u32 = 256;
    let mut height: u32 = 256;
    let mut tile: u32 = 32;
    let mut output = PathBuf::from("./checkerboard.ppm");

    while let Some(arg) = argv.next() {
        match arg.as_str() {
            "--server" => server = argv.next().ok_or("--server needs a value")?.parse().map_err(|e| format!("invalid --server: {e}"))?,
            "--ca-cert" | "--cert" => ca_cert = PathBuf::from(argv.next().ok_or("--ca-cert needs a value")?),
            "--server-name" => server_name = argv.next().ok_or("--server-name needs a value")?,
            "--iroh" => iroh = true,
            "--endpoint-id" => endpoint_id = Some(argv.next().ok_or("--endpoint-id needs a value")?),
            "--count" => count = argv.next().ok_or("--count needs a value")?.parse().map_err(|e| format!("invalid --count: {e}"))?,
            "--width" => width = argv.next().ok_or("--width needs a value")?.parse().map_err(|e| format!("invalid --width: {e}"))?,
            "--height" => height = argv.next().ok_or("--height needs a value")?.parse().map_err(|e| format!("invalid --height: {e}"))?,
            "--tile" => tile = argv.next().ok_or("--tile needs a value")?.parse().map_err(|e| format!("invalid --tile: {e}"))?,
            "--output" => output = PathBuf::from(argv.next().ok_or("--output needs a value")?),
            "-h" | "--help" => {
                print!("{USAGE}");
                std::process::exit(0);
            }
            other if !other.starts_with('-') && sub.is_none() => {
                match other {
                    "ping" | "compute-double" | "render-checkerboard" => sub = Some(other.to_string()),
                    _ => return Err(format!("unknown subcommand: {other}")),
                }
            }
            other => return Err(format!("unknown argument: {other}")),
        }
    }

    let sub = sub.ok_or_else(|| "missing subcommand".to_string())?;

    let transport = if iroh {
        let id = endpoint_id.ok_or("--iroh requires --endpoint-id")?;
        TransportOpts::Iroh(IrohOpts { endpoint_id: id })
    } else {
        TransportOpts::Quic(QuicOpts {
            server,
            ca_cert,
            server_name,
        })
    };

    let workload = match sub.as_str() {
        "ping" => Workload::Ping,
        "compute-double" => Workload::ComputeDouble(count),
        "render-checkerboard" => Workload::RenderCheckerboard(CheckerboardArgs {
            width,
            height,
            tile,
            output,
        }),
        _ => unreachable!(),
    };

    Ok(Cmd { transport, workload })
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

    match cmd.transport {
        TransportOpts::Quic(opts) => {
            println!("connecting to {} via QUIC…", opts.server);
            let conn = connect_quic(&opts).await?;
            let client = wgpu_remote::client::Client::new(conn);
            run_workload(client, cmd.workload).await
        }
        TransportOpts::Iroh(opts) => {
            println!("connecting to {} via iroh…", &opts.endpoint_id[..12.min(opts.endpoint_id.len())]);
            let conn = connect_iroh(&opts).await?;
            let client = wgpu_remote::client::Client::new(conn);
            run_workload(client, cmd.workload).await
        }
    }
}

async fn connect_quic(
    opts: &QuicOpts,
) -> anyhow::Result<wgpu_remote::transport::quic::QuicConnection> {
    use wgpu_remote::transport::quic::QuicEndpoint;
    let cert_bytes = std::fs::read(&opts.ca_cert)
        .map_err(|e| anyhow::anyhow!("read cert {}: {e}", opts.ca_cert.display()))?;
    let endpoint = QuicEndpoint::client(CertificateDer::from(cert_bytes))?;
    let connection = endpoint.connect(opts.server, &opts.server_name).await?;
    std::mem::forget(endpoint);
    Ok(connection)
}

async fn connect_iroh(
    opts: &IrohOpts,
) -> anyhow::Result<wgpu_remote::transport::iroh::IrohConnection> {
    use wgpu_remote::transport::Transport;
    use wgpu_remote::transport::iroh::IrohEndpoint;

    let ep = IrohEndpoint::with_discovery().await?;
    ep.endpoint().online().await;

    let remote_id: wgpu_remote::transport::iroh::EndpointId = opts
        .endpoint_id
        .parse()
        .map_err(|e| anyhow::anyhow!("invalid endpoint ID: {e}"))?;
    let addr = wgpu_remote::transport::iroh::EndpointAddr::new(remote_id);
    let conn = ep.dial(addr).await?;
    // Keep endpoint alive for the connection's lifetime.
    std::mem::forget(ep);
    Ok(conn)
}

// ---------------------------------------------------------------------------
// Workload dispatch (generic over any Connection)
// ---------------------------------------------------------------------------

async fn run_workload<C>(client: wgpu_remote::client::Client<C>, workload: Workload) -> anyhow::Result<()>
where
    C: Connection + Clone + 'static,
{
    match workload {
        Workload::Ping => ping(client).await,
        Workload::ComputeDouble(n) => compute_double(client, n).await,
        Workload::RenderCheckerboard(args) => render_checkerboard(client, args).await,
    }
}

async fn ping<C: Connection + Clone + 'static>(
    client: wgpu_remote::client::Client<C>,
) -> anyhow::Result<()> {
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

async fn compute_double<C: Connection + Clone + 'static>(
    client: wgpu_remote::client::Client<C>,
    count: u32,
) -> anyhow::Result<()> {
    use wgpu_remote::client::*;

    if count == 0 {
        anyhow::bail!("--count must be at least 1");
    }

    let instance = Instance::new(client);
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

    let storage = device.create_buffer(&descriptors::BufferDescriptor {
        label: Some("storage".into()),
        size: buffer_size,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging = device.create_buffer(&descriptors::BufferDescriptor {
        label: Some("staging".into()),
        size: buffer_size,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    queue.write_buffer(&storage, 0, input_bytes.clone());

    let shader = device.create_shader_module(descriptors::ShaderModuleDescriptor {
        label: Some("double".into()),
        source: descriptors::ShaderSource::Wgsl(DOUBLE_SHADER.into()),
    });

    let bgl = device.create_bind_group_layout(descriptors::BindGroupLayoutDescriptor {
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
    });

    let bind_group = device.create_bind_group(descriptors::BindGroupDescriptor {
        label: Some("storage-bg".into()),
        layout: bgl.id(),
        entries: vec![descriptors::BindGroupEntry {
            binding: 0,
            resource: descriptors::BindingResource::Buffer {
                buffer: storage.id(),
                offset: 0,
                size: None,
            },
        }],
    });

    let pipeline_layout = device.create_pipeline_layout(descriptors::PipelineLayoutDescriptor {
        label: Some("compute-layout".into()),
        bind_group_layouts: vec![bgl.id()],
        #[cfg(feature = "wgpu-27")]
        push_constant_ranges: vec![],
    });

    let pipeline = device.create_compute_pipeline(descriptors::ComputePipelineDescriptor {
        label: Some("double-pipeline".into()),
        layout: Some(pipeline_layout.id()),
        module: shader.id(),
        entry_point: Some("main".into()),
        constants: vec![],
    });

    let mut encoder = device.create_command_encoder(Some("double-cmds".into()));
    {
        let mut pass = encoder.begin_compute_pass(Some("double-pass".into()));
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(count, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&storage, 0, &staging, 0, buffer_size);
    queue.submit([encoder.finish()])?;

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

const CHECKERBOARD_SHADER: &str = r#"
struct Params {
    tile_size: u32,
    width: u32,
    height: u32,
    _pad: u32,
};
@group(0) @binding(0) var<uniform> params: Params;

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    return vec4<f32>(positions[i], 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let cell_x = u32(frag.x) / params.tile_size;
    let cell_y = u32(frag.y) / params.tile_size;
    let dark = ((cell_x + cell_y) & 1u) == 0u;
    if (dark) {
        // Deep blue
        return vec4<f32>(0.10, 0.15, 0.40, 1.0);
    } else {
        // Warm gold
        return vec4<f32>(0.95, 0.75, 0.20, 1.0);
    }
}
"#;

/// `COPY_BYTES_PER_ROW_ALIGNMENT` for `copy_texture_to_buffer`.
const COPY_BPR_ALIGNMENT: u32 = 256;

fn align_up(v: u32, align: u32) -> u32 {
    v.div_ceil(align) * align
}

async fn render_checkerboard<C: Connection + Clone + 'static>(
    client: wgpu_remote::client::Client<C>,
    args: CheckerboardArgs,
) -> anyhow::Result<()> {
    use wgpu_remote::client::*;

    if args.width == 0 || args.height == 0 || args.tile == 0 {
        anyhow::bail!("--width, --height, and --tile must all be at least 1");
    }

    let instance = Instance::new(client);
    let adapter = instance.request_adapter().await?;
    let (device, queue) = adapter.request_device().await?;
    println!(
        "rendering {}×{} checkerboard with {}-pixel tiles…",
        args.width, args.height, args.tile
    );

    // 1. Render target.
    let texture = device.create_texture(&descriptors::TextureDescriptor {
        label: Some("checkerboard".into()),
        size: Extent3d {
            width: args.width,
            height: args.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
        view_formats: vec![],
    });
    let view = texture.create_view(descriptors::TextureViewDescriptor::default());

    // 2. Uniform buffer carrying tile size + image dimensions.
    let unpadded_bpr = args.width * 4;
    let padded_bpr = align_up(unpadded_bpr, COPY_BPR_ALIGNMENT);
    let staging_size = (padded_bpr * args.height) as u64;

    #[repr(C)]
    #[derive(Copy, Clone)]
    struct Params {
        tile_size: u32,
        width: u32,
        height: u32,
        _pad: u32,
    }
    let params = Params {
        tile_size: args.tile,
        width: args.width,
        height: args.height,
        _pad: 0,
    };
    let params_bytes: [u8; 16] = unsafe { std::mem::transmute(params) };

    let uniform = device.create_buffer(&descriptors::BufferDescriptor {
        label: Some("params".into()),
        size: 16,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&uniform, 0, Bytes::copy_from_slice(&params_bytes));

    let staging = device.create_buffer(&descriptors::BufferDescriptor {
        label: Some("staging".into()),
        size: staging_size,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    // 3. Bind group + pipeline.
    let bgl = device.create_bind_group_layout(descriptors::BindGroupLayoutDescriptor {
        label: Some("params-bgl".into()),
        entries: vec![BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::FRAGMENT,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: NonZeroU64::new(16),
            },
            count: None,
        }],
    });
    let bind_group = device.create_bind_group(descriptors::BindGroupDescriptor {
        label: Some("params-bg".into()),
        layout: bgl.id(),
        entries: vec![descriptors::BindGroupEntry {
            binding: 0,
            resource: descriptors::BindingResource::Buffer {
                buffer: uniform.id(),
                offset: 0,
                size: None,
            },
        }],
    });
    let pipeline_layout = device.create_pipeline_layout(descriptors::PipelineLayoutDescriptor {
        label: Some("checker-layout".into()),
        bind_group_layouts: vec![bgl.id()],
        #[cfg(feature = "wgpu-27")]
        push_constant_ranges: vec![],
    });
    let shader = device.create_shader_module(descriptors::ShaderModuleDescriptor {
        label: Some("checkerboard".into()),
        source: descriptors::ShaderSource::Wgsl(CHECKERBOARD_SHADER.into()),
    });
    let pipeline = device.create_render_pipeline(descriptors::RenderPipelineDescriptor {
        label: Some("checker-pipe".into()),
        layout: Some(pipeline_layout.id()),
        vertex: descriptors::VertexState {
            module: shader.id(),
            entry_point: Some("vs_main".into()),
            constants: vec![],
            buffers: vec![],
        },
        primitive: PrimitiveState::default(),
        depth_stencil: None,
        multisample: MultisampleState::default(),
        fragment: Some(descriptors::FragmentState {
            module: shader.id(),
            entry_point: Some("fs_main".into()),
            constants: vec![],
            targets: vec![Some(ColorTargetState {
                format: TextureFormat::Rgba8Unorm,
                blend: None,
                write_mask: ColorWrites::ALL,
            })],
        }),
        multiview: None,
    });

    // 4. Encode + submit.
    let mut encoder = device.create_command_encoder(Some("checker-cmds".into()));
    {
        let color_atts = [Some(ColorAttachment {
            view: &view,
            depth_slice: None,
            resolve_target: None,
            ops: Operations {
                load: LoadOp::Clear(Color::BLACK),
                store: StoreOp::Store,
            },
        })];
        let mut pass = encoder.begin_render_pass(RenderPassDescriptor {
            label: Some("checker-pass".into()),
            color_attachments: &color_atts,
            depth_stencil_attachment: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
    encoder.copy_texture_to_buffer(
        &texture,
        0,
        Origin3d::ZERO,
        TextureAspect::All,
        &staging,
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(padded_bpr),
            rows_per_image: Some(args.height),
        },
        Extent3d {
            width: args.width,
            height: args.height,
            depth_or_array_layers: 1,
        },
    );
    queue.submit([encoder.finish()])?;

    // 5. Read back, strip padding, write PPM (P6) RGB.
    let raw = staging.read_all().await?;
    let mut rgb = Vec::with_capacity((args.width * args.height * 3) as usize);
    for row in 0..args.height {
        let base = (row * padded_bpr) as usize;
        for col in 0..args.width {
            let p = base + (col * 4) as usize;
            rgb.push(raw[p]);
            rgb.push(raw[p + 1]);
            rgb.push(raw[p + 2]);
            // alpha discarded for PPM
        }
    }

    let header = format!("P6\n{} {}\n255\n", args.width, args.height);
    if args.output == std::path::Path::new("-") {
        use std::io::Write;
        let stdout = std::io::stdout();
        let mut out = stdout.lock();
        out.write_all(header.as_bytes())?;
        out.write_all(&rgb)?;
        out.flush()?;
        eprintln!(
            "ok — {}×{} PPM written to stdout ({} bytes)",
            args.width,
            args.height,
            header.len() + rgb.len()
        );
    } else {
        use std::io::Write;
        let mut f = std::fs::File::create(&args.output)?;
        f.write_all(header.as_bytes())?;
        f.write_all(&rgb)?;
        println!(
            "ok — {}×{} PPM written to {} ({} bytes)",
            args.width,
            args.height,
            args.output.display(),
            header.len() + rgb.len()
        );
    }

    Ok(())
}
