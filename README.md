# wgpu-remote

Run [wgpu](https://github.com/gfx-rs/wgpu) workloads on a GPU that lives on another machine.

Your code talks to a wgpu-shaped API. Under the hood every call is serialized, sent over QUIC (or [iroh](https://iroh.computer)) to a server process on a remote host, executed on a real GPU, and the results shipped back. The client machine doesn't need a GPU at all.

**Use cases:**

- Offload heavy compute or rendering from a laptop to a beefy GPU on your desk or in the cloud
- Target a specific GPU (4090, A100, …) without running your whole application on that machine
- Record, replay, or test GPU traffic through a clean RPC boundary

This is a research prototype — see [Status](#status) for what's working.

## Install

Everything ships in a single crate — `wgpu-remote` — gated by cargo features. Add it alongside a matching `wgpu`:

```toml
[dependencies]
# Client drop-in over QUIC (the default).
wgpu-remote = "0.1"
wgpu = "27"
```

```bash
# Or from the command line:
cargo add wgpu-remote wgpu@27
```

The client and server must agree on the exact `wgpu` version — this crate pins **wgpu 27**.

### Features

| Feature      | Default | Enables                                                                       |
|--------------|:-------:|-------------------------------------------------------------------------------|
| `client`     |   ✅    | wgpu drop-in (`install`, `connect_quic`, `connect_iroh`) + the wgpu-shaped facade |
| `quic`       |   ✅    | quinn-based QUIC transport + private CA                                        |
| `iroh`       |         | iroh transport — NAT-traversed QUIC, connect by endpoint ID (no certs/IPs)     |
| `server`     |         | replay engine as a library (`wgpu_remote::server`)                             |
| `server-bin` |         | the `wgpu-remote-server` binary (implies `server`)                             |
| `cli`        |         | the `wgpu-remote-cli` demo binary (implies `client`)                           |

```toml
# Client with iroh NAT traversal:
wgpu-remote = { version = "0.1", features = ["iroh"] }

# Embed a server in your own process:
wgpu-remote = { version = "0.1", features = ["server"] }

# Client-only, pick your transport explicitly:
wgpu-remote = { version = "0.1", default-features = false, features = ["client", "iroh"] }
```

### Binaries

The server and demo CLI are feature-gated binaries in the same crate:

```bash
# Installs `wgpu-remote-server` and `wgpu-remote-cli` onto your PATH.
# (both binaries support QUIC and iroh out of the box)
cargo install wgpu-remote --features "server-bin cli"
```

## Try it

### 1. Start the server

The server binds a QUIC socket and replays GPU commands against a real `wgpu::Device`.

```bash
# First run auto-generates a private CA (ca-cert.der + ca-key.der)
wgpu-remote-server
```

You should see the server listening on `0.0.0.0:4433`.

### 2. Ping the server

```bash
wgpu-remote-cli ping
```

This connects, exchanges a protocol handshake, and disconnects. If you see `ok — server speaks protocol v1`, the transport layer is working.

### 3. Run a compute shader

```bash
wgpu-remote-cli compute-double --count 16
```

Uploads `[1, 2, …, 16]` to a GPU storage buffer, dispatches a WGSL compute shader that doubles each value, copies the result to a staging buffer, maps it back, and prints input vs. output. Every step crosses the QUIC connection.

### 4. Render an image

```bash
wgpu-remote-cli render-checkerboard \
    --width 512 --height 512 --tile 64 \
    --output checkerboard.ppm
```

Allocates a render target, runs a fragment shader that paints a checkerboard, copies the texture to a staging buffer, reads it back, and writes a PPM image.

```bash
open checkerboard.ppm   # macOS; use xdg-open on Linux
```

### Over iroh (no certs, NAT-traversed)

iroh connects peers by endpoint ID — no IP addresses, no certificates, works across NATs.

```bash
# Terminal 1: start the server with iroh
wgpu-remote-server --iroh
# prints: endpoint id: 6c190b769dd5...
```

```bash
# Terminal 2: connect by endpoint ID
wgpu-remote-cli \
    --iroh --endpoint-id 6c190b769dd5... ping

wgpu-remote-cli \
    --iroh --endpoint-id 6c190b769dd5... compute-double --count 16
```

### Remote machines (QUIC)

For direct QUIC without iroh, copy `ca-cert.der` to the client and point both sides at it:

```bash
# Server (on the GPU machine)
wgpu-remote-server \
    --bind 0.0.0.0:4433 \
    --ca-cert ca-cert.der --ca-key ca-key.der \
    --san my-gpu-box.local --san 192.168.1.42

# Client (on your laptop)
wgpu-remote-cli compute-double \
    --server 192.168.1.42:4433 \
    --ca-cert ca-cert.der \
    --server-name my-gpu-box.local \
    --count 64
```

The CA cert is stable — server certs rotate on every restart without redistributing trust anchors.

## Architecture

```
┌────────────── client machine ──────────────┐
│  user code                                  │
│      │                                      │
│      ▼                                      │
│  wgpu::Instance / Device / Queue / ...      │  ← real wgpu types
│      │  (via wgpu_remote::install)          │
│      ▼                                      │
│  wgpu_remote::client::Instance / Device /   │
│  Buffer / Queue / CommandEncoder            │  ← wgpu-shaped facade
│      │                                      │
│      ▼                                      │
│  wgpu_remote::client::Client                │  ← Action / Response RPC
│      │                                      │
│      ▼                                      │
│  wgpu_remote::transport::Connection         │  ← pluggable transport
│  ┌─────────────┬───────────────────────┐    │
│  │ quinn (QUIC)│ iroh (NAT-traversed)  │    │
│  │ [quic]      │ [iroh]                │    │  ← cargo features
│  └─────────────┴───────────────────────┘    │
└────────────────│────────────────────────────┘
                 │  UDP + TLS 1.3
┌────────────────│────────────────────────────┐
│                ▼                            │
│  wgpu-remote-server (binary / [server])     │
│      │                                      │
│      ▼                                      │
│  wgpu_remote::server::Engine                │  ← replay against real wgpu
│      │                                      │
│      ▼                                      │
│  wgpu::Instance → adapter → device → GPU    │
└─────────────────────────────────────────────┘
```

### Why not a wgpu-hal backend?

"Just write a network backend for wgpu-hal" sounds natural, but:

- **wgpu-hal is unstable.** The trait surface breaks on most wgpu releases.
- **Some hal calls don't translate over a network**: `map_buffer` returns a raw `*mut u8`; `create_surface` takes a window handle.
- **The surface is large** (~150–200 methods).

We serialize at the level of typed actions on the user-facing surface (descriptors + command buffer recordings), not at the hal trait. Smaller surface, more stable across releases, free serde via `wgpu-types`.

## Using as a wgpu drop-in

`wgpu_remote::install(connection)` gives you a real `wgpu::Instance`. From there it's stock wgpu — no new types, no facade-specific APIs.

For the common case, skip straight to a connected instance with `wgpu_remote::connect_quic("127.0.0.1:4433", "ca-cert.der").await?` (or `connect_iroh(endpoint_id)` with the `iroh` feature). To manage the connection yourself:

```rust
use rustls::pki_types::CertificateDer;
use wgpu_remote::transport::quic::QuicEndpoint;

let ca_cert = CertificateDer::from(std::fs::read("ca-cert.der")?);
let endpoint = QuicEndpoint::client(ca_cert)?;
let connection = endpoint.connect(addr, "localhost").await?;

let instance: wgpu::Instance = wgpu_remote::install(connection);

// Everything below is plain wgpu.
let adapter = instance.request_adapter(&Default::default()).await?;
let (device, queue) = adapter.request_device(&Default::default()).await?;

let buffer = device.create_buffer(&wgpu::BufferDescriptor { /* ... */ });
queue.write_buffer(&buffer, 0, bytes);

let mut encoder = device.create_command_encoder(&Default::default());
{
    let mut pass = encoder.begin_compute_pass(&Default::default());
    pass.set_pipeline(&pipeline);
    pass.set_bind_group(0, &bind_group, &[]);
    pass.dispatch_workgroups(64, 1, 1);
}
queue.submit([encoder.finish()]);
```

Built on wgpu 27's `custom` cargo feature. Unsupported features route through `Device::on_uncaptured_error` rather than panicking.

## Using the facade directly

For async creates, simpler readback (`read_range(..)` / `read_all()`), or to skip the wgpu dispatch indirection, use the `wgpu_remote::client` facade directly:

```rust
use wgpu_remote::client::prelude::quic::*;

let instance = Instance::new(Client::new(connection));
let adapter  = instance.request_adapter().await?;
let (device, queue) = adapter.request_device().await?;
let buffer = device.create_buffer(&BufferDescriptor { /* ... */ });
let bytes = readback_buffer.read_all().await?;
```

The facade carries a `<C: Connection>` generic at every type; the `client::prelude::quic` module hides it for the shipping QUIC transport.

## Modules

Everything lives in the single `wgpu-remote` crate. Each module maps to a cargo feature so you only compile what you use.

| Path                        | Feature      | Purpose                                                                                   |
|-----------------------------|--------------|-------------------------------------------------------------------------------------------|
| `wgpu_remote` (root)        | `client`     | Entry points: `install`, `connect_quic`, `connect_iroh`.                                   |
| `wgpu_remote::protocol`     | always       | Wire format. Action/Response enums, typed IDs, descriptor mirrors, bincode codec. No I/O.  |
| `wgpu_remote::transport`    | always       | `Transport` + `Connection` traits, quinn QUIC impl (`quic`), private CA, iroh (`iroh`).    |
| `wgpu_remote::client`       | `client`     | Low-level `Client` + wgpu-shaped facade (`Instance`, `Device`, etc.).                      |
| `wgpu_remote::server`       | `server`     | Replay engine (`Engine`, `run_connection`).                                                |
| `wgpu-remote-server` (bin)  | `server-bin` | Server binary.                                                                             |
| `wgpu-remote-cli` (bin)     | `cli`        | Demo / smoke-test CLI (`ping`, `compute-double`, `render-checkerboard`).                   |

## Certificate management

```bash
# Generate a CA (once — keep the key secret)
wgpu-remote-server init-ca \
    --ca-cert ca-cert.der --ca-key ca-key.der

# Start the server — loads the CA and issues its own server cert
wgpu-remote-server \
    --ca-cert ca-cert.der --ca-key ca-key.der

# If no CA files exist, the server auto-generates one on first start
wgpu-remote-server

# For quick testing, --self-signed skips the CA entirely
wgpu-remote-server --self-signed
```

Add extra SANs for non-localhost use:

```bash
wgpu-remote-server \
    --ca-cert ca.der --ca-key ca.key \
    --bind 0.0.0.0:4433 \
    --san my-gpu-box.local \
    --san 192.168.1.42
```

## Tests

```bash
cargo test --all-features
```

Notable end-to-end tests:

| Test                                               | What it proves                                                                    |
|----------------------------------------------------|-----------------------------------------------------------------------------------|
| `protocol::*`                                      | Wire format roundtrips.                                                           |
| `engine_buffer` / `engine_compute`                 | Engine dispatches against a real GPU.                                             |
| `quic_compute`                                     | Compute workload over real QUIC, in-process.                                      |
| `binary_compute_double`                            | Spawns the server binary, runs a workload over a real socket across OS processes. |
| `binary_handshake_ca`                              | Server in CA mode — proves CA → server cert → client trust chain.                 |
| `facade_compute_double`                            | Compute workload using only wgpu-shaped facade types.                             |
| `facade_render_to_texture`                         | Fragment shader → texture → readback → per-pixel assertion.                       |
| `wgpu_drop_in_compute_double`                      | Compute workload using only `wgpu::*` types — the drop-in milestone.              |
| `unsupported_feature_routes_through_error_handler` | Unsupported features fire `on_uncaptured_error` instead of panicking.             |
| `bidi_stream_round_trip` / `datagram_round_trip`   | iroh transport bidi streams and datagrams over a local relay.                     |

## Status

**Working today:**
- Drop-in for stock wgpu (compute path) via `wgpu_remote::install`
- Buffers (create, write, copy, `map_async` for read)
- Textures, texture views, samplers
- Compute pipelines (WGSL shaders) + compute pass dispatch
- Render pipelines + render-to-texture (color attachments, load/store ops)
- Bind group layouts + bind groups
- Pipeline layouts (compute and render)
- Command encoder + compute pass + render pass recording
- `copy_buffer_to_buffer`, `copy_texture_to_buffer`, `copy_buffer_to_texture`
- Queue submit + `write_buffer`
- Unsupported-feature errors routed through `Device::on_uncaptured_error`
- Private CA for cert management (`CertAuthority` + `init-ca` subcommand)
- Cross-process / cross-machine over QUIC with CA-signed or self-signed certs
- iroh transport (NAT-traversed QUIC with relay + hole-punching, bidi streams + datagrams)

**Not yet:**
- Full render-path surface area (indirect draws, push constants, scissor/viewport/blend constants, `Queue::write_texture`)
- SPIR-V / GLSL shader sources (WGSL only today)
- Multi-client session scoping (currently one shared engine = one shared GPU device)
- mTLS client authentication (CA-signed client certs)
- Compressed texture formats and acceleration structures

**Out of scope:**
- Surface acquire / present (use video streaming for that)
- Raw window handles
- `wgpu::Device::create_shader_module_passthrough` (no local GPU on the client)

## License

MIT 
