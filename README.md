# wgpu-remote

Run [wgpu](https://github.com/gfx-rs/wgpu) workloads against a GPU on a different machine.

The client speaks a wgpu-shaped API; under it, every call gets serialized and sent to a server process on the remote host, which executes it on a real GPU and ships any results back. Useful when:

- you want to run heavy compute / rendering from a laptop without the GPU
- you're targeting a cloud GPU instance (4090, A100, …) without paying for the cloud machine to host your whole app
- you want a clean RPC seam under wgpu for testing or recording GPU traffic

This is a research prototype — see *Status* below for what's implemented.

## Quickstart

```bash
# Build everything
cargo build --release

# One-time: generate a private CA
./target/release/wgpu-remote-server init-ca \
    --ca-cert /tmp/wgpu-remote-ca.der \
    --ca-key  /tmp/wgpu-remote-ca.key

# Terminal 1: start the server (binds 0.0.0.0:4433 by default)
./target/release/wgpu-remote-server \
    --bind 127.0.0.1:4433 \
    --ca-cert /tmp/wgpu-remote-ca.der \
    --ca-key  /tmp/wgpu-remote-ca.key

# Terminal 2: ping it
./target/release/wgpu-remote-cli ping \
    --server 127.0.0.1:4433 \
    --ca-cert /tmp/wgpu-remote-ca.der

# Terminal 2: run a real compute workload
./target/release/wgpu-remote-cli compute-double \
    --server 127.0.0.1:4433 \
    --ca-cert /tmp/wgpu-remote-ca.der \
    --count 16

# Terminal 2: render an actual image
./target/release/wgpu-remote-cli render-checkerboard \
    --server 127.0.0.1:4433 \
    --ca-cert /tmp/wgpu-remote-ca.der \
    --width 512 --height 512 --tile 64 \
    --output /tmp/checker.ppm
open /tmp/checker.ppm   # macOS; xdg-open elsewhere
```

`compute-double` uploads `[1..=N]` to a storage buffer, dispatches a WGSL compute shader that doubles each value, copies the result to a staging buffer, maps it for read, and prints input vs. output.

`render-checkerboard` allocates a render-target texture, runs a fragment shader that paints a checkerboard pattern, copies the texture to a staging buffer, reads it back, and writes a PPM image you can open in any viewer.

Every step crosses the QUIC connection. For multi-machine use, copy the CA cert to the client side. Unlike the old self-signed flow, the CA cert is stable — server certs can rotate on every restart without redistributing trust anchors.

## Architecture

```
┌────────────── client machine ──────────────┐
│  user code                                  │
│      │                                      │
│      ▼                                      │
│  wgpu::Instance / Device / Queue / ...      │  ← real wgpu types
│      │  (via wgpu_remote_wgpu::install)     │
│      ▼                                      │
│  wgpu_remote_client::Instance / Device /    │
│  Buffer / Queue / CommandEncoder            │  ← wgpu-shaped facade
│      │                                      │
│      ▼                                      │
│  wgpu_remote_client::Client                 │  ← Action / Response RPC
│      │                                      │
│      ▼                                      │
│  wgpu_remote_transport::Connection          │  ← pluggable transport
│  ┌─────────────┬───────────────────────┐    │
│  │ quinn (QUIC)│ in-memory / iroh*     │    │
│  └─────────────┴───────────────────────┘    │
└────────────────│────────────────────────────┘
                 │  UDP + TLS (rustls)
┌────────────────│────────────────────────────┐
│                ▼                            │
│  wgpu-remote-server (binary)                │
│      │                                      │
│      ▼                                      │
│  wgpu_remote_server::Engine                 │  ← replay against real wgpu
│      │                                      │
│      ▼                                      │
│  wgpu::Instance → adapter → device → GPU    │
└─────────────────────────────────────────────┘
```

\* iroh adapter scaffolded; not yet implemented.

### Why not implement a wgpu-hal backend?

That's the natural-sounding approach — "just write a network backend for wgpu-hal" — but it has real downsides:

- **wgpu-hal is unstable.** The trait surface breaks on most wgpu releases.
- **Some hal calls don't translate over a network**: `map_buffer` returns a raw `*mut u8`; `create_surface` takes a window handle.
- **The surface is large** (~150–200 methods).

We use the seam Firefox uses for its content↔GPU process bridge: serialize at the level of typed actions on the user-facing surface (descriptors + command buffer recordings), not at the hal trait. Smaller, more stable, free serde via `wgpu-types`'s `serde` feature.

## Crates

| Crate                          | Purpose |
|--------------------------------|---------|
| `wgpu-remote-protocol`         | Wire format. Action/Response enums, typed IDs, descriptor mirrors, length-delimited bincode codec. No I/O. |
| `wgpu-remote-transport`        | `Transport` + `Connection` traits, quinn-based QUIC implementation, private CA (`pki` module). |
| `wgpu-remote-transport-iroh`   | iroh adapter (stub — v1.1). |
| `wgpu-remote-server`           | Replay engine + server binary. |
| `wgpu-remote-client`           | Low-level `Client` + wgpu-shaped facade (`Instance`, `Device`, etc.). |
| `wgpu-remote-wgpu`             | True drop-in for stock `wgpu`: `install(connection) -> wgpu::Instance`. |
| `wgpu-remote-cli`              | Demo / smoke-test CLI. |
| `wgpu-remote-tests`            | End-to-end tests. |

## Certificate management

The server uses a private CA to issue short-lived server certificates.
Clients pin the CA cert, so server certs can rotate freely.

```bash
# Generate a CA (once, keep the key secret)
wgpu-remote-server init-ca --ca-cert ca-cert.der --ca-key ca-key.der

# Start the server — it loads the CA and issues its own server cert
wgpu-remote-server --ca-cert ca-cert.der --ca-key ca-key.der

# If no CA files exist, the server auto-generates one on first start
wgpu-remote-server   # creates ./ca-cert.der + ./ca-key.der

# For quick testing, --self-signed skips the CA entirely (old v0 behavior)
wgpu-remote-server --self-signed --cert-out server-cert.der
```

The server cert includes `localhost` as a SAN by default. Add more with
`--san`:

```bash
wgpu-remote-server --ca-cert ca.der --ca-key ca.key \
    --bind 0.0.0.0:4433 \
    --san my-gpu-box.local \
    --san 192.168.1.42
```

For multi-machine use, copy `ca-cert.der` to the client machine. The CA
key never leaves the server.

## Using as a wgpu drop-in

Hand `wgpu_remote_wgpu::install(connection)` a transport handle and you get
back a real `wgpu::Instance`. From that point on it's stock wgpu — no new
types, no `<C>` generic, no facade-specific APIs. Built on wgpu 27's
`custom` cargo feature.

```rust
use rustls::pki_types::CertificateDer;
use wgpu_remote_transport::quic::QuicEndpoint;

let ca_cert = CertificateDer::from(std::fs::read("ca-cert.der")?);
let endpoint = QuicEndpoint::client(ca_cert)?;
let connection = endpoint.connect(addr, "localhost").await?;

let instance: wgpu::Instance = wgpu_remote_wgpu::install(connection);

// From here, every type is wgpu's own.
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

let (tx, rx) = tokio::sync::oneshot::channel();
readback.slice(..).map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
rx.await??;
let view = readback.slice(..).get_mapped_range();
// ... read view ...
drop(view);
readback.unmap();
```

Unsupported features route through `Device::on_uncaptured_error` rather
than panicking, so apps that probe `Adapter::features()` and degrade
gracefully behave correctly. Truly impossible operations (Surface,
debug capture, shader passthrough) panic with a clear message.

## Using the facade directly

If you want async creates, simpler readback (`read_range(..)` /
`read_all()`), or to skip the wgpu dispatch indirection, the
`wgpu-remote-client` facade exposes the same surface in a more direct
shape:

```rust
use wgpu_remote_client::prelude::quic::*;

let instance = Instance::new(Client::new(connection));
let adapter  = instance.request_adapter().await?;
let (device, queue) = adapter.request_device().await?;
let buffer = device.create_buffer(&BufferDescriptor { /* ... */ });
let bytes = readback_buffer.read_all().await?;
```

The facade carries a `<C: Connection>` generic at every type; the preludes
(`prelude::quic` and `wgpu_remote_tests::prelude::in_memory`) hide it for
the two shipping transports.

## Status

**Working today**:
- Drop-in for stock wgpu (compute path) via `wgpu_remote_wgpu::install`
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

**Not yet** (v1.2+):
- Drop-in for stock wgpu *render* path (encoder/render-pass surface area
  beyond what compute exercises: indirect draws, push constants,
  scissor/viewport/blend constants, `Queue::write_texture`)
- SPIR-V / GLSL shader sources (WGSL only)
- Multi-client session scoping (currently one shared engine = one shared GPU device)
- iroh transport (NodeId / hole-punch / relay)
- mTLS client authentication (CA-signed client certs)
- Compressed texture formats and acceleration structures

**Probably never** (out of scope unless someone needs it):
- Surface acquire / present (use video streaming for that)
- Raw window handles
- `wgpu::Device::create_shader_module_passthrough` (no local GPU on the client)

## Tests

```bash
cargo test --workspace
```

Notable end-to-end tests:

| Test                                                | What it proves                                                                                                                       |
|-----------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------|
| `wgpu-remote-protocol::*`                           | Wire format roundtrips.                                                                                                              |
| `engine_buffer / engine_compute`                    | Engine dispatches against a real GPU.                                                                                                |
| `quic_compute`                                      | Same workload over real QUIC, in-process.                                                                                            |
| `spawn_binary::binary_compute_double`               | Spawns the actual server binary, runs the workload over a real socket — proves the architecture works across an OS process boundary. |
| `spawn_binary::binary_handshake_ca`                 | Spawns the server in CA mode — proves the CA → server cert → client trust chain works across a process boundary.                    |
| `facade_compute_double`                             | The compute workload using only wgpu-shaped facade types — no `Action` enum visible.                                                 |
| `facade_render_to_texture`                          | Same shape but for the render path: fragment shader → texture → readback → per-pixel assertion.                                      |
| `wgpu_drop_in_compute_double`                       | The compute workload using only `wgpu::*` types via `wgpu_remote_wgpu::install` — the drop-in milestone.                             |
| `unsupported_feature_routes_through_error_handler`  | Calling an unsupported feature fires `on_uncaptured_error` instead of panicking.                                                     |

## Roadmap (rough)

1. Render-path completeness in the wgpu drop-in — indirect draws, push
   constants, scissor / viewport / blend constants, `Queue::write_texture`.
   Each is a small protocol addition + a small dispatch-side rewrite.
2. iroh transport — laptop ↔ home-GPU NAT-traversed scenario.
3. Per-connection session scoping in the server (so multiple clients don't share an ID namespace).
4. mTLS client authentication — CA-signed client certs for access control.
5. SPIR-V / GLSL shader source variants on the wire.

## License

MIT OR Apache-2.0
