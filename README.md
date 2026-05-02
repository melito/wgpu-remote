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

# Terminal 1: start the server (binds 0.0.0.0:4433 by default)
./target/release/wgpu-remote-server \
    --bind 127.0.0.1:4433 \
    --cert-out /tmp/wgpu-remote-cert.der

# Terminal 2: ping it
./target/release/wgpu-remote-cli ping \
    --server 127.0.0.1:4433 \
    --cert /tmp/wgpu-remote-cert.der

# Terminal 2: run a real compute workload
./target/release/wgpu-remote-cli compute-double \
    --server 127.0.0.1:4433 \
    --cert /tmp/wgpu-remote-cert.der \
    --count 16

# Terminal 2: render an actual image
./target/release/wgpu-remote-cli render-checkerboard \
    --server 127.0.0.1:4433 \
    --cert /tmp/wgpu-remote-cert.der \
    --width 512 --height 512 --tile 64 \
    --output /tmp/checker.ppm
open /tmp/checker.ppm   # macOS; xdg-open elsewhere
```

`compute-double` uploads `[1..=N]` to a storage buffer, dispatches a WGSL compute shader that doubles each value, copies the result to a staging buffer, maps it for read, and prints input vs. output.

`render-checkerboard` allocates a render-target texture, runs a fragment shader that paints a checkerboard pattern, copies the texture to a staging buffer, reads it back, and writes a PPM image you can open in any viewer.

Every step crosses the QUIC connection. For multi-machine use, copy `/tmp/wgpu-remote-cert.der` to the client side; it's regenerated on every server start.

## Architecture

```
┌────────────── client machine ──────────────┐
│  user code                                  │
│      │                                      │
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
| `wgpu-remote-transport`        | `Transport` + `Connection` traits. quinn-based QUIC implementation. |
| `wgpu-remote-transport-iroh`   | iroh adapter (stub — v1.1). |
| `wgpu-remote-server`           | Replay engine + server binary. |
| `wgpu-remote-client`           | Low-level `Client` + wgpu-shaped facade (`Instance`, `Device`, etc.). |
| `wgpu-remote-cli`              | Demo / smoke-test CLI. |
| `wgpu-remote-tests`            | End-to-end tests. |

## Using the facade

```rust
use wgpu_remote_client::prelude::quic::*;
use rustls::pki_types::CertificateDer;

let cert = CertificateDer::from(std::fs::read("server-cert.der")?);
let endpoint = QuicEndpoint::client(cert)?;
let connection = endpoint.connect(addr, "localhost").await?;

let instance = Instance::new(Client::new(connection));
let adapter  = instance.request_adapter().await?;
let (device, queue) = adapter.request_device().await?;

let buffer = device.create_buffer(&BufferDescriptor { /* ... */ });
queue.write_buffer(&buffer, 0, bytes);

let mut encoder = device.create_command_encoder(None);
{
    let mut pass = encoder.begin_compute_pass(None);
    pass.set_pipeline(&pipeline);
    pass.set_bind_group(0, &bind_group, &[]);
    pass.dispatch_workgroups(64, 1, 1);
}
queue.submit([encoder.finish()])?;

let bytes = readback_buffer.read_all().await?;
```

The facade is *similar* to wgpu but not yet a drop-in replacement:

- `request_adapter` and `request_device` are `async` (they make a real round-trip).
- Buffer readback uses `read_range(..)` / `read_all()` instead of
  `slice(..).map_async(..) + get_mapped_range`.
- The full-strength API uses generic types (`Buffer<C>`, `Device<C>`, …); the
  preludes [`prelude::quic`] and `wgpu_remote_tests::prelude::in_memory`
  hide the generic for the common transports.

The remaining gap to a full drop-in is wgpu's `custom` cargo feature integration — see *Roadmap*.

## Status

**Working today**:
- Buffers (create, write, copy, map for read)
- Textures, texture views, samplers
- Compute pipelines (WGSL shaders) + compute pass dispatch
- Render pipelines + render-to-texture (color attachments, load/store ops)
- Bind group layouts + bind groups
- Pipeline layouts (compute and render)
- Command encoder + compute pass + render pass recording
- `copy_buffer_to_buffer`, `copy_texture_to_buffer`, `copy_buffer_to_texture`
- Queue submit
- Cross-process / cross-machine over QUIC with self-signed cert pinning

**Not yet** (v1.2+):
- SPIR-V / GLSL shader sources (WGSL only)
- Multi-client session scoping (currently one shared engine = one shared GPU device)
- iroh transport (NodeId / hole-punch / relay)
- Real PKI flow for cert provisioning
- Compressed texture formats and acceleration structures
- True drop-in for stock wgpu apps via wgpu's `custom` cargo feature

**Probably never** (out of scope unless someone needs it):
- Surface acquire / present (use video streaming for that)
- Raw window handles

## Tests

```bash
cargo test --workspace
```

Notable end-to-end tests:

| Test                                  | What it proves                                                                                                                       |
|---------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------|
| `wgpu-remote-protocol::*`             | Wire format roundtrips.                                                                                                              |
| `engine_buffer / engine_compute`      | Engine dispatches against a real GPU.                                                                                                |
| `quic_compute`                        | Same workload over real QUIC, in-process.                                                                                            |
| `spawn_binary::binary_compute_double` | Spawns the actual server binary, runs the workload over a real socket — proves the architecture works across an OS process boundary. |
| `facade_compute_double`               | The compute workload using only wgpu-shaped facade types — no `Action` enum visible.                                                 |
| `facade_render_to_texture`            | Same shape but for the render path: fragment shader → texture → readback → per-pixel assertion.                                      |

## Roadmap (rough)

1. wgpu's `custom` cargo feature integration — true drop-in for existing wgpu apps.
2. iroh transport — laptop ↔ home-GPU NAT-traversed scenario.
3. Per-connection session scoping in the server (so multiple clients don't share an ID namespace).
4. Vertex/index buffer ergonomics in the facade (typed wrappers, mesh helpers).
5. SPIR-V / GLSL shader source variants on the wire.

## License

MIT OR Apache-2.0
