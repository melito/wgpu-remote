//! Drop-in coverage for the wgpu API surface a real renderer exercises but a
//! synthetic compute/checkerboard workload doesn't. Each of these was an
//! `unimplemented!()` a headless krad render hit, in order:
//!
//!   1. `Queue::write_texture`                    — upload pixels to a texture
//!   2. `CommandEncoder::copy_texture_to_buffer`  — read a texture back
//!   3. `mapped_at_creation` (write-mapped buffers) — `create_buffer_init`
//!   4. `RenderPipeline::get_bind_group_layout`   — derive a bind-group layout
//!
//! All drive the public `wgpu` API through `wgpu_remote::install`, so a
//! regression here means the drop-in stopped honoring stock wgpu code.

use std::sync::Arc;

use wgpu_remote::server::{Engine, run_connection};
use wgpu_remote_tests::pair;

/// Boilerplate: an in-memory-connected engine + drop-in device/queue.
async fn device_queue() -> anyhow::Result<(
    wgpu::Device,
    wgpu::Queue,
    tokio::task::JoinHandle<()>,
)> {
    let (client_conn, server_conn) = pair();
    let engine = Arc::new(Engine::new().await?);
    let server = tokio::spawn(async move {
        run_connection(engine, server_conn).await.unwrap();
    });
    let instance = wgpu_remote::install(client_conn);
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await?;
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor::default())
        .await?;
    Ok((device, queue, server))
}

/// Map a buffer for read and return its bytes (async, no `poll(Wait)`).
async fn read_back(device_queue_buffer: &wgpu::Buffer) -> anyhow::Result<Vec<u8>> {
    let (tx, rx) = tokio::sync::oneshot::channel();
    device_queue_buffer
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
    rx.await??;
    let view = device_queue_buffer.slice(..).get_mapped_range();
    let bytes = view.to_vec();
    drop(view);
    device_queue_buffer.unmap();
    Ok(bytes)
}

/// write_texture uploads pixels; copy_texture_to_buffer reads them back. One
/// test covers both, plus the texture-side readback path a render finishes on.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn write_texture_then_copy_to_buffer() -> anyhow::Result<()> {
    let (device, queue, server) = device_queue().await?;

    const W: u32 = 2;
    const H: u32 = 2;
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("t"),
        size: wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });

    // 4 RGBA8 pixels, all distinct so a channel/row mixup would show.
    let pixels: [u8; 16] = [
        10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160,
    ];
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &pixels,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(W * 4),
            rows_per_image: Some(H),
        },
        wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
    );

    // Readback buffer wants 256-byte row alignment (COPY_BYTES_PER_ROW_ALIGNMENT).
    const PADDED_BPR: u32 = 256;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging"),
        size: (PADDED_BPR * H) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    enc.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(PADDED_BPR),
                rows_per_image: Some(H),
            },
        },
        wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
    );
    queue.submit(std::iter::once(enc.finish()));

    let raw = read_back(&staging).await?;
    // Strip row padding: each 8-byte row starts on a PADDED_BPR boundary.
    let mut got = Vec::new();
    for row in 0..H as usize {
        let base = row * PADDED_BPR as usize;
        got.extend_from_slice(&raw[base..base + (W * 4) as usize]);
    }
    assert_eq!(got, pixels, "texture bytes did not survive write→copy→readback");

    drop(server);
    Ok(())
}

/// A buffer born `mapped_at_creation: true`, filled via `get_mapped_range_mut`,
/// then unmapped — the `wgpu::util::create_buffer_init` shape. We copy it into a
/// MAP_READ staging and confirm the init bytes made it to the server.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn mapped_at_creation_round_trip() -> anyhow::Result<()> {
    let (device, queue, server) = device_queue().await?;

    let init: [u8; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];

    // Note: usage is COPY_SRC only — the drop-in adds COPY_DST itself so it can
    // ship the staged bytes on unmap. If that trick regressed, the server would
    // reject the write and the readback would be zeroes.
    let src = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("src"),
        size: init.len() as u64,
        usage: wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: true,
    });
    src.slice(..).get_mapped_range_mut().copy_from_slice(&init);
    src.unmap();

    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging"),
        size: init.len() as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    enc.copy_buffer_to_buffer(&src, 0, &staging, 0, Some(init.len() as u64));
    queue.submit(std::iter::once(enc.finish()));

    let got = read_back(&staging).await?;
    assert_eq!(got, init, "mapped_at_creation init bytes did not reach the server");

    drop(server);
    Ok(())
}

/// A compute-double where the bind group is built from the pipeline's derived
/// bind-group layout (`get_bind_group_layout(0)`) rather than an explicit BGL.
/// If the derive-and-store round-trip regressed, the bind group would reference
/// an unknown layout and the dispatch would produce wrong output.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn get_bind_group_layout_drives_compute() -> anyhow::Result<()> {
    const SHADER: &str = r#"
@group(0) @binding(0) var<storage, read_write> data: array<u32>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    data[gid.x] = data[gid.x] * 2u;
}
"#;
    let (device, queue, server) = device_queue().await?;

    let input: Vec<u32> = (1..=8).collect();
    let input_bytes: Vec<u8> = input.iter().flat_map(|n| n.to_ne_bytes()).collect();
    let size = input_bytes.len() as u64;

    let storage = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("storage"),
        size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    queue.write_buffer(&storage, 0, &input_bytes);

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("double"),
        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
    });
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("bgl"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("pl"),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("double"),
        layout: Some(&pl),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    // The feature under test: build the bind group from the pipeline's own
    // layout rather than `bgl`.
    let derived = pipeline.get_bind_group_layout(0);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bg"),
        layout: &derived,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: storage.as_entire_binding(),
        }],
    });

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(input.len() as u32, 1, 1);
    }
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    enc.copy_buffer_to_buffer(&storage, 0, &staging, 0, Some(size));
    queue.submit(std::iter::once(enc.finish()));

    let bytes = read_back(&staging).await?;
    let output: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|c| u32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let expected: Vec<u32> = input.iter().map(|n| n * 2).collect();
    assert_eq!(output, expected, "derived bind-group layout produced wrong output");

    drop(server);
    Ok(())
}
