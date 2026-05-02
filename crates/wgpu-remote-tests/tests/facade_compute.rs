//! Compute round-trip using only the wgpu-shaped facade types. The protocol
//! `Action`/`Response` enum should not appear in this test — that's the
//! whole point of the facade.

use std::num::NonZeroU64;

use bytes::Bytes;
use wgpu_remote_server::{Engine, run_connection};
use wgpu_remote_tests::prelude::in_memory::*;

const DOUBLE_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read_write> data: array<u32>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    data[gid.x] = data[gid.x] * 2u;
}
"#;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn facade_compute_double() -> anyhow::Result<()> {
    // In-memory transport so we don't need TLS for this test — the goal is
    // to validate the facade's ergonomics, not the network.
    let (client_conn, server_conn) = pair();

    // Server side.
    let engine = std::sync::Arc::new(Engine::new().await?);
    let server_handle = tokio::spawn(async move {
        run_connection(engine, server_conn).await.unwrap();
    });

    // Client side: this is the part a user writes.
    let instance = Instance::new(Client::new(client_conn));
    let adapter = instance.request_adapter().await?;
    let (device, queue) = adapter.request_device().await?;

    let input: Vec<u32> = (1..=8).collect();
    let input_bytes: Bytes = input.iter().flat_map(|n| n.to_ne_bytes()).collect::<Vec<u8>>().into();
    let buffer_size = input_bytes.len() as u64;

    let storage = device
        .create_buffer(&BufferDescriptor {
            label: Some("storage".into()),
            size: buffer_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
    let staging = device
        .create_buffer(&BufferDescriptor {
            label: Some("staging".into()),
            size: buffer_size,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

    queue.write_buffer(&storage, 0, input_bytes.clone());

    let shader = device
        .create_shader_module(ShaderModuleDescriptor {
            label: Some("double".into()),
            source: ShaderSource::Wgsl(DOUBLE_SHADER.into()),
        });

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
        });

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
        });

    let pipeline_layout = device
        .create_pipeline_layout(PipelineLayoutDescriptor {
            label: Some("compute-layout".into()),
            bind_group_layouts: vec![bgl.id()],
            push_constant_ranges: vec![],
        });

    let pipeline = device
        .create_compute_pipeline(ComputePipelineDescriptor {
            label: Some("double-pipeline".into()),
            layout: Some(pipeline_layout.id()),
            module: shader.id(),
            entry_point: Some("main".into()),
            constants: vec![],
        });

    // Encode: compute pass + copy storage→staging.
    let mut encoder = device.create_command_encoder(Some("double-cmds".into()));
    {
        let mut pass = encoder.begin_compute_pass(Some("double-pass".into()));
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(input.len() as u32, 1, 1);
    } // pass drops, ops flushed into encoder
    encoder.copy_buffer_to_buffer(&storage, 0, &staging, 0, buffer_size);
    let cb = encoder.finish();

    queue.submit([cb])?;

    // Read back.
    let bytes = staging.read_all().await?;
    let output: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|c| u32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let expected: Vec<u32> = input.iter().map(|n| n * 2).collect();
    assert_eq!(output, expected);

    // Tear down. Drop user-side handles to fire the destroys, then drop the
    // instance (which holds the Arc<Client>) so the connection ends.
    drop(staging);
    drop(storage);
    drop(bind_group);
    drop(bgl);
    drop(pipeline);
    drop(pipeline_layout);
    drop(shader);
    drop(queue);
    drop(device);
    drop(adapter);
    drop(instance);

    let _ = tokio::time::timeout(std::time::Duration::from_secs(2), server_handle).await;
    Ok(())
}
