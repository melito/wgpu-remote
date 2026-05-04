//! End-to-end smoke checks for the wgpu drop-in. Each test verifies a
//! progressively larger slice of the dispatch plumbing.

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

const DOUBLE_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read_write> data: array<u32>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    data[gid.x] = data[gid.x] * 2u;
}
"#;

/// Resource-create path: build a complete compute pipeline through the wgpu
/// drop-in. Doesn't dispatch the pipeline (Queue::submit + map_async aren't
/// wired yet), but proves that every Device::create_* method translates
/// correctly and that resources downcasted via `as_custom` are recognized
/// by sibling create methods.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn build_compute_pipeline_through_wgpu_drop_in() -> anyhow::Result<()> {
    let (client_conn, server_conn) = pair();

    let engine = std::sync::Arc::new(Engine::new().await?);
    let server = tokio::spawn(async move {
        run_connection(engine, server_conn).await.unwrap();
    });

    let instance = wgpu_remote_wgpu::install(client_conn);
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await?;
    let (device, _queue) = adapter
        .request_device(&wgpu::DeviceDescriptor::default())
        .await?;

    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("storage"),
        size: 32,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("double"),
        source: wgpu::ShaderSource::Wgsl(DOUBLE_SHADER.into()),
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

    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bg"),
        layout: &bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &buffer,
                offset: 0,
                size: None,
            }),
        }],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("pl"),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });

    let _pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("double_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    // Drop everything before the server task is torn down. The facade ships
    // Action::Destroy fire-and-forget on drop; the in-memory transport
    // delivers them in order before the server task ends.
    drop(_pipeline);
    drop(pipeline_layout);
    drop(bg);
    drop(bgl);
    drop(shader);
    drop(buffer);
    drop(device);
    drop(adapter);
    drop(instance);

    drop(server);
    Ok(())
}

/// The milestone test for the drop-in: dispatch a compute shader through
/// stock `wgpu::*` types and verify it produced the expected output on the
/// remote GPU.
///
/// This mirrors `wgpu-remote-tests::tests::facade_compute::facade_compute_double`
/// but uses `wgpu_remote_wgpu::install` and the public `wgpu` API
/// throughout — the protocol's `Action` enum and the facade's typed
/// resource handles never appear in this test.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn wgpu_drop_in_compute_double() -> anyhow::Result<()> {
    let (client_conn, server_conn) = pair();

    let engine = std::sync::Arc::new(Engine::new().await?);
    let server = tokio::spawn(async move {
        run_connection(engine, server_conn).await.unwrap();
    });

    let instance = wgpu_remote_wgpu::install(client_conn);
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await?;
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor::default())
        .await?;

    let input: Vec<u32> = (1..=8).collect();
    let input_bytes: Vec<u8> = input
        .iter()
        .flat_map(|n| n.to_ne_bytes())
        .collect();
    let buffer_size = input_bytes.len() as u64;

    let storage = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("storage"),
        size: buffer_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    queue.write_buffer(&storage, 0, &input_bytes);

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("double"),
        source: wgpu::ShaderSource::Wgsl(DOUBLE_SHADER.into()),
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

    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bg"),
        layout: &bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &storage,
                offset: 0,
                size: None,
            }),
        }],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("pl"),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("double_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    // Record + dispatch the compute pass.
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("double_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(input.len() as u32, 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));

    // Read back via wgpu's map_async API. We need a separate readback
    // buffer because storage doesn't have MAP_READ — and wgpu's standard
    // read-back pattern is exactly this: copy_buffer_to_buffer into a
    // staging buffer, then map_async.
    //
    // Currently the wgpu drop-in's CommandEncoder doesn't implement
    // copy_buffer_to_buffer for cross-storage copies… wait, it does (we
    // just wired it). And the storage buffer can be MAP_READ if we
    // declare it that way. Let's just use storage directly with
    // MAP_READ + COPY_DST + STORAGE — note we already gave it
    // MAP_READ-incompatible flags above. Recreate.
    //
    // (For the milestone test we go simpler: read directly from a
    // single buffer that has both STORAGE and MAP_READ. wgpu actually
    // forbids that pairing as of WebGPU spec, but our remote server
    // (built on stock wgpu) will refuse. So we do the proper staging
    // dance.)

    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging"),
        size: buffer_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut readback_encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("readback"),
        });
    readback_encoder.copy_buffer_to_buffer(&storage, 0, &staging, 0, Some(buffer_size));
    queue.submit(std::iter::once(readback_encoder.finish()));

    // map_async + get_mapped_range round-trip. wgpu::Buffer::slice
    // returns a BufferSlice that has its own map_async helper, but the
    // dispatch path is the same.
    let (tx, rx) = tokio::sync::oneshot::channel::<Result<(), wgpu::BufferAsyncError>>();
    staging
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
    rx.await??;

    let view = staging.slice(..).get_mapped_range();
    let output: Vec<u32> = view
        .chunks_exact(4)
        .map(|c| u32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    drop(view);
    staging.unmap();

    let expected: Vec<u32> = input.iter().map(|n| n * 2).collect();
    assert_eq!(output, expected);

    drop(server);
    Ok(())
}
