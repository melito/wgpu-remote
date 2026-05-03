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
