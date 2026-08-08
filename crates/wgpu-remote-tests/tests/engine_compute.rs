//! End-to-end compute test: write a buffer of u32s, dispatch a compute shader
//! that doubles each value, copy to a readback buffer, map and verify.
//!
//! Exercises the full action surface for compute: shader module → bind group
//! layout → bind group → pipeline layout → compute pipeline → submit
//! (containing a copy op + a compute pass) → map for read.

use std::num::NonZeroU64;

use bytes::Bytes;
use wgpu_remote::protocol::{
    Response,
    actions::{Action, Frame, RequestId},
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
use wgpu_remote::server::Engine;
use wgpu_types::{
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferUsages, ShaderStages,
};

const DOUBLE_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read_write> data: array<u32>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    data[gid.x] = data[gid.x] * 2u;
}
"#;

async fn ok(engine: &Engine, rid: u64, action: Action) -> Response {
    engine
        .dispatch(Frame {
            request_id: Some(RequestId(rid)),
            action,
        })
        .await
        .expect("response expected")
        .response
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn compute_double_round_trip() -> anyhow::Result<()> {
    let engine = Engine::new().await?;

    // Resource IDs — chosen by hand for clarity; in production these come
    // from the client's monotonic ID minter.
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

    // 1. Storage buffer (compute target) + staging buffer (for readback).
    assert!(matches!(
        ok(
            &engine,
            1,
            Action::CreateBuffer {
                id: storage_id,
                desc: BufferDescriptor {
                    label: Some("storage".into()),
                    size: buffer_size,
                    usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                },
            }
        )
        .await,
        Response::Ok
    ));

    assert!(matches!(
        ok(
            &engine,
            2,
            Action::CreateBuffer {
                id: staging_id,
                desc: BufferDescriptor {
                    label: Some("staging".into()),
                    size: buffer_size,
                    usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                },
            }
        )
        .await,
        Response::Ok
    ));

    // 2. Upload input bytes into storage.
    assert!(matches!(
        ok(
            &engine,
            3,
            Action::WriteBuffer {
                buffer: storage_id,
                offset: 0,
                data: Bytes::from(input_bytes.clone()),
            }
        )
        .await,
        Response::Ok
    ));

    // 3. Shader module + bind group layout + bind group + pipeline layout +
    //    pipeline.
    assert!(matches!(
        ok(
            &engine,
            4,
            Action::CreateShaderModule {
                id: shader_id,
                desc: ShaderModuleDescriptor {
                    label: Some("double".into()),
                    source: ShaderSource::Wgsl(DOUBLE_SHADER.into()),
                },
            }
        )
        .await,
        Response::Ok
    ));

    assert!(matches!(
        ok(
            &engine,
            5,
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
            }
        )
        .await,
        Response::Ok
    ));

    assert!(matches!(
        ok(
            &engine,
            6,
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
            }
        )
        .await,
        Response::Ok
    ));

    assert!(matches!(
        ok(
            &engine,
            7,
            Action::CreatePipelineLayout {
                id: layout_id,
                desc: PipelineLayoutDescriptor {
                    label: Some("compute-layout".into()),
                    bind_group_layouts: vec![bgl_id],
                    push_constant_ranges: vec![],
                },
            }
        )
        .await,
        Response::Ok
    ));

    assert!(matches!(
        ok(
            &engine,
            8,
            Action::CreateComputePipeline {
                id: pipeline_id,
                desc: ComputePipelineDescriptor {
                    label: Some("double-pipeline".into()),
                    layout: Some(layout_id),
                    module: shader_id,
                    entry_point: Some("main".into()),
                    constants: vec![],
                },
            }
        )
        .await,
        Response::Ok
    ));

    // 4. Build the command buffer recording: dispatch the compute pass, then
    //    copy storage → staging.
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
    let recording_bytes = recording.encode()?;

    assert!(matches!(
        ok(
            &engine,
            9,
            Action::Submit {
                recordings: vec![recording_bytes],
            }
        )
        .await,
        Response::Ok
    ));

    // 5. Map the staging buffer and verify each input was doubled.
    let response = ok(
        &engine,
        10,
        Action::MapBufferForRead {
            buffer: staging_id,
            offset: 0,
            size: buffer_size,
        },
    )
    .await;
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

    Ok(())
}
