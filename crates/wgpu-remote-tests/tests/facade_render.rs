//! Render-to-texture round-trip via the facade.
//!
//! Renders a fullscreen triangle into a 4×4 RGBA8 texture using a constant
//! fragment color, copies the texture to a staging buffer, maps it for read,
//! and verifies every pixel matches the expected color (after stripping
//! row-alignment padding).
//!
//! Like `facade_compute.rs`, the protocol `Action`/`Response` enum is not
//! visible here — only the wgpu-shaped facade types.

use bytes::Bytes;
use wgpu_remote_client::{
    BufferUsages, Client, ColorAttachment, Instance, RenderPassDescriptor,
    descriptors::{
        BufferDescriptor, FragmentState, PipelineLayoutDescriptor, RenderPipelineDescriptor,
        ShaderModuleDescriptor, ShaderSource, TextureDescriptor, TextureViewDescriptor,
        VertexState,
    },
};
use wgpu_remote_server::{Engine, run_connection};
use wgpu_remote_tests::pair;
use wgpu_types::{
    Color, ColorTargetState, ColorWrites, Extent3d, LoadOp, MultisampleState, Operations,
    Origin3d, PrimitiveState, StoreOp, TexelCopyBufferLayout, TextureAspect, TextureDimension,
    TextureFormat, TextureUsages,
};

const SHADER: &str = r#"
@vertex
fn vs_main(@builtin(vertex_index) in_vertex_index: u32) -> @builtin(position) vec4<f32> {
    // Standard fullscreen triangle: (-1,-1), (3,-1), (-1,3). The rasterizer
    // clips it to NDC [-1,1]×[-1,1], which fully covers our render target.
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    return vec4<f32>(positions[in_vertex_index], 0.0, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.5, 0.25, 1.0);
}
"#;

const TEX_W: u32 = 4;
const TEX_H: u32 = 4;
/// Bytes per row in the *unpadded* image (TEX_W * 4 channels).
const UNPADDED_BPR: u32 = TEX_W * 4;
/// `wgpu`'s `COPY_BYTES_PER_ROW_ALIGNMENT` is 256.
const PADDED_BPR: u32 = 256;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn facade_render_to_texture() -> anyhow::Result<()> {
    let (client_conn, server_conn) = pair();
    let engine = std::sync::Arc::new(Engine::new().await?);
    let server_handle = tokio::spawn(async move {
        run_connection(engine, server_conn).await.unwrap();
    });

    let instance = Instance::new(Client::new(client_conn));
    let adapter = instance.request_adapter().await?;
    let (device, queue) = adapter.request_device().await?;

    // 1. The render target.
    let texture = device
        .create_texture(&TextureDescriptor {
            label: Some("rt".into()),
            size: Extent3d {
                width: TEX_W,
                height: TEX_H,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
            view_formats: vec![],
        });
    let view = texture.create_view(TextureViewDescriptor::default());

    // 2. Staging buffer for readback. Has to allow PADDED_BPR per row.
    let staging_size = (PADDED_BPR * TEX_H) as u64;
    let staging = device
        .create_buffer(&BufferDescriptor {
            label: Some("staging".into()),
            size: staging_size,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

    // 3. Pipeline (no bind groups → no BGL needed; use empty pipeline layout).
    let shader = device
        .create_shader_module(ShaderModuleDescriptor {
            label: Some("solid".into()),
            source: ShaderSource::Wgsl(SHADER.into()),
        });
    let layout = device
        .create_pipeline_layout(PipelineLayoutDescriptor {
            label: Some("empty".into()),
            bind_group_layouts: vec![],
            push_constant_ranges: vec![],
        });
    let pipeline = device
        .create_render_pipeline(RenderPipelineDescriptor {
            label: Some("solid-pipe".into()),
            layout: Some(layout.id()),
            vertex: VertexState {
                module: shader.id(),
                entry_point: Some("vs_main".into()),
                constants: vec![],
                buffers: vec![],
            },
            primitive: PrimitiveState::default(),
            depth_stencil: None,
            multisample: MultisampleState::default(),
            fragment: Some(FragmentState {
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

    // 4. Encode the render pass + readback copy.
    let mut encoder = device.create_command_encoder(Some("render-cmds".into()));
    {
        let color_atts = [Some(ColorAttachment {
            view: &view,
            depth_slice: None,
            resolve_target: None,
            ops: Operations {
                load: LoadOp::Clear(Color {
                    r: 0.0,
                    g: 0.0,
                    b: 0.0,
                    a: 1.0,
                }),
                store: StoreOp::Store,
            },
        })];
        let mut pass = encoder.begin_render_pass(RenderPassDescriptor {
            label: Some("solid-pass".into()),
            color_attachments: &color_atts,
            depth_stencil_attachment: None,
        });
        pass.set_pipeline(&pipeline);
        pass.draw(0..3, 0..1);
    } // pass drops, ops flush into encoder

    encoder.copy_texture_to_buffer(
        &texture,
        0,
        Origin3d::ZERO,
        TextureAspect::All,
        &staging,
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(PADDED_BPR),
            rows_per_image: Some(TEX_H),
        },
        Extent3d {
            width: TEX_W,
            height: TEX_H,
            depth_or_array_layers: 1,
        },
    );
    queue.submit([encoder.finish()])?;

    // 5. Read back. Strip padding row-by-row.
    let raw: Bytes = staging.read_all().await?;
    let mut pixels = Vec::with_capacity((UNPADDED_BPR * TEX_H) as usize);
    for row in 0..TEX_H {
        let start = (row * PADDED_BPR) as usize;
        let end = start + UNPADDED_BPR as usize;
        pixels.extend_from_slice(&raw[start..end]);
    }

    // 6. Assert: every pixel should be (1.0, 0.5, 0.25, 1.0) → roughly
    //    (255, 128, 64, 255) in u8. Allow ±1 for any sRGB / rounding fuzz.
    let expected = (255u8, 128u8, 64u8, 255u8);
    for (i, px) in pixels.chunks_exact(4).enumerate() {
        let (r, g, b, a) = (px[0], px[1], px[2], px[3]);
        let close = |v: u8, t: u8| v.abs_diff(t) <= 1;
        assert!(
            close(r, expected.0)
                && close(g, expected.1)
                && close(b, expected.2)
                && close(a, expected.3),
            "pixel {i}: got ({r},{g},{b},{a}), expected ~{expected:?}"
        );
    }

    drop(staging);
    drop(view);
    drop(texture);
    drop(pipeline);
    drop(layout);
    drop(shader);
    drop(queue);
    drop(device);
    drop(adapter);
    drop(instance);
    let _ = tokio::time::timeout(std::time::Duration::from_secs(2), server_handle).await;
    Ok(())
}
