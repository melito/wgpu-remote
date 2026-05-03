//! Translation between `wgpu`'s public descriptor types and the wire-format
//! descriptors in [`wgpu_remote_protocol::descriptors`].
//!
//! Each `wgpu::FooDescriptor<'_>` has the same shape as its protocol mirror,
//! modulo two transformations:
//!
//! 1. Borrowed labels (`Label<'a> = Option<&'a str>`) become owned
//!    `Option<String>`. `wgpu_types` already exposes its descriptors as
//!    generic over the label type with a `map_label` helper, which makes
//!    this a one-liner for descriptors that don't reference other resources.
//! 2. References to resource handles (`&BindGroupLayout`, `&Buffer`, …)
//!    become protocol IDs, extracted via wgpu's public
//!    [`wgpu::BindGroupLayout::as_custom`] etc. — this only succeeds for
//!    resources we ourselves created. A foreign resource (e.g., one minted
//!    by a different wgpu backend mixed in) returns
//!    [`TranslateError::ForeignResource`].
//!
//! Errors here are programming-time, not runtime — they signal that the
//! caller passed a descriptor we can't faithfully ship across the wire.
//! The dispatch layer routes them to `Device::on_uncaptured_error`.

use wgpu_remote_protocol::descriptors::{
    BindGroupDescriptor as ProtoBindGroupDescriptor,
    BindGroupEntry as ProtoBindGroupEntry,
    BindGroupLayoutDescriptor as ProtoBindGroupLayoutDescriptor,
    BindingResource as ProtoBindingResource, BufferBinding as ProtoBufferBinding,
    BufferDescriptor as ProtoBufferDescriptor,
    ComputePipelineDescriptor as ProtoComputePipelineDescriptor,
    FragmentState as ProtoFragmentState,
    PipelineLayoutDescriptor as ProtoPipelineLayoutDescriptor,
    RenderPipelineDescriptor as ProtoRenderPipelineDescriptor,
    SamplerDescriptor as ProtoSamplerDescriptor,
    ShaderModuleDescriptor as ProtoShaderModuleDescriptor,
    ShaderSource as ProtoShaderSource,
    TextureDescriptor as ProtoTextureDescriptor,
    TextureViewDescriptor as ProtoTextureViewDescriptor,
    VertexBufferLayout as ProtoVertexBufferLayout,
    VertexState as ProtoVertexState,
};
use wgpu_remote_protocol::ids::{
    BindGroupLayoutId, BufferId, PipelineLayoutId, SamplerId, ShaderModuleId, TextureViewId,
};
use wgpu_remote_transport::Connection;

use crate::dispatch::{
    BindGroupLayout, Buffer, PipelineLayout, Sampler, ShaderModule, TextureView,
};

/// Errors raised when translating a wgpu descriptor to the protocol's wire
/// form. Always indicates a programming error or a feature the remote
/// backend doesn't implement — never a transport problem.
#[derive(Debug)]
pub(crate) enum TranslateError {
    /// A `&BindGroupLayout` / `&Buffer` / etc. inside the descriptor wasn't
    /// produced by this backend. wgpu allows mixing backends in theory; this
    /// crate doesn't.
    ForeignResource(&'static str),
}

impl std::fmt::Display for TranslateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TranslateError::ForeignResource(kind) => write!(
                f,
                "{kind} reference came from a different wgpu backend; \
                 wgpu-remote-wgpu can only see resources it created"
            ),
        }
    }
}

impl std::error::Error for TranslateError {}

// -- Owned-label helpers -----------------------------------------------------

fn own_label(label: wgpu::Label<'_>) -> Option<String> {
    label.map(str::to_owned)
}

// -- ID extraction from wgpu's public types ----------------------------------
//
// `wgpu::Foo::as_custom::<T>() -> Option<&T>` is the public path for
// extracting a custom-backend-specific impl out of a wgpu wrapper. Using
// `Dispatch*::downcast` directly would also work but bypasses the public
// API contract.

pub(crate) fn buffer_id<C: Connection + Clone + 'static>(
    buf: &wgpu::Buffer,
) -> Result<BufferId, TranslateError> {
    buf.as_custom::<Buffer<C>>()
        .map(|b| b.id())
        .ok_or(TranslateError::ForeignResource("buffer"))
}

pub(crate) fn sampler_id<C: Connection + Clone + 'static>(
    s: &wgpu::Sampler,
) -> Result<SamplerId, TranslateError> {
    s.as_custom::<Sampler<C>>()
        .map(|s| s.id())
        .ok_or(TranslateError::ForeignResource("sampler"))
}

pub(crate) fn texture_view_id<C: Connection + Clone + 'static>(
    v: &wgpu::TextureView,
) -> Result<TextureViewId, TranslateError> {
    v.as_custom::<TextureView<C>>()
        .map(|v| v.id())
        .ok_or(TranslateError::ForeignResource("texture view"))
}

pub(crate) fn shader_module_id<C: Connection + Clone + 'static>(
    m: &wgpu::ShaderModule,
) -> Result<ShaderModuleId, TranslateError> {
    m.as_custom::<ShaderModule<C>>()
        .map(|m| m.id())
        .ok_or(TranslateError::ForeignResource("shader module"))
}

pub(crate) fn bind_group_layout_id<C: Connection + Clone + 'static>(
    l: &wgpu::BindGroupLayout,
) -> Result<BindGroupLayoutId, TranslateError> {
    l.as_custom::<BindGroupLayout<C>>()
        .map(|l| l.id())
        .ok_or(TranslateError::ForeignResource("bind group layout"))
}

pub(crate) fn pipeline_layout_id<C: Connection + Clone + 'static>(
    l: &wgpu::PipelineLayout,
) -> Result<PipelineLayoutId, TranslateError> {
    l.as_custom::<PipelineLayout<C>>()
        .map(|l| l.id())
        .ok_or(TranslateError::ForeignResource("pipeline layout"))
}

// -- Leaf descriptors (no resource references) -------------------------------

pub(crate) fn buffer_descriptor(desc: &wgpu::BufferDescriptor<'_>) -> ProtoBufferDescriptor {
    desc.map_label(|l| own_label(*l))
}

pub(crate) fn texture_descriptor(desc: &wgpu::TextureDescriptor<'_>) -> ProtoTextureDescriptor {
    desc.map_label_and_view_formats(|l| own_label(*l), |formats| formats.to_vec())
}

pub(crate) fn sampler_descriptor(desc: &wgpu::SamplerDescriptor<'_>) -> ProtoSamplerDescriptor {
    // wgpu_types::SamplerDescriptor has no `map_label` helper (unlike its
    // sibling descriptors), so this one is hand-mirrored.
    ProtoSamplerDescriptor {
        label: own_label(desc.label),
        address_mode_u: desc.address_mode_u,
        address_mode_v: desc.address_mode_v,
        address_mode_w: desc.address_mode_w,
        mag_filter: desc.mag_filter,
        min_filter: desc.min_filter,
        mipmap_filter: desc.mipmap_filter,
        lod_min_clamp: desc.lod_min_clamp,
        lod_max_clamp: desc.lod_max_clamp,
        compare: desc.compare,
        anisotropy_clamp: desc.anisotropy_clamp,
        border_color: desc.border_color,
    }
}

pub(crate) fn texture_view_descriptor(
    desc: &wgpu::TextureViewDescriptor<'_>,
) -> ProtoTextureViewDescriptor {
    ProtoTextureViewDescriptor {
        label: own_label(desc.label),
        format: desc.format,
        dimension: desc.dimension,
        usage: desc.usage,
        aspect: desc.aspect,
        base_mip_level: desc.base_mip_level,
        mip_level_count: desc.mip_level_count,
        base_array_layer: desc.base_array_layer,
        array_layer_count: desc.array_layer_count,
    }
}

pub(crate) fn bind_group_layout_descriptor(
    desc: &wgpu::BindGroupLayoutDescriptor<'_>,
) -> ProtoBindGroupLayoutDescriptor {
    ProtoBindGroupLayoutDescriptor {
        label: own_label(desc.label),
        entries: desc.entries.to_vec(),
    }
}

pub(crate) fn shader_module_descriptor(
    desc: &wgpu::ShaderModuleDescriptor<'_>,
) -> ProtoShaderModuleDescriptor {
    // `wgpu::ShaderSource` is `#[non_exhaustive]` and gates its non-WGSL
    // variants behind feature flags (`spirv`, `glsl`, `naga-ir`). With the
    // default `wgpu` feature set the only reachable variant is `Wgsl`.
    // If a downstream project enables those features the wildcard arm
    // panics with a clear message — better than silently shipping an
    // empty WGSL string. (The protocol itself supports SPIR-V and GLSL;
    // wiring those through requires this crate also opt into the matching
    // wgpu features, which is a deliberate future change.)
    let source = match &desc.source {
        wgpu::ShaderSource::Wgsl(s) => ProtoShaderSource::Wgsl(s.to_string()),
        other => panic!(
            "wgpu-remote-wgpu only supports `wgpu::ShaderSource::Wgsl` in this build; \
             got {other:?}. Enable matching features on this crate to bridge SPIR-V / GLSL."
        ),
    };
    ProtoShaderModuleDescriptor {
        label: own_label(desc.label),
        source,
    }
}

// -- Descriptors with resource references ------------------------------------

pub(crate) fn bind_group_descriptor<C: Connection + Clone + 'static>(
    desc: &wgpu::BindGroupDescriptor<'_>,
) -> Result<ProtoBindGroupDescriptor, TranslateError> {
    let layout = bind_group_layout_id::<C>(desc.layout)?;
    let entries = desc
        .entries
        .iter()
        .map(|e| {
            Ok(ProtoBindGroupEntry {
                binding: e.binding,
                resource: binding_resource::<C>(&e.resource)?,
            })
        })
        .collect::<Result<Vec<_>, TranslateError>>()?;
    Ok(ProtoBindGroupDescriptor {
        label: own_label(desc.label),
        layout,
        entries,
    })
}

fn binding_resource<C: Connection + Clone + 'static>(
    r: &wgpu::BindingResource<'_>,
) -> Result<ProtoBindingResource, TranslateError> {
    match r {
        wgpu::BindingResource::Buffer(b) => Ok(ProtoBindingResource::Buffer {
            buffer: buffer_id::<C>(b.buffer)?,
            offset: b.offset,
            size: b.size,
        }),
        wgpu::BindingResource::BufferArray(arr) => {
            let bs = arr
                .iter()
                .map(|b| {
                    Ok(ProtoBufferBinding {
                        buffer: buffer_id::<C>(b.buffer)?,
                        offset: b.offset,
                        size: b.size,
                    })
                })
                .collect::<Result<Vec<_>, TranslateError>>()?;
            Ok(ProtoBindingResource::BufferArray(bs))
        }
        wgpu::BindingResource::Sampler(s) => {
            Ok(ProtoBindingResource::Sampler(sampler_id::<C>(s)?))
        }
        wgpu::BindingResource::SamplerArray(arr) => {
            let ids = arr
                .iter()
                .map(|s| sampler_id::<C>(s))
                .collect::<Result<Vec<_>, TranslateError>>()?;
            Ok(ProtoBindingResource::SamplerArray(ids))
        }
        wgpu::BindingResource::TextureView(v) => {
            Ok(ProtoBindingResource::TextureView(texture_view_id::<C>(v)?))
        }
        wgpu::BindingResource::TextureViewArray(arr) => {
            let ids = arr
                .iter()
                .map(|v| texture_view_id::<C>(v))
                .collect::<Result<Vec<_>, TranslateError>>()?;
            Ok(ProtoBindingResource::TextureViewArray(ids))
        }
        // wgpu::BindingResource is `#[non_exhaustive]`. Any new variant the
        // upstream adds is a feature we haven't bridged yet.
        _ => Err(TranslateError::ForeignResource(
            "unknown wgpu::BindingResource variant",
        )),
    }
}

pub(crate) fn pipeline_layout_descriptor<C: Connection + Clone + 'static>(
    desc: &wgpu::PipelineLayoutDescriptor<'_>,
) -> Result<ProtoPipelineLayoutDescriptor, TranslateError> {
    let bind_group_layouts = desc
        .bind_group_layouts
        .iter()
        .map(|l| bind_group_layout_id::<C>(l))
        .collect::<Result<Vec<_>, TranslateError>>()?;
    Ok(ProtoPipelineLayoutDescriptor {
        label: own_label(desc.label),
        bind_group_layouts,
        push_constant_ranges: desc.push_constant_ranges.to_vec(),
    })
}

pub(crate) fn vertex_state<C: Connection + Clone + 'static>(
    v: &wgpu::VertexState<'_>,
) -> Result<ProtoVertexState, TranslateError> {
    Ok(ProtoVertexState {
        module: shader_module_id::<C>(v.module)?,
        entry_point: v.entry_point.map(str::to_owned),
        constants: v
            .compilation_options
            .constants
            .iter()
            .map(|(k, val)| (k.to_string(), *val))
            .collect(),
        buffers: v
            .buffers
            .iter()
            .map(|b| ProtoVertexBufferLayout {
                array_stride: b.array_stride,
                step_mode: b.step_mode,
                attributes: b.attributes.to_vec(),
            })
            .collect(),
    })
}

pub(crate) fn fragment_state<C: Connection + Clone + 'static>(
    f: &wgpu::FragmentState<'_>,
) -> Result<ProtoFragmentState, TranslateError> {
    Ok(ProtoFragmentState {
        module: shader_module_id::<C>(f.module)?,
        entry_point: f.entry_point.map(str::to_owned),
        constants: f
            .compilation_options
            .constants
            .iter()
            .map(|(k, v)| (k.to_string(), *v))
            .collect(),
        targets: f.targets.to_vec(),
    })
}

pub(crate) fn render_pipeline_descriptor<C: Connection + Clone + 'static>(
    desc: &wgpu::RenderPipelineDescriptor<'_>,
) -> Result<ProtoRenderPipelineDescriptor, TranslateError> {
    let layout = match desc.layout {
        Some(l) => Some(pipeline_layout_id::<C>(l)?),
        None => None,
    };
    Ok(ProtoRenderPipelineDescriptor {
        label: own_label(desc.label),
        layout,
        vertex: vertex_state::<C>(&desc.vertex)?,
        primitive: desc.primitive,
        depth_stencil: desc.depth_stencil.clone(),
        multisample: desc.multisample,
        fragment: match &desc.fragment {
            Some(f) => Some(fragment_state::<C>(f)?),
            None => None,
        },
        // wgpu uses Option<NonZeroU32>; the protocol stores Option<u32>.
        multiview: desc.multiview.map(|n| n.get()),
    })
}

pub(crate) fn compute_pipeline_descriptor<C: Connection + Clone + 'static>(
    desc: &wgpu::ComputePipelineDescriptor<'_>,
) -> Result<ProtoComputePipelineDescriptor, TranslateError> {
    let layout = match desc.layout {
        Some(l) => Some(pipeline_layout_id::<C>(l)?),
        None => None,
    };
    Ok(ProtoComputePipelineDescriptor {
        label: own_label(desc.label),
        layout,
        module: shader_module_id::<C>(desc.module)?,
        entry_point: desc.entry_point.map(str::to_owned),
        constants: desc
            .compilation_options
            .constants
            .iter()
            .map(|(k, v)| (k.to_string(), *v))
            .collect(),
    })
}
