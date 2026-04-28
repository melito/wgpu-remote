//! Typed resource handles. The client mints monotonic `u64`s; the server
//! keeps a table per resource type. Newtypes prevent cross-type confusion
//! (passing a `BufferId` where a `TextureId` was expected).

use serde::{Deserialize, Serialize};

macro_rules! id_type {
    ($name:ident) => {
        #[derive(
            Copy, Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd, Serialize, Deserialize,
        )]
        #[repr(transparent)]
        pub struct $name(pub u64);

        impl $name {
            pub const fn new(raw: u64) -> Self {
                Self(raw)
            }
            pub const fn raw(self) -> u64 {
                self.0
            }
        }
    };
}

id_type!(DeviceId);
id_type!(QueueId);
id_type!(BufferId);
id_type!(TextureId);
id_type!(TextureViewId);
id_type!(SamplerId);
id_type!(ShaderModuleId);
id_type!(BindGroupLayoutId);
id_type!(BindGroupId);
id_type!(PipelineLayoutId);
id_type!(RenderPipelineId);
id_type!(ComputePipelineId);
id_type!(CommandBufferId);

/// Tagged union over every resource ID. Used for `Destroy` and any code
/// that needs to refer to "some resource" without caring which kind.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum ResourceId {
    Device(DeviceId),
    Queue(QueueId),
    Buffer(BufferId),
    Texture(TextureId),
    TextureView(TextureViewId),
    Sampler(SamplerId),
    ShaderModule(ShaderModuleId),
    BindGroupLayout(BindGroupLayoutId),
    BindGroup(BindGroupId),
    PipelineLayout(PipelineLayoutId),
    RenderPipeline(RenderPipelineId),
    ComputePipeline(ComputePipelineId),
    CommandBuffer(CommandBufferId),
}
