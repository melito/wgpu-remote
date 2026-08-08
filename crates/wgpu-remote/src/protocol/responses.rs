//! Server → client messages.

use bytes::Bytes;
use serde::{Deserialize, Serialize};

use crate::protocol::actions::RequestId;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ResponseFrame {
    pub request_id: RequestId,
    pub response: Response,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Response {
    /// Reply to [`Action::Hello`](crate::protocol::Action::Hello). If versions disagree
    /// the connection should be closed.
    HelloAck {
        protocol_version: u32,
    },

    /// Resource creation / destruction / submit completed without error.
    Ok,

    /// Reply to [`Action::MapBufferForRead`](crate::protocol::Action::MapBufferForRead).
    BufferData {
        data: Bytes,
    },

    /// A submission has completed on the GPU (fence reached value).
    SubmissionComplete {
        fence_value: u64,
    },

    Error {
        code: ErrorCode,
        message: String,
    },
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum ErrorCode {
    ProtocolVersionMismatch,
    UnknownResource,
    InvalidArgument,
    OutOfMemory,
    DeviceLost,
    Internal,
}
