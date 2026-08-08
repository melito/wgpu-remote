//! Length-delimited bincode framing.
//!
//! QUIC streams are byte streams — message boundaries aren't preserved. Each
//! frame is `[u32 BE length][bincode bytes]`. We cap frame size to limit
//! adversarial allocation; bump if a real workload trips it.

use bincode::serde::{decode_from_slice, encode_to_vec};
use serde::{Serialize, de::DeserializeOwned};
use thiserror::Error;

const MAX_FRAME_BYTES: usize = 64 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum CodecError {
    #[error("encode failed: {0}")]
    Encode(#[from] bincode::error::EncodeError),
    #[error("decode failed: {0}")]
    Decode(#[from] bincode::error::DecodeError),
    #[error("frame too large: {len} bytes (max {MAX_FRAME_BYTES})")]
    FrameTooLarge { len: usize },
    #[error("frame truncated: needed {needed}, got {got}")]
    Truncated { needed: usize, got: usize },
}

fn config() -> bincode::config::Configuration {
    bincode::config::standard()
}

/// Serialize `value` and prepend a 4-byte big-endian length.
pub fn encode_frame<T: Serialize>(value: &T) -> Result<Vec<u8>, CodecError> {
    let body = encode_to_vec(value, config())?;
    if body.len() > MAX_FRAME_BYTES {
        return Err(CodecError::FrameTooLarge { len: body.len() });
    }
    let mut out = Vec::with_capacity(4 + body.len());
    out.extend_from_slice(&(body.len() as u32).to_be_bytes());
    out.extend_from_slice(&body);
    Ok(out)
}

/// Decode one length-prefixed frame from the head of `buf`. Returns the value
/// and the number of bytes consumed. Returns `Truncated` if `buf` doesn't yet
/// hold a full frame — caller should read more and retry.
pub fn decode_frame<T: DeserializeOwned>(buf: &[u8]) -> Result<(T, usize), CodecError> {
    if buf.len() < 4 {
        return Err(CodecError::Truncated {
            needed: 4,
            got: buf.len(),
        });
    }
    let len = u32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]]) as usize;
    if len > MAX_FRAME_BYTES {
        return Err(CodecError::FrameTooLarge { len });
    }
    let total = 4 + len;
    if buf.len() < total {
        return Err(CodecError::Truncated {
            needed: total,
            got: buf.len(),
        });
    }
    let (value, _) = decode_from_slice(&buf[4..total], config())?;
    Ok((value, total))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::actions::{Action, Frame, RequestId};

    #[test]
    fn roundtrip_hello() {
        let frame = Frame {
            request_id: Some(RequestId(7)),
            action: Action::Hello {
                protocol_version: crate::protocol::PROTOCOL_VERSION,
            },
        };
        let bytes = encode_frame(&frame).unwrap();
        let (decoded, consumed): (Frame, usize) = decode_frame(&bytes).unwrap();
        assert_eq!(consumed, bytes.len());
        assert_eq!(decoded.request_id, Some(RequestId(7)));
        assert!(matches!(decoded.action, Action::Hello { .. }));
    }

    #[test]
    fn truncated_returns_error() {
        let frame = Frame {
            request_id: None,
            action: Action::Hello {
                protocol_version: 1,
            },
        };
        let bytes = encode_frame(&frame).unwrap();
        let err = decode_frame::<Frame>(&bytes[..bytes.len() - 1]).unwrap_err();
        assert!(matches!(err, CodecError::Truncated { .. }));
    }
}
