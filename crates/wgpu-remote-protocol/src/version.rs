//! Wire-format version. Bump on any `serde`-incompatible change to [`Action`]
//! or [`Response`]. The handshake exchanges this and refuses mismatches —
//! a hard failure here beats silent deserialization corruption.
//!
//! [`Action`]: crate::Action
//! [`Response`]: crate::Response

pub const PROTOCOL_VERSION: u32 = 1;
