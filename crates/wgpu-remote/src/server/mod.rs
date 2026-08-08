//! Replay server. Reads framed [`Action`](crate::protocol::Action)s,
//! looks up resources in an ID table, dispatches to a real `wgpu::Instance`,
//! and writes [`Response`](crate::protocol::Response)s back.

pub mod engine;
pub mod handler;
pub mod tables;

pub use engine::{Engine, EngineError};
pub use handler::{handle_stream, run_connection};
pub use tables::ResourceTables;
