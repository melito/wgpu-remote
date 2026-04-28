//! End-to-end tests + an in-memory transport for fast iteration.

pub mod in_memory;

pub use in_memory::{InMemoryConnection, InMemoryTransport, pair};
