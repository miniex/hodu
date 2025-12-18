//! Hodu plugin protocol types
//!
//! This crate defines the JSON-RPC protocol types and common data structures
//! shared between hodu-cli and plugins.

pub mod backend;
pub mod error;
pub mod rpc;
pub mod tensor;

// Re-export commonly used types
pub use backend::{current_host_triple, device_type, parse_device_id, BuildTarget, BuildTargetError, Device};
pub use error::{PluginError, PluginResult};
pub use rpc::*;
pub use tensor::{ParseDTypeError, PluginDType, TensorData, TensorDataError};

/// Plugin protocol version for compatibility checking (semver string)
pub const PLUGIN_VERSION: &str = env!("CARGO_PKG_VERSION");
