//! Plugin system for Hodu CLI
//!
//! This module re-exports the plugin runtime and provides CLI-specific
//! notification handling.

// Re-export from hodu_plugin_runtime
pub use hodu_plugin_runtime::backend;
pub use hodu_plugin_runtime::format;
pub use hodu_plugin_runtime::{
    detect_plugin_type, CancellationHandle, ClientError, DetectedPluginType, PluginCapabilities, PluginClient,
    PluginDetectError, PluginEntry, PluginRegistry, PluginSource, PluginType, RegistryError, DEFAULT_TIMEOUT,
};

mod process;

pub use process::*;

// Plugin name prefixes
pub const BACKEND_PREFIX: &str = "hodu-backend-";
pub const FORMAT_PREFIX: &str = "hodu-format-";

/// Generate backend plugin name from short name
pub fn backend_plugin_name(name: &str) -> String {
    format!("{}{}", BACKEND_PREFIX, name)
}

/// Generate format plugin name from short name
pub fn format_plugin_name(name: &str) -> String {
    format!("{}{}", FORMAT_PREFIX, name)
}

/// Load plugin registry from default path
pub fn load_registry() -> Result<PluginRegistry, Box<dyn std::error::Error>> {
    let registry_path = PluginRegistry::default_path()?;
    let registry = PluginRegistry::load(&registry_path)?;
    Ok(registry)
}

/// Load mutable plugin registry from default path (returns path for saving)
pub fn load_registry_mut() -> Result<(PluginRegistry, std::path::PathBuf), Box<dyn std::error::Error>> {
    let registry_path = PluginRegistry::default_path()?;
    let registry = PluginRegistry::load(&registry_path)?;
    Ok((registry, registry_path))
}

/// Get the registry path without loading (for lock-first patterns)
pub fn get_registry_path() -> Result<std::path::PathBuf, Box<dyn std::error::Error>> {
    Ok(PluginRegistry::default_path()?)
}
