//! Plugin process management for CLI
//!
//! This module provides a unified plugin manager for the CLI that handles
//! both format and backend plugins with CLI-specific notification handling.

use crate::output;
use hodu_plugin::rpc::{methods, InitializeResult, LogParams};
use hodu_plugin_runtime::{
    CancellationHandle, ClientError, PluginClient, PluginEntry, PluginRegistry, RegistryError, DEFAULT_TIMEOUT,
};
use std::collections::HashMap;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::Duration;

/// Maximum number of concurrent plugin processes
const MAX_PLUGIN_PROCESSES: usize = 16;

/// Timeout for plugin spawn and initialization (30 seconds)
const PLUGIN_SPAWN_TIMEOUT: Duration = Duration::from_secs(30);

/// Unified plugin manager for CLI
///
/// Manages both format and backend plugin processes with CLI-specific
/// notification handling (uses crate::output for Cargo-style messages).
pub struct PluginManager {
    /// Running plugin processes (name -> process)
    processes: HashMap<String, ManagedPlugin>,
    /// Plugin registry
    registry: PluginRegistry,
    /// Plugins directory
    plugins_dir: PathBuf,
    /// Timeout for plugin operations
    timeout: Duration,
}

/// A managed plugin process
struct ManagedPlugin {
    child: Child,
    client: PluginClient,
    info: InitializeResult,
}

impl PluginManager {
    /// Create a new plugin manager
    pub fn new() -> Result<Self, ProcessError> {
        let registry_path = PluginRegistry::default_path().map_err(ProcessError::Registry)?;
        let registry = PluginRegistry::load(&registry_path).map_err(ProcessError::Registry)?;
        let plugins_dir = PluginRegistry::plugins_dir().map_err(ProcessError::Registry)?;

        Ok(Self {
            processes: HashMap::new(),
            registry,
            plugins_dir,
            timeout: DEFAULT_TIMEOUT,
        })
    }

    /// Create a new plugin manager with a custom timeout
    pub fn with_timeout(timeout_secs: u64) -> Result<Self, ProcessError> {
        let mut manager = Self::new()?;
        manager.timeout = Duration::from_secs(timeout_secs);
        Ok(manager)
    }

    /// Set the timeout for plugin operations
    pub fn set_timeout(&mut self, timeout_secs: u64) {
        self.timeout = Duration::from_secs(timeout_secs);
    }

    /// Get or spawn a plugin by name
    pub fn get_plugin(&mut self, name: &str) -> Result<&mut PluginClient, ProcessError> {
        // Check if already running
        if self.processes.contains_key(name) {
            // SAFETY: We just confirmed the key exists via contains_key(), and nothing
            // can modify the map between these calls in single-threaded Rust code.
            return Ok(&mut self
                .processes
                .get_mut(name)
                .expect("key exists after contains_key check")
                .client);
        }

        // Check process limit before spawning new process
        if self.processes.len() >= MAX_PLUGIN_PROCESSES {
            return Err(ProcessError::TooManyProcesses(MAX_PLUGIN_PROCESSES));
        }

        // Find plugin in registry
        let entry = self
            .registry
            .find(name)
            .ok_or_else(|| ProcessError::NotFound(name.to_string()))?;

        // Check if plugin is enabled
        if !entry.enabled {
            return Err(ProcessError::Disabled(name.to_string()));
        }

        let entry = entry.clone();

        // Spawn plugin
        let managed = self.spawn_plugin(&entry)?;
        self.processes.insert(name.to_string(), managed);

        // SAFETY: We just inserted the key on the line above, so it must exist.
        Ok(&mut self.processes.get_mut(name).expect("key exists after insert").client)
    }

    /// Get a format plugin by extension (tries model format first, then tensor format)
    pub fn get_format_for_extension(&mut self, ext: &str) -> Result<&mut PluginClient, ProcessError> {
        let entry = self
            .registry
            .find_model_format_by_extension(ext)
            .or_else(|| self.registry.find_tensor_format_by_extension(ext))
            .ok_or_else(|| ProcessError::NoFormatForExtension(ext.to_string()))?
            .clone();

        self.get_plugin(&entry.name)
    }

    /// Get a backend plugin by device
    pub fn get_backend_for_device(&mut self, device: &str) -> Result<&mut PluginClient, ProcessError> {
        let entry = self
            .registry
            .find_backend_by_device(device)
            .ok_or_else(|| ProcessError::NoBackendForDevice(device.to_string()))?
            .clone();

        self.get_plugin(&entry.name)
    }

    /// Spawn a plugin process
    fn spawn_plugin(&self, entry: &PluginEntry) -> Result<ManagedPlugin, ProcessError> {
        let binary_path = self.plugins_dir.join(&entry.name).join(&entry.binary);

        if !binary_path.exists() {
            return Err(ProcessError::BinaryNotFound(binary_path.to_string_lossy().to_string()));
        }

        // Spawn process
        let mut child = Command::new(&binary_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| ProcessError::Spawn(e.to_string()))?;

        // Create client
        let mut client = PluginClient::new(&mut child).map_err(ProcessError::Client)?;

        // Set spawn timeout for initialization (shorter than operation timeout)
        client.set_timeout(PLUGIN_SPAWN_TIMEOUT);

        // Set CLI-specific notification handler
        client.set_notification_handler(Box::new(cli_notification_handler));

        // Initialize with spawn timeout
        let info = client.initialize().map_err(ProcessError::Client)?;

        // Set operation timeout for subsequent calls
        client.set_timeout(self.timeout);

        Ok(ManagedPlugin { child, client, info })
    }

    /// Shutdown a specific plugin
    pub fn shutdown_plugin(&mut self, name: &str) -> Result<(), ProcessError> {
        if let Some(mut managed) = self.processes.remove(name) {
            let _ = managed.client.shutdown();
            let _ = managed.child.wait();
        }
        Ok(())
    }

    /// Shutdown all plugins
    pub fn shutdown_all(&mut self) {
        let names: Vec<String> = self.processes.keys().cloned().collect();
        for name in names {
            let _ = self.shutdown_plugin(&name);
        }
    }

    /// Get plugin info if running
    pub fn get_info(&self, name: &str) -> Option<&InitializeResult> {
        self.processes.get(name).map(|p| &p.info)
    }

    /// Get cancellation handle for a plugin (for Ctrl+C handling)
    pub fn get_cancellation_handle(&self, name: &str) -> Option<CancellationHandle> {
        self.processes.get(name).map(|p| p.client.cancellation_handle())
    }
}

impl Drop for PluginManager {
    fn drop(&mut self) {
        self.shutdown_all();
    }
}

/// CLI-specific notification handler that uses Cargo-style output
fn cli_notification_handler(method: &str, params: Option<&serde_json::Value>) {
    match method {
        methods::NOTIFY_PROGRESS => {
            // Progress notifications are silent - status is shown via output module
        },
        methods::NOTIFY_LOG => {
            if let Some(params) = params {
                if let Ok(p) = serde_json::from_value::<LogParams>(params.clone()) {
                    match p.level.to_lowercase().as_str() {
                        "error" => output::error(&p.message),
                        "warn" | "warning" => output::warning(&p.message),
                        _ => {
                            // Info and debug logs are silent in normal mode
                        },
                    }
                }
            }
        },
        _ => {
            // Unknown notification, ignore
        },
    }
}

/// Process management errors
#[derive(Debug)]
pub enum ProcessError {
    Registry(RegistryError),
    NotFound(String),
    Disabled(String),
    NoFormatForExtension(String),
    NoBackendForDevice(String),
    BinaryNotFound(String),
    Spawn(String),
    Client(ClientError),
    TooManyProcesses(usize),
}

impl std::fmt::Display for ProcessError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProcessError::Registry(e) => write!(f, "Registry error: {}", e),
            ProcessError::NotFound(name) => write!(f, "Plugin not found: {}", name),
            ProcessError::Disabled(name) => write!(
                f,
                "Plugin is disabled: {} (use `hodu plugin enable {}` to enable)",
                name, name
            ),
            ProcessError::NoFormatForExtension(ext) => {
                write!(f, "No format plugin found for extension: {}", ext)
            },
            ProcessError::NoBackendForDevice(device) => {
                write!(f, "No backend plugin found for device: {}", device)
            },
            ProcessError::BinaryNotFound(path) => {
                write!(f, "Plugin binary not found: {}", path)
            },
            ProcessError::Spawn(e) => write!(f, "Failed to spawn plugin: {}", e),
            ProcessError::Client(e) => write!(f, "Client error: {}", e),
            ProcessError::TooManyProcesses(max) => {
                write!(f, "Too many plugin processes (max: {})", max)
            },
        }
    }
}

impl std::error::Error for ProcessError {}
