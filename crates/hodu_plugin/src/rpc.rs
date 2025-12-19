//! JSON-RPC 2.0 protocol types for plugin communication
//!
//! This module defines the message types for CLI <-> Plugin communication over stdio.

use serde::{Deserialize, Serialize};

/// JSON-RPC version string
pub const JSONRPC_VERSION: &str = "2.0";

/// Plugin protocol version for compatibility checking
pub const PROTOCOL_VERSION: &str = "1.0.0";

// ============================================================================
// Validation Helpers
// ============================================================================

/// Maximum number of capabilities a plugin can declare
pub const MAX_CAPABILITIES: usize = 50;

/// Maximum number of extensions (model or tensor) a plugin can support
pub const MAX_EXTENSIONS: usize = 100;

/// Maximum number of devices a plugin can support
pub const MAX_DEVICES: usize = 100;

/// Maximum number of supported targets in metadata
pub const MAX_SUPPORTED_TARGETS: usize = 50;

/// Maximum number of inputs in a RunParams request
pub const MAX_INPUTS: usize = 1000;

/// Maximum number of hints in error data
pub const MAX_HINTS: usize = 20;

/// Maximum tensor name length (bytes)
pub const MAX_TENSOR_NAME_LEN: usize = 255;

/// Maximum error message/cause length (64KB)
pub const MAX_ERROR_STRING_LEN: usize = 64 * 1024;

/// Reserved field names in error data
const RESERVED_ERROR_FIELDS: &[&str] = &["cause", "hints", "details", "context", "original_data"];

/// Maximum metadata description length (1KB)
pub const MAX_METADATA_DESCRIPTION_LEN: usize = 1024;

/// Maximum metadata author length (256 bytes)
pub const MAX_METADATA_AUTHOR_LEN: usize = 256;

/// Maximum metadata URL length (2KB for homepage/repository)
pub const MAX_METADATA_URL_LEN: usize = 2048;

/// Maximum metadata license length (64 bytes)
pub const MAX_METADATA_LICENSE_LEN: usize = 64;

/// Maximum metadata version length (64 bytes)
pub const MAX_METADATA_VERSION_LEN: usize = 64;

/// Maximum target triple length (128 bytes)
pub const MAX_TARGET_TRIPLE_LEN: usize = 128;

/// Truncate a string to at most `max_bytes` bytes, respecting UTF-8 character boundaries.
///
/// Returns a new String if truncation is needed, or the original String if within limit.
/// Never panics, even with multi-byte characters (emoji, CJK, etc.).
fn truncate_utf8_owned(s: String, max_bytes: usize) -> String {
    if s.len() <= max_bytes {
        return s;
    }
    // Find the last valid UTF-8 character boundary at or before max_bytes
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    s[..end].to_string()
}

/// Error type for parameter validation
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationError {
    /// Field that failed validation
    pub field: String,
    /// Description of the validation failure
    pub message: String,
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.field, self.message)
    }
}

impl std::error::Error for ValidationError {}

/// Validate a path string (non-empty, no null bytes, not all whitespace, no path traversal)
///
/// # Path Expectations
///
/// This validation accepts both absolute and relative paths. The caller is responsible
/// for determining which type of path is expected for their use case. Paths are validated
/// for basic security concerns but not for existence or accessibility.
fn validate_path(path: &str, field: &str) -> Result<(), ValidationError> {
    if path.is_empty() {
        return Err(ValidationError {
            field: field.to_string(),
            message: "path cannot be empty".to_string(),
        });
    }
    if path.chars().all(|c| c.is_whitespace()) {
        return Err(ValidationError {
            field: field.to_string(),
            message: "path cannot be only whitespace".to_string(),
        });
    }
    if path.contains('\0') {
        return Err(ValidationError {
            field: field.to_string(),
            message: "path contains null byte".to_string(),
        });
    }
    // Check for path traversal sequences (both forward and back slashes)
    // ".." is the standard parent directory reference
    if path.contains("..") {
        return Err(ValidationError {
            field: field.to_string(),
            message: "path contains traversal sequence (..)".to_string(),
        });
    }
    // Check for URL-encoded path traversal (%2e = '.', case insensitive)
    // This catches %2e%2e, %2E%2E, and mixed case variants
    let lower = path.to_lowercase();
    if lower.contains("%2e") {
        return Err(ValidationError {
            field: field.to_string(),
            message: "path contains URL-encoded characters (%2e)".to_string(),
        });
    }
    // Check for double URL-encoding (%25 = '%')
    if lower.contains("%25") {
        return Err(ValidationError {
            field: field.to_string(),
            message: "path contains double-encoded characters (%25)".to_string(),
        });
    }
    // Check for backslash encoding (%5c = '\')
    if lower.contains("%5c") {
        return Err(ValidationError {
            field: field.to_string(),
            message: "path contains encoded backslash (%5c)".to_string(),
        });
    }
    // Check for forward slash encoding (%2f = '/')
    if lower.contains("%2f") {
        return Err(ValidationError {
            field: field.to_string(),
            message: "path contains encoded forward slash (%2f)".to_string(),
        });
    }
    // Check for home directory expansion (could escape intended directories)
    if path.starts_with('~') {
        return Err(ValidationError {
            field: field.to_string(),
            message: "path contains home directory expansion (~)".to_string(),
        });
    }
    // Check for control characters that could cause issues
    if path.chars().any(|c| c.is_control()) {
        return Err(ValidationError {
            field: field.to_string(),
            message: "path contains control characters".to_string(),
        });
    }
    Ok(())
}

/// Validate a tensor name (non-empty, no control chars, no path separators, within length limit)
fn is_valid_tensor_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= MAX_TENSOR_NAME_LEN
        && !name.chars().any(|c| c.is_control())
        && !name.contains('/')
        && !name.contains('\\')
}

/// Validate a non-empty string field
fn validate_non_empty(value: &str, field: &str) -> Result<(), ValidationError> {
    if value.is_empty() {
        return Err(ValidationError {
            field: field.to_string(),
            message: "cannot be empty".to_string(),
        });
    }
    Ok(())
}

// ============================================================================
// Core JSON-RPC Types
// ============================================================================

/// JSON-RPC request message
///
/// Represents a method call from the client to the server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Request {
    /// JSON-RPC version, always "2.0"
    pub jsonrpc: String,
    /// Method name to invoke (e.g., "backend.run", "format.load_model")
    pub method: String,
    /// Optional parameters for the method
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub params: Option<serde_json::Value>,
    /// Unique request identifier for correlating responses
    pub id: RequestId,
}

/// JSON-RPC response message
///
/// Contains either a result or an error, never both.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Response {
    /// JSON-RPC version, always "2.0"
    pub jsonrpc: String,
    /// Success result (mutually exclusive with error)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<serde_json::Value>,
    /// Error object (mutually exclusive with result)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<RpcError>,
    /// Request ID this response corresponds to
    pub id: RequestId,
}

/// JSON-RPC notification message (no id, no response expected)
///
/// Used for one-way messages like progress updates and log messages.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Notification {
    /// JSON-RPC version, always "2.0"
    pub jsonrpc: String,
    /// Notification method name (e.g., "$/progress", "$/log")
    pub method: String,
    /// Optional parameters for the notification
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub params: Option<serde_json::Value>,
}

/// Request ID (can be number, string, or null)
///
/// Per JSON-RPC 2.0 spec, IDs can be numbers, strings, or null.
/// Null is used in error responses when the request ID cannot be determined
/// (e.g., parse errors, invalid JSON).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum RequestId {
    /// Numeric request ID
    Number(i64),
    /// String request ID
    String(String),
    /// Null ID (used for parse errors per JSON-RPC 2.0 spec)
    Null,
}

impl serde::Serialize for RequestId {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            RequestId::Number(n) => serializer.serialize_i64(*n),
            RequestId::String(s) => serializer.serialize_str(s),
            RequestId::Null => serializer.serialize_none(),
        }
    }
}

impl<'de> serde::Deserialize<'de> for RequestId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::{self, Visitor};

        struct RequestIdVisitor;

        impl<'de> Visitor<'de> for RequestIdVisitor {
            type Value = RequestId;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("a number, string, or null")
            }

            fn visit_i64<E: de::Error>(self, v: i64) -> Result<RequestId, E> {
                Ok(RequestId::Number(v))
            }

            fn visit_u64<E: de::Error>(self, v: u64) -> Result<RequestId, E> {
                Ok(RequestId::Number(v as i64))
            }

            fn visit_str<E: de::Error>(self, v: &str) -> Result<RequestId, E> {
                Ok(RequestId::String(v.to_string()))
            }

            fn visit_string<E: de::Error>(self, v: String) -> Result<RequestId, E> {
                Ok(RequestId::String(v))
            }

            fn visit_none<E: de::Error>(self) -> Result<RequestId, E> {
                Ok(RequestId::Null)
            }

            fn visit_unit<E: de::Error>(self) -> Result<RequestId, E> {
                Ok(RequestId::Null)
            }
        }

        deserializer.deserialize_any(RequestIdVisitor)
    }
}

/// JSON-RPC error object
///
/// Standard error format with code, message, and optional data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RpcError {
    /// Error code (see [`error_codes`] for standard values)
    pub code: i32,
    /// Human-readable error message
    pub message: String,
    /// Optional additional error data
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
}

// ============================================================================
// Error Codes (JSON-RPC standard + custom)
// ============================================================================

/// Standard JSON-RPC 2.0 error codes and custom plugin error codes
pub mod error_codes {
    /// Parse error - invalid JSON was received
    pub const PARSE_ERROR: i32 = -32700;
    /// Invalid request - the JSON sent is not a valid Request object
    pub const INVALID_REQUEST: i32 = -32600;
    /// Method not found - the method does not exist or is not available
    pub const METHOD_NOT_FOUND: i32 = -32601;
    /// Invalid params - invalid method parameters
    pub const INVALID_PARAMS: i32 = -32602;
    /// Internal error - internal JSON-RPC error
    pub const INTERNAL_ERROR: i32 = -32603;

    /// Generic plugin error
    pub const PLUGIN_ERROR: i32 = -32000;
    /// Feature or capability not supported
    pub const NOT_SUPPORTED: i32 = -32001;
    /// File not found at the specified path
    pub const FILE_NOT_FOUND: i32 = -32002;
    /// Invalid file format
    pub const INVALID_FORMAT: i32 = -32003;
    /// Requested device is not available
    pub const DEVICE_NOT_AVAILABLE: i32 = -32004;
    /// Error loading or processing model
    pub const MODEL_ERROR: i32 = -32005;
    /// Error loading or processing tensor
    pub const TENSOR_ERROR: i32 = -32006;
    /// Request was cancelled by client
    pub const REQUEST_CANCELLED: i32 = -32007;
}

// ============================================================================
// Method Names
// ============================================================================

/// Standard method names for plugin communication
pub mod methods {
    /// Initialize plugin and exchange capabilities
    pub const INITIALIZE: &str = "initialize";
    /// Graceful shutdown request
    pub const SHUTDOWN: &str = "shutdown";

    /// Load a model file and convert to snapshot
    pub const FORMAT_LOAD_MODEL: &str = "format.load_model";
    /// Save a snapshot to model file
    pub const FORMAT_SAVE_MODEL: &str = "format.save_model";
    /// Load a tensor file
    pub const FORMAT_LOAD_TENSOR: &str = "format.load_tensor";
    /// Save a tensor to file
    pub const FORMAT_SAVE_TENSOR: &str = "format.save_tensor";

    /// Run model inference
    pub const BACKEND_RUN: &str = "backend.run";
    /// AOT compile model to native code
    pub const BACKEND_BUILD: &str = "backend.build";
    /// Query supported devices
    pub const BACKEND_SUPPORTED_DEVICES: &str = "backend.supported_devices";
    /// Query supported build targets
    pub const BACKEND_SUPPORTED_TARGETS: &str = "backend.supported_targets";

    /// Progress notification (plugin -> CLI)
    pub const NOTIFY_PROGRESS: &str = "$/progress";
    /// Log message notification (plugin -> CLI)
    pub const NOTIFY_LOG: &str = "$/log";

    /// Cancel a running request (CLI -> plugin)
    pub const CANCEL: &str = "$/cancel";
}

// ============================================================================
// Request/Response Params
// ============================================================================

/// Initialize request params
///
/// Sent by CLI to initialize the plugin and exchange version information.
///
/// # Version Format
///
/// Both version fields follow [Semantic Versioning](https://semver.org/) format: `MAJOR.MINOR.PATCH`
///
/// - `plugin_version`: The CLI's plugin SDK version (e.g., "0.1.0")
/// - `protocol_version`: The JSON-RPC protocol version, currently "1.0.0"
///
/// ## Compatibility Rules
///
/// - Plugins should check `protocol_version` matches their expected version
/// - Major version changes indicate breaking changes
/// - Minor version changes are backward compatible
/// - Patch version changes are bug fixes only
///
/// # Example
///
/// ```ignore
/// let params = InitializeParams {
///     plugin_version: "0.1.0".to_string(),
///     protocol_version: "1.0.0".to_string(),
/// };
/// assert!(params.validate().is_ok());
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InitializeParams {
    /// CLI's plugin SDK version (semver format: "MAJOR.MINOR.PATCH")
    ///
    /// Used for compatibility checking. Plugins can use this to enable/disable
    /// features based on the CLI version.
    pub plugin_version: String,
    /// JSON-RPC protocol version (semver format, currently "1.0.0")
    ///
    /// Plugins should verify this matches their expected protocol version.
    /// Protocol version changes indicate changes to the RPC message format.
    pub protocol_version: String,
}

impl InitializeParams {
    /// Validate the parameters
    ///
    /// Checks that version strings are non-empty. Note that this does not
    /// validate the semver format - it only ensures the fields are present.
    /// Callers should perform additional validation if strict semver
    /// compliance is required.
    pub fn validate(&self) -> Result<(), ValidationError> {
        validate_non_empty(&self.plugin_version, "plugin_version")?;
        validate_non_empty(&self.protocol_version, "protocol_version")
    }
}

/// Plugin metadata for discovery and compatibility checking
///
/// Optional metadata that plugins can provide for better integration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PluginMetadataRpc {
    /// Human-readable plugin description
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Plugin author name or organization
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub author: Option<String>,
    /// Plugin homepage URL
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub homepage: Option<String>,
    /// License identifier (e.g., "MIT", "Apache-2.0")
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub license: Option<String>,
    /// Source repository URL
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repository: Option<String>,
    /// Supported target triples (e.g., "x86_64-*-*", "aarch64-apple-darwin")
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supported_targets: Option<Vec<String>>,
    /// Minimum required hodu version (semver, e.g., "0.1.0")
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_hodu_version: Option<String>,
}

impl PluginMetadataRpc {
    /// Validate metadata field lengths
    ///
    /// Returns `Ok(())` if all fields are within limits, or
    /// `Err(ValidationError)` with details about which limit was exceeded.
    pub fn validate(&self) -> Result<(), ValidationError> {
        if let Some(ref desc) = self.description {
            if desc.len() > MAX_METADATA_DESCRIPTION_LEN {
                return Err(ValidationError {
                    field: "description".to_string(),
                    message: format!(
                        "description exceeds {} bytes (got {})",
                        MAX_METADATA_DESCRIPTION_LEN,
                        desc.len()
                    ),
                });
            }
        }
        if let Some(ref author) = self.author {
            if author.len() > MAX_METADATA_AUTHOR_LEN {
                return Err(ValidationError {
                    field: "author".to_string(),
                    message: format!(
                        "author exceeds {} bytes (got {})",
                        MAX_METADATA_AUTHOR_LEN,
                        author.len()
                    ),
                });
            }
        }
        if let Some(ref homepage) = self.homepage {
            if homepage.len() > MAX_METADATA_URL_LEN {
                return Err(ValidationError {
                    field: "homepage".to_string(),
                    message: format!(
                        "homepage URL exceeds {} bytes (got {})",
                        MAX_METADATA_URL_LEN,
                        homepage.len()
                    ),
                });
            }
        }
        if let Some(ref license) = self.license {
            if license.len() > MAX_METADATA_LICENSE_LEN {
                return Err(ValidationError {
                    field: "license".to_string(),
                    message: format!(
                        "license exceeds {} bytes (got {})",
                        MAX_METADATA_LICENSE_LEN,
                        license.len()
                    ),
                });
            }
        }
        if let Some(ref repository) = self.repository {
            if repository.len() > MAX_METADATA_URL_LEN {
                return Err(ValidationError {
                    field: "repository".to_string(),
                    message: format!(
                        "repository URL exceeds {} bytes (got {})",
                        MAX_METADATA_URL_LEN,
                        repository.len()
                    ),
                });
            }
        }
        if let Some(ref targets) = self.supported_targets {
            if targets.len() > MAX_SUPPORTED_TARGETS {
                return Err(ValidationError {
                    field: "supported_targets".to_string(),
                    message: format!(
                        "supported_targets exceeds {} items (got {})",
                        MAX_SUPPORTED_TARGETS,
                        targets.len()
                    ),
                });
            }
            for (i, target) in targets.iter().enumerate() {
                if target.len() > MAX_TARGET_TRIPLE_LEN {
                    return Err(ValidationError {
                        field: format!("supported_targets[{}]", i),
                        message: format!(
                            "target triple exceeds {} bytes (got {})",
                            MAX_TARGET_TRIPLE_LEN,
                            target.len()
                        ),
                    });
                }
            }
        }
        if let Some(ref version) = self.min_hodu_version {
            if version.len() > MAX_METADATA_VERSION_LEN {
                return Err(ValidationError {
                    field: "min_hodu_version".to_string(),
                    message: format!(
                        "min_hodu_version exceeds {} bytes (got {})",
                        MAX_METADATA_VERSION_LEN,
                        version.len()
                    ),
                });
            }
        }
        Ok(())
    }

    /// Sanitize metadata by truncating fields to their maximum lengths
    ///
    /// This is a lenient alternative to validation that ensures all fields
    /// are within limits by truncating them if necessary.
    pub fn sanitize(&mut self) {
        if let Some(ref mut desc) = self.description {
            *desc = truncate_utf8_owned(std::mem::take(desc), MAX_METADATA_DESCRIPTION_LEN);
        }
        if let Some(ref mut author) = self.author {
            *author = truncate_utf8_owned(std::mem::take(author), MAX_METADATA_AUTHOR_LEN);
        }
        if let Some(ref mut homepage) = self.homepage {
            *homepage = truncate_utf8_owned(std::mem::take(homepage), MAX_METADATA_URL_LEN);
        }
        if let Some(ref mut license) = self.license {
            *license = truncate_utf8_owned(std::mem::take(license), MAX_METADATA_LICENSE_LEN);
        }
        if let Some(ref mut repository) = self.repository {
            *repository = truncate_utf8_owned(std::mem::take(repository), MAX_METADATA_URL_LEN);
        }
        if let Some(ref mut targets) = self.supported_targets {
            targets.truncate(MAX_SUPPORTED_TARGETS);
            for target in targets.iter_mut() {
                *target = truncate_utf8_owned(std::mem::take(target), MAX_TARGET_TRIPLE_LEN);
            }
        }
        if let Some(ref mut version) = self.min_hodu_version {
            *version = truncate_utf8_owned(std::mem::take(version), MAX_METADATA_VERSION_LEN);
        }
    }
}

/// Initialize response result
///
/// Contains plugin information and capabilities for CLI discovery.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InitializeResult {
    /// Plugin name (e.g., "hodu-format-onnx")
    pub name: String,
    /// Plugin version (semver)
    pub version: String,
    /// Supported protocol version
    pub protocol_version: String,
    /// Plugin SDK version used to build the plugin
    pub plugin_version: String,
    /// List of supported method names (e.g., ["format.load_model", "format.save_model"])
    pub capabilities: Vec<String>,
    /// Supported model file extensions (for format plugins)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_extensions: Option<Vec<String>>,
    /// Supported tensor file extensions (for format plugins)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tensor_extensions: Option<Vec<String>>,
    /// Supported devices (for backend plugins, e.g., ["cpu", "cuda::0"])
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub devices: Option<Vec<String>>,
    /// Optional plugin metadata
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<PluginMetadataRpc>,
}

impl InitializeResult {
    /// Check if collection sizes are within limits
    ///
    /// Returns `true` if all collections are within their size limits.
    /// For detailed error information, use `validate_limits()` instead.
    pub fn is_within_limits(&self) -> bool {
        self.validate_limits().is_ok()
    }

    /// Validate collection sizes
    ///
    /// Returns `Ok(())` if all collections are within limits, or
    /// `Err(ValidationError)` with details about which limit was exceeded.
    pub fn validate_limits(&self) -> Result<(), ValidationError> {
        if self.capabilities.len() > MAX_CAPABILITIES {
            return Err(ValidationError {
                field: "capabilities".to_string(),
                message: format!(
                    "too many capabilities ({} > {})",
                    self.capabilities.len(),
                    MAX_CAPABILITIES
                ),
            });
        }
        if let Some(ref exts) = self.model_extensions {
            if exts.len() > MAX_EXTENSIONS {
                return Err(ValidationError {
                    field: "model_extensions".to_string(),
                    message: format!("too many extensions ({} > {})", exts.len(), MAX_EXTENSIONS),
                });
            }
        }
        if let Some(ref exts) = self.tensor_extensions {
            if exts.len() > MAX_EXTENSIONS {
                return Err(ValidationError {
                    field: "tensor_extensions".to_string(),
                    message: format!("too many extensions ({} > {})", exts.len(), MAX_EXTENSIONS),
                });
            }
        }
        if let Some(ref devices) = self.devices {
            if devices.len() > MAX_DEVICES {
                return Err(ValidationError {
                    field: "devices".to_string(),
                    message: format!("too many devices ({} > {})", devices.len(), MAX_DEVICES),
                });
            }
        }
        if let Some(ref meta) = self.metadata {
            // Validate all metadata fields including string lengths
            meta.validate().map_err(|e| ValidationError {
                field: format!("metadata.{}", e.field),
                message: e.message,
            })?;
        }
        Ok(())
    }
}

/// Load model request params
///
/// Request to load a model file and convert it to hodu snapshot format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadModelParams {
    /// Path to the input model file
    pub path: String,
}

impl LoadModelParams {
    /// Validate the parameters
    pub fn validate(&self) -> Result<(), ValidationError> {
        validate_path(&self.path, "path")
    }
}

/// Load model response result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadModelResult {
    /// Path to the generated snapshot file (.hdss)
    pub snapshot_path: String,
}

/// Save model request params
///
/// Request to convert a snapshot back to a specific model format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SaveModelParams {
    /// Path to the input snapshot file
    pub snapshot_path: String,
    /// Path for the output model file
    pub output_path: String,
}

impl SaveModelParams {
    /// Validate the parameters
    pub fn validate(&self) -> Result<(), ValidationError> {
        validate_path(&self.snapshot_path, "snapshot_path")?;
        validate_path(&self.output_path, "output_path")
    }
}

/// Load tensor request params
///
/// Request to load a tensor file and convert to hodu tensor format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadTensorParams {
    /// Path to the input tensor file
    pub path: String,
}

impl LoadTensorParams {
    /// Validate the parameters
    pub fn validate(&self) -> Result<(), ValidationError> {
        validate_path(&self.path, "path")
    }
}

/// Load tensor response result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadTensorResult {
    /// Path to the generated tensor file (.hdt)
    pub tensor_path: String,
}

/// Save tensor request params
///
/// Request to convert a tensor to a specific format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SaveTensorParams {
    /// Path to the input tensor file (.hdt)
    pub tensor_path: String,
    /// Path for the output tensor file
    pub output_path: String,
}

impl SaveTensorParams {
    /// Validate the parameters
    pub fn validate(&self) -> Result<(), ValidationError> {
        validate_path(&self.tensor_path, "tensor_path")?;
        validate_path(&self.output_path, "output_path")
    }
}

/// Backend run request params
///
/// Request to execute model inference on a compiled library.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunParams {
    /// Path to compiled library (.dylib, .so, .dll)
    pub library_path: String,
    /// Path to snapshot (needed for input/output metadata)
    pub snapshot_path: String,
    /// Target device (e.g., "cpu", "cuda::0", "metal")
    pub device: String,
    /// Input tensors to feed into the model
    pub inputs: Vec<TensorInput>,
}

impl RunParams {
    /// Validate the parameters
    pub fn validate(&self) -> Result<(), ValidationError> {
        validate_path(&self.library_path, "library_path")?;
        validate_path(&self.snapshot_path, "snapshot_path")?;
        validate_non_empty(&self.device, "device")?;
        if self.inputs.len() > MAX_INPUTS {
            return Err(ValidationError {
                field: "inputs".to_string(),
                message: format!("too many inputs ({} > {})", self.inputs.len(), MAX_INPUTS),
            });
        }
        for (i, input) in self.inputs.iter().enumerate() {
            input.validate().map_err(|mut e| {
                e.field = format!("inputs[{}].{}", i, e.field);
                e
            })?;
        }
        Ok(())
    }
}

/// Input tensor reference for model execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorInput {
    /// Input tensor name (must match model input name)
    pub name: String,
    /// Path to tensor file (.hdt)
    pub path: String,
}

impl TensorInput {
    /// Create new tensor input
    pub fn new(name: impl Into<String>, path: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            path: path.into(),
        }
    }

    /// Validate tensor name (non-empty, no control chars, no path separators)
    pub fn is_valid_name(&self) -> bool {
        is_valid_tensor_name(&self.name)
    }

    /// Validate the tensor input
    pub fn validate(&self) -> Result<(), ValidationError> {
        if !self.is_valid_name() {
            return Err(ValidationError {
                field: "name".to_string(),
                message: format!(
                    "invalid tensor name (must be non-empty, max {} bytes, no control chars or path separators)",
                    MAX_TENSOR_NAME_LEN
                ),
            });
        }
        validate_path(&self.path, "path")
    }
}

/// Backend run response result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunResult {
    /// Output tensors from model execution
    pub outputs: Vec<TensorOutput>,
}

/// Output tensor reference from model execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorOutput {
    /// Output tensor name
    pub name: String,
    /// Path to output tensor file (.hdt)
    pub path: String,
}

impl TensorOutput {
    /// Create new tensor output
    pub fn new(name: impl Into<String>, path: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            path: path.into(),
        }
    }

    /// Validate the tensor output
    pub fn validate(&self) -> Result<(), ValidationError> {
        if !self.is_valid_name() {
            return Err(ValidationError {
                field: "name".to_string(),
                message: format!(
                    "invalid tensor name (must be non-empty, max {} bytes, no control chars or path separators)",
                    MAX_TENSOR_NAME_LEN
                ),
            });
        }
        validate_path(&self.path, "path")
    }

    /// Validate tensor name (non-empty, no control chars, no path separators, within length limit)
    pub fn is_valid_name(&self) -> bool {
        is_valid_tensor_name(&self.name)
    }
}

/// Backend build request params
///
/// Request to AOT compile a model for a specific target.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildParams {
    /// Path to the snapshot file to compile
    pub snapshot_path: String,
    /// Target triple (e.g., "x86_64-apple-darwin", "aarch64-unknown-linux-gnu")
    pub target: String,
    /// Target device (e.g., "cpu", "cuda::0", "metal")
    pub device: String,
    /// Output format (e.g., "sharedlib", "staticlib")
    pub format: String,
    /// Path for the compiled output
    pub output_path: String,
}

impl BuildParams {
    /// Validate the parameters
    pub fn validate(&self) -> Result<(), ValidationError> {
        validate_path(&self.snapshot_path, "snapshot_path")?;
        validate_non_empty(&self.target, "target")?;
        validate_non_empty(&self.device, "device")?;
        validate_non_empty(&self.format, "format")?;
        validate_path(&self.output_path, "output_path")
    }
}

/// Backend list targets response result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListTargetsResult {
    /// List of supported target triples
    pub targets: Vec<String>,
    /// Human-readable formatted list of targets (for display)
    pub formatted: String,
}

/// Progress notification params (plugin -> CLI)
///
/// Sent by plugins to report progress during long-running operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressParams {
    /// Progress percentage (0-100), None for indeterminate progress
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub percent: Option<u8>,
    /// Human-readable progress message
    pub message: String,
}

impl ProgressParams {
    /// Validate the parameters
    pub fn validate(&self) -> Result<(), ValidationError> {
        if let Some(percent) = self.percent {
            if percent > 100 {
                return Err(ValidationError {
                    field: "percent".to_string(),
                    message: format!("percent must be 0-100, got {}", percent),
                });
            }
        }
        validate_non_empty(&self.message, "message")
    }
}

/// Valid log levels for LogParams
pub const VALID_LOG_LEVELS: &[&str] = &["error", "warn", "info", "debug", "trace"];

/// Log notification params (plugin -> CLI)
///
/// Sent by plugins to emit log messages to the CLI.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogParams {
    /// Log level: "error", "warn", "info", "debug", "trace"
    pub level: String,
    /// Log message content
    pub message: String,
}

impl LogParams {
    /// Create new log params with validation
    pub fn new(level: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            level: level.into(),
            message: message.into(),
        }
    }

    /// Validate the parameters
    pub fn validate(&self) -> Result<(), ValidationError> {
        if !self.is_valid_level() {
            return Err(ValidationError {
                field: "level".to_string(),
                message: format!(
                    "invalid log level '{}', expected one of: {}",
                    self.level,
                    VALID_LOG_LEVELS.join(", ")
                ),
            });
        }
        validate_non_empty(&self.message, "message")
    }

    /// Check if the log level is valid
    pub fn is_valid_level(&self) -> bool {
        VALID_LOG_LEVELS.contains(&self.level.as_str())
    }

    /// Get the normalized log level (returns "info" for invalid levels)
    pub fn normalized_level(&self) -> &str {
        if self.is_valid_level() {
            &self.level
        } else {
            "info"
        }
    }
}

/// Cancel request params (CLI -> plugin)
///
/// Sent by CLI to request cancellation of an in-progress operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CancelParams {
    /// Request ID of the operation to cancel
    pub id: RequestId,
}

// ============================================================================
// Helper Implementations
// ============================================================================

/// Error type for Request validation
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RequestValidationError {
    /// Method name is empty
    EmptyMethod,
    /// Method name contains invalid characters
    InvalidMethodChars,
}

impl std::fmt::Display for RequestValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyMethod => write!(f, "Method name cannot be empty"),
            Self::InvalidMethodChars => write!(f, "Method name contains invalid characters"),
        }
    }
}

impl std::error::Error for RequestValidationError {}

/// Error type for Response validation
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResponseValidationError {
    /// Both result and error are set
    BothResultAndError,
    /// Neither result nor error is set
    NeitherResultNorError,
}

impl std::fmt::Display for ResponseValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BothResultAndError => write!(f, "Response cannot have both result and error"),
            Self::NeitherResultNorError => write!(f, "Response must have either result or error"),
        }
    }
}

impl std::error::Error for ResponseValidationError {}

impl Request {
    /// Create a new JSON-RPC request
    ///
    /// # Arguments
    /// * `method` - Method name to invoke
    /// * `params` - Optional parameters as JSON value
    /// * `id` - Request ID for response correlation
    pub fn new(method: impl Into<String>, params: Option<serde_json::Value>, id: RequestId) -> Self {
        Self {
            jsonrpc: JSONRPC_VERSION.to_string(),
            method: method.into(),
            params,
            id,
        }
    }

    /// Create a new JSON-RPC request with validation
    ///
    /// Returns an error if the method name is empty, contains invalid characters,
    /// or doesn't contain at least one alphanumeric character.
    pub fn new_checked(
        method: impl Into<String>,
        params: Option<serde_json::Value>,
        id: RequestId,
    ) -> Result<Self, RequestValidationError> {
        let method = method.into();

        if method.is_empty() || method.chars().all(|c| c.is_whitespace()) {
            return Err(RequestValidationError::EmptyMethod);
        }

        // Method names should be alphanumeric with dots, underscores, slashes, and $ for internal methods
        let valid_chars = method
            .chars()
            .all(|c| c.is_alphanumeric() || c == '.' || c == '_' || c == '/' || c == '$');
        if !valid_chars {
            return Err(RequestValidationError::InvalidMethodChars);
        }

        // Must contain at least one alphanumeric character (not just punctuation)
        if !method.chars().any(|c| c.is_alphanumeric()) {
            return Err(RequestValidationError::InvalidMethodChars);
        }

        Ok(Self {
            jsonrpc: JSONRPC_VERSION.to_string(),
            method,
            params,
            id,
        })
    }

    /// Check if this request has a valid method name
    ///
    /// Validates that the method name:
    /// - Is not empty or whitespace-only
    /// - Contains only valid characters (alphanumeric, '.', '_', '/', '$')
    /// - Contains at least one alphanumeric character
    pub fn is_valid(&self) -> bool {
        if self.method.is_empty() || self.method.chars().all(|c| c.is_whitespace()) {
            return false;
        }
        // Method names should be alphanumeric with dots, underscores, slashes, and $ for internal methods
        let valid_chars = self
            .method
            .chars()
            .all(|c| c.is_alphanumeric() || c == '.' || c == '_' || c == '/' || c == '$');
        if !valid_chars {
            return false;
        }
        // Must contain at least one alphanumeric character
        self.method.chars().any(|c| c.is_alphanumeric())
    }
}

impl Response {
    /// Create a success response with the given result
    ///
    /// # Arguments
    /// * `id` - Request ID this response corresponds to
    /// * `result` - Success result as JSON value
    pub fn success(id: RequestId, result: serde_json::Value) -> Self {
        Self {
            jsonrpc: JSONRPC_VERSION.to_string(),
            result: Some(result),
            error: None,
            id,
        }
    }

    /// Create an error response with the given error
    ///
    /// # Arguments
    /// * `id` - Request ID this response corresponds to
    /// * `error` - Error object with code and message
    pub fn error(id: RequestId, error: RpcError) -> Self {
        Self {
            jsonrpc: JSONRPC_VERSION.to_string(),
            result: None,
            error: Some(error),
            id,
        }
    }

    /// Check if this is a success response
    pub fn is_success(&self) -> bool {
        self.result.is_some() && self.error.is_none()
    }

    /// Check if this is an error response
    pub fn is_error(&self) -> bool {
        self.error.is_some() && self.result.is_none()
    }

    /// Check if this response is valid per JSON-RPC 2.0 spec
    ///
    /// A valid response must have exactly one of `result` or `error`, not both, not neither.
    pub fn is_valid(&self) -> bool {
        matches!((&self.result, &self.error), (Some(_), None) | (None, Some(_)))
    }

    /// Validate this response and return an error if invalid
    ///
    /// Per JSON-RPC 2.0 spec, a response must have exactly one of `result` or `error`.
    pub fn validate(&self) -> Result<(), ResponseValidationError> {
        match (&self.result, &self.error) {
            (Some(_), None) | (None, Some(_)) => Ok(()),
            (Some(_), Some(_)) => Err(ResponseValidationError::BothResultAndError),
            (None, None) => Err(ResponseValidationError::NeitherResultNorError),
        }
    }
}

impl Notification {
    /// Create a new JSON-RPC notification (one-way message)
    ///
    /// # Arguments
    /// * `method` - Notification method name
    /// * `params` - Optional parameters as JSON value
    pub fn new(method: impl Into<String>, params: Option<serde_json::Value>) -> Self {
        Self {
            jsonrpc: JSONRPC_VERSION.to_string(),
            method: method.into(),
            params,
        }
    }

    /// Create a progress notification
    ///
    /// # Arguments
    /// * `percent` - Progress percentage (0-100), None for indeterminate. Values > 100 are clamped.
    /// * `message` - Human-readable progress message
    ///
    /// # Percent Handling: Clamping vs Validation
    ///
    /// This method and the SDK's `notify_progress()` use a **convenience clamping** strategy:
    /// values > 100 are silently clamped to 100. This is intentional for ease of use since
    /// progress calculations may occasionally overflow due to rounding.
    ///
    /// In contrast, `ProgressParams::validate()` uses **strict validation** that rejects
    /// values > 100. Use this when you need to ensure data integrity (e.g., deserializing
    /// untrusted input).
    ///
    /// | Method | Strategy | Use Case |
    /// |--------|----------|----------|
    /// | `Notification::progress()` | Clamps > 100 | Fire-and-forget progress |
    /// | `notify_progress()` | Clamps > 100 | SDK convenience function |
    /// | `ProgressParams::validate()` | Rejects > 100 | Validating untrusted input |
    pub fn progress(percent: Option<u8>, message: impl Into<String>) -> Self {
        // Clamp percent to valid range 0-100 (for convenience; strict validation available via ProgressParams::validate())
        let percent = percent.map(|p| p.min(100));
        let message = message.into();

        // Use json! macro which is infallible for primitive types
        Self::new(
            methods::NOTIFY_PROGRESS,
            Some(serde_json::json!({
                "percent": percent,
                "message": message
            })),
        )
    }

    /// Create a log notification
    ///
    /// # Arguments
    /// * `level` - Log level: "error", "warn", "info", "debug", "trace"
    /// * `message` - Log message content
    pub fn log(level: impl Into<String>, message: impl Into<String>) -> Self {
        let level = level.into();
        let message = message.into();

        // Use json! macro which is infallible for primitive types
        Self::new(
            methods::NOTIFY_LOG,
            Some(serde_json::json!({
                "level": level,
                "message": message
            })),
        )
    }
}

impl RpcError {
    /// Create a new RPC error with code and message
    pub fn new(code: i32, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            data: None,
        }
    }

    /// Create a new RPC error with additional data
    pub fn with_data(code: i32, message: impl Into<String>, data: serde_json::Value) -> Self {
        Self {
            code,
            message: message.into(),
            data: Some(data),
        }
    }

    /// Create a parse error (-32700) - invalid JSON received
    pub fn parse_error(msg: impl Into<String>) -> Self {
        Self::new(error_codes::PARSE_ERROR, msg)
    }

    /// Create an invalid request error (-32600) - malformed request object
    pub fn invalid_request(msg: impl Into<String>) -> Self {
        Self::new(error_codes::INVALID_REQUEST, msg)
    }

    /// Create a method not found error (-32601)
    pub fn method_not_found(method: impl Into<String>) -> Self {
        let method = method.into();
        Self::new(error_codes::METHOD_NOT_FOUND, format!("Method not found: {}", method))
    }

    /// Create an invalid params error (-32602) - wrong or missing parameters
    pub fn invalid_params(msg: impl Into<String>) -> Self {
        Self::new(error_codes::INVALID_PARAMS, msg)
    }

    /// Create an internal error (-32603) - internal server/plugin error
    pub fn internal_error(msg: impl Into<String>) -> Self {
        Self::new(error_codes::INTERNAL_ERROR, msg)
    }

    /// Create a not supported error (-32001) - requested feature unavailable
    pub fn not_supported(feature: impl Into<String>) -> Self {
        let feature = feature.into();
        Self::new(error_codes::NOT_SUPPORTED, format!("Not supported: {}", feature))
    }

    /// Create a file not found error (-32002)
    pub fn file_not_found(path: impl Into<String>) -> Self {
        let path = path.into();
        Self::new(error_codes::FILE_NOT_FOUND, format!("File not found: {}", path))
    }

    /// Create a cancelled error (-32007) - request was cancelled
    pub fn cancelled() -> Self {
        Self::new(error_codes::REQUEST_CANCELLED, "Request cancelled")
    }

    /// Create a device not available error (-32004)
    pub fn device_not_available(device: impl Into<String>) -> Self {
        let device = device.into();
        Self::new(
            error_codes::DEVICE_NOT_AVAILABLE,
            format!("Device not available: {}", device),
        )
    }

    /// Create a model error (-32005) - error loading or processing model
    pub fn model_error(msg: impl Into<String>) -> Self {
        Self::new(error_codes::MODEL_ERROR, msg)
    }

    /// Create a tensor error (-32006) - error loading or processing tensor
    pub fn tensor_error(msg: impl Into<String>) -> Self {
        Self::new(error_codes::TENSOR_ERROR, msg)
    }

    /// Create a generic plugin error (-32000)
    pub fn plugin_error(msg: impl Into<String>) -> Self {
        Self::new(error_codes::PLUGIN_ERROR, msg)
    }

    /// Create an invalid format error (-32003)
    pub fn invalid_format(msg: impl Into<String>) -> Self {
        Self::new(error_codes::INVALID_FORMAT, msg)
    }

    // =========================================================================
    // Error Chain/Cause Support
    // =========================================================================

    /// Add a cause/source error to this error
    ///
    /// The cause is stored in the `data` field as `{"cause": "..."}`.
    /// Multiple calls chain the causes.
    ///
    /// # Limits
    ///
    /// Cause strings exceeding 64KB are truncated to prevent memory issues.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let error = RpcError::internal_error("Failed to process")
    ///     .with_cause("IO error: file not found");
    /// ```
    pub fn with_cause(mut self, cause: impl Into<String>) -> Self {
        // Truncate overly long cause strings (UTF-8 safe)
        let cause_str = truncate_utf8_owned(cause.into(), MAX_ERROR_STRING_LEN);
        self.data = Some(match self.data {
            Some(mut data) => {
                if let Some(obj) = data.as_object_mut() {
                    // Chain causes if already exists
                    if let Some(existing) = obj.get("cause") {
                        // Extract string from existing cause, handling non-string values
                        let existing_str = match existing {
                            serde_json::Value::String(s) => s.clone(),
                            other => other.to_string(),
                        };
                        // Truncate combined cause if too long (UTF-8 safe)
                        let combined = format!("{} <- {}", cause_str, existing_str);
                        let combined = truncate_utf8_owned(combined, MAX_ERROR_STRING_LEN);
                        obj.insert("cause".to_string(), serde_json::json!(combined));
                    } else {
                        obj.insert("cause".to_string(), serde_json::json!(cause_str));
                    }
                    data
                } else {
                    serde_json::json!({"cause": cause_str, "original_data": data})
                }
            },
            None => serde_json::json!({"cause": cause_str}),
        });
        self
    }

    /// Add a cause from a std::error::Error
    ///
    /// Automatically extracts the error chain using `source()`.
    pub fn with_error_cause<E: std::error::Error>(self, error: &E) -> Self {
        let mut causes = vec![error.to_string()];
        let mut source = error.source();
        while let Some(err) = source {
            causes.push(err.to_string());
            source = err.source();
        }
        self.with_cause(causes.join(" <- "))
    }

    // =========================================================================
    // Recovery Hints Support
    // =========================================================================

    /// Add a recovery hint to this error
    ///
    /// Hints suggest possible actions to resolve the error.
    ///
    /// # Limits
    ///
    /// - Maximum 20 hints per error (additional hints are dropped with a debug warning)
    /// - Hints exceeding 64KB are truncated
    ///
    /// # Example
    ///
    /// ```ignore
    /// let error = RpcError::file_not_found("/path/to/model.onnx")
    ///     .with_hint("Check if the file path is correct")
    ///     .with_hint("Ensure the file exists and is readable");
    /// ```
    pub fn with_hint(mut self, hint: impl Into<String>) -> Self {
        // Truncate overly long hints (UTF-8 safe)
        let hint_str = truncate_utf8_owned(hint.into(), MAX_ERROR_STRING_LEN);
        self.data = Some(match self.data {
            Some(mut data) => {
                if let Some(obj) = data.as_object_mut() {
                    if let Some(hints) = obj.get_mut("hints") {
                        if let Some(arr) = hints.as_array_mut() {
                            // Enforce hints limit with warning
                            if arr.len() < MAX_HINTS {
                                arr.push(serde_json::json!(hint_str));
                            } else {
                                // Log warning in both debug and release builds
                                eprintln!(
                                    "Warning: Maximum hints ({}) reached, hint dropped: {}",
                                    MAX_HINTS,
                                    if hint_str.len() > 50 {
                                        // UTF-8 safe truncation for preview
                                        format!("{}...", truncate_utf8_owned(hint_str.clone(), 50))
                                    } else {
                                        hint_str.clone()
                                    }
                                );
                            }
                        } else {
                            // hints exists but is not an array - convert to array
                            let existing = hints.take();
                            *hints = serde_json::json!([existing, hint_str]);
                        }
                    } else {
                        obj.insert("hints".to_string(), serde_json::json!([hint_str]));
                    }
                    data
                } else {
                    serde_json::json!({"hints": [hint_str], "original_data": data})
                }
            },
            None => serde_json::json!({"hints": [hint_str]}),
        });
        self
    }

    /// Add multiple recovery hints at once
    pub fn with_hints<I, S>(mut self, hints: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        for hint in hints {
            self = self.with_hint(hint);
        }
        self
    }

    // =========================================================================
    // Structured Error Data Support
    // =========================================================================

    /// Add a structured field to the error data
    ///
    /// # Example
    ///
    /// ```ignore
    /// let error = RpcError::invalid_params("Shape mismatch")
    ///     .with_field("expected", serde_json::json!([1, 3, 224, 224]))
    ///     .with_field("actual", serde_json::json!([1, 3, 256, 256]));
    /// ```
    pub fn with_field(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        let key = key.into();
        // Warn if using reserved field names (they might be overwritten by other methods)
        // Log in all builds since this indicates a programming error
        if RESERVED_ERROR_FIELDS.contains(&key.as_str()) {
            eprintln!(
                "Warning: with_field() using reserved field name '{}', may be overwritten",
                key
            );
        }
        self.data = Some(match self.data {
            Some(mut data) => {
                if let Some(obj) = data.as_object_mut() {
                    obj.insert(key, value);
                    data
                } else {
                    serde_json::json!({key: value, "original_data": data})
                }
            },
            None => serde_json::json!({key: value}),
        });
        self
    }

    /// Add a details string to the error
    ///
    /// Shortcut for `.with_field("details", ...)`.
    pub fn with_details(self, details: impl Into<String>) -> Self {
        self.with_field("details", serde_json::json!(details.into()))
    }

    /// Add context information (e.g., file path, operation name)
    ///
    /// Shortcut for `.with_field("context", ...)`.
    pub fn with_context(self, context: impl Into<String>) -> Self {
        self.with_field("context", serde_json::json!(context.into()))
    }
}

impl From<i64> for RequestId {
    fn from(n: i64) -> Self {
        RequestId::Number(n)
    }
}

impl From<String> for RequestId {
    fn from(s: String) -> Self {
        RequestId::String(s)
    }
}

impl From<&str> for RequestId {
    fn from(s: &str) -> Self {
        RequestId::String(s.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_serialization() {
        let request = Request::new("test.method", Some(serde_json::json!({"key": "value"})), 1.into());
        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("\"jsonrpc\":\"2.0\""));
        assert!(json.contains("\"method\":\"test.method\""));
        assert!(json.contains("\"id\":1"));
    }

    #[test]
    fn test_response_success() {
        let response = Response::success(1.into(), serde_json::json!({"result": "ok"}));
        assert!(response.is_success());
        assert!(!response.is_error());
        assert!(response.is_valid());
    }

    #[test]
    fn test_response_error() {
        let response = Response::error(1.into(), RpcError::internal_error("test error"));
        assert!(!response.is_success());
        assert!(response.is_error());
        assert!(response.is_valid());
    }

    #[test]
    fn test_response_invalid() {
        // Both result and error set - invalid
        let response = Response {
            jsonrpc: JSONRPC_VERSION.to_string(),
            result: Some(serde_json::json!({})),
            error: Some(RpcError::internal_error("error")),
            id: 1.into(),
        };
        assert!(!response.is_valid());

        // Neither result nor error set - also invalid
        let response = Response {
            jsonrpc: JSONRPC_VERSION.to_string(),
            result: None,
            error: None,
            id: 1.into(),
        };
        assert!(!response.is_valid());
    }

    #[test]
    fn test_notification_progress() {
        let notification = Notification::progress(Some(50), "Processing...");
        assert_eq!(notification.method, methods::NOTIFY_PROGRESS);
        assert!(notification.params.is_some());
    }

    #[test]
    fn test_notification_log() {
        let notification = Notification::log("info", "Test message");
        assert_eq!(notification.method, methods::NOTIFY_LOG);
        assert!(notification.params.is_some());
    }

    #[test]
    fn test_rpc_error_factories() {
        let err = RpcError::method_not_found("test.method");
        assert_eq!(err.code, error_codes::METHOD_NOT_FOUND);
        assert!(err.message.contains("test.method"));

        let err = RpcError::file_not_found("/path/to/file");
        assert_eq!(err.code, error_codes::FILE_NOT_FOUND);
        assert!(err.message.contains("/path/to/file"));

        let err = RpcError::not_supported("feature");
        assert_eq!(err.code, error_codes::NOT_SUPPORTED);
        assert!(err.message.contains("feature"));

        let err = RpcError::cancelled();
        assert_eq!(err.code, error_codes::REQUEST_CANCELLED);
    }

    #[test]
    fn test_request_id_conversions() {
        let id: RequestId = 42i64.into();
        assert!(matches!(id, RequestId::Number(42)));

        let id: RequestId = "test-id".into();
        assert!(matches!(id, RequestId::String(s) if s == "test-id"));

        let id: RequestId = String::from("test-id").into();
        assert!(matches!(id, RequestId::String(s) if s == "test-id"));
    }

    #[test]
    fn test_json_roundtrip() {
        let request = Request::new("test", Some(serde_json::json!({"x": 1})), 1.into());
        let json = serde_json::to_string(&request).unwrap();
        let parsed: Request = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.method, "test");
        assert_eq!(parsed.id, RequestId::Number(1));
    }

    #[test]
    fn test_request_new_checked_valid() {
        let result = Request::new_checked("backend.run", None, 1.into());
        assert!(result.is_ok());

        let result = Request::new_checked("$/ping", None, 1.into());
        assert!(result.is_ok());

        let result = Request::new_checked("format.load_model", None, 1.into());
        assert!(result.is_ok());
    }

    #[test]
    fn test_request_new_checked_empty() {
        let result = Request::new_checked("", None, 1.into());
        assert_eq!(result.unwrap_err(), RequestValidationError::EmptyMethod);

        let result = Request::new_checked("   ", None, 1.into());
        assert_eq!(result.unwrap_err(), RequestValidationError::EmptyMethod);
    }

    #[test]
    fn test_request_new_checked_invalid_chars() {
        let result = Request::new_checked("method with spaces", None, 1.into());
        assert_eq!(result.unwrap_err(), RequestValidationError::InvalidMethodChars);

        let result = Request::new_checked("method\nwith\nnewlines", None, 1.into());
        assert_eq!(result.unwrap_err(), RequestValidationError::InvalidMethodChars);
    }

    #[test]
    fn test_request_is_valid() {
        let valid = Request::new("test", None, 1.into());
        assert!(valid.is_valid());

        let invalid = Request::new("", None, 1.into());
        assert!(!invalid.is_valid());
    }

    #[test]
    fn test_response_validate() {
        // Valid success
        let response = Response::success(1.into(), serde_json::json!({}));
        assert!(response.validate().is_ok());

        // Valid error
        let response = Response::error(1.into(), RpcError::internal_error("error"));
        assert!(response.validate().is_ok());

        // Invalid: both set
        let response = Response {
            jsonrpc: JSONRPC_VERSION.to_string(),
            result: Some(serde_json::json!({})),
            error: Some(RpcError::internal_error("error")),
            id: 1.into(),
        };
        assert_eq!(
            response.validate().unwrap_err(),
            ResponseValidationError::BothResultAndError
        );

        // Invalid: neither set
        let response = Response {
            jsonrpc: JSONRPC_VERSION.to_string(),
            result: None,
            error: None,
            id: 1.into(),
        };
        assert_eq!(
            response.validate().unwrap_err(),
            ResponseValidationError::NeitherResultNorError
        );
    }

    // =========================================================================
    // Param Validation Tests
    // =========================================================================

    #[test]
    fn test_load_model_params_validate() {
        // Valid path
        let params = LoadModelParams {
            path: "/path/to/model.onnx".to_string(),
        };
        assert!(params.validate().is_ok());

        // Empty path
        let params = LoadModelParams { path: "".to_string() };
        assert!(params.validate().is_err());

        // Path traversal
        let params = LoadModelParams {
            path: "/path/../etc/passwd".to_string(),
        };
        assert!(params.validate().is_err());

        // URL encoded traversal
        let params = LoadModelParams {
            path: "/path/%2e%2e/etc/passwd".to_string(),
        };
        assert!(params.validate().is_err());
    }

    #[test]
    fn test_save_model_params_validate() {
        // Valid paths
        let params = SaveModelParams {
            snapshot_path: "/path/to/snapshot.hdss".to_string(),
            output_path: "/output/model.onnx".to_string(),
        };
        assert!(params.validate().is_ok());

        // Empty snapshot path
        let params = SaveModelParams {
            snapshot_path: "".to_string(),
            output_path: "/output/model.onnx".to_string(),
        };
        assert!(params.validate().is_err());

        // Empty output path
        let params = SaveModelParams {
            snapshot_path: "/path/to/snapshot.hdss".to_string(),
            output_path: "".to_string(),
        };
        assert!(params.validate().is_err());
    }

    #[test]
    fn test_run_params_validate() {
        // Valid params
        let params = RunParams {
            library_path: "/path/to/lib.dylib".to_string(),
            snapshot_path: "/path/to/snapshot.hdss".to_string(),
            device: "cpu".to_string(),
            inputs: vec![],
        };
        assert!(params.validate().is_ok());

        // Empty device
        let params = RunParams {
            library_path: "/path/to/lib.dylib".to_string(),
            snapshot_path: "/path/to/snapshot.hdss".to_string(),
            device: "".to_string(),
            inputs: vec![],
        };
        assert!(params.validate().is_err());

        // Too many inputs
        let params = RunParams {
            library_path: "/path/to/lib.dylib".to_string(),
            snapshot_path: "/path/to/snapshot.hdss".to_string(),
            device: "cpu".to_string(),
            inputs: (0..MAX_INPUTS + 1)
                .map(|i| TensorInput {
                    name: format!("input{}", i),
                    path: format!("/path/to/input{}.hdt", i),
                })
                .collect(),
        };
        assert!(params.validate().is_err());
    }

    #[test]
    fn test_build_params_validate() {
        // Valid params
        let params = BuildParams {
            snapshot_path: "/path/to/snapshot.hdss".to_string(),
            target: "x86_64-apple-darwin".to_string(),
            device: "cpu".to_string(),
            format: "sharedlib".to_string(),
            output_path: "/output/model.dylib".to_string(),
        };
        assert!(params.validate().is_ok());

        // Empty target
        let params = BuildParams {
            snapshot_path: "/path/to/snapshot.hdss".to_string(),
            target: "".to_string(),
            device: "cpu".to_string(),
            format: "sharedlib".to_string(),
            output_path: "/output/model.dylib".to_string(),
        };
        assert!(params.validate().is_err());

        // Empty format
        let params = BuildParams {
            snapshot_path: "/path/to/snapshot.hdss".to_string(),
            target: "x86_64-apple-darwin".to_string(),
            device: "cpu".to_string(),
            format: "".to_string(),
            output_path: "/output/model.dylib".to_string(),
        };
        assert!(params.validate().is_err());
    }

    #[test]
    fn test_tensor_input_validate() {
        // Valid input
        let input = TensorInput {
            name: "input0".to_string(),
            path: "/path/to/tensor.hdt".to_string(),
        };
        assert!(input.validate().is_ok());

        // Empty name
        let input = TensorInput {
            name: "".to_string(),
            path: "/path/to/tensor.hdt".to_string(),
        };
        assert!(input.validate().is_err());

        // Name with path separator
        let input = TensorInput {
            name: "input/0".to_string(),
            path: "/path/to/tensor.hdt".to_string(),
        };
        assert!(input.validate().is_err());

        // Empty path
        let input = TensorInput {
            name: "input0".to_string(),
            path: "".to_string(),
        };
        assert!(input.validate().is_err());
    }

    #[test]
    fn test_tensor_output_validate() {
        // Valid output
        let output = TensorOutput {
            name: "output0".to_string(),
            path: "/path/to/output.hdt".to_string(),
        };
        assert!(output.validate().is_ok());

        // Empty name
        let output = TensorOutput {
            name: "".to_string(),
            path: "/path/to/output.hdt".to_string(),
        };
        assert!(output.validate().is_err());

        // Name with path separator
        let output = TensorOutput {
            name: "output\\0".to_string(),
            path: "/path/to/output.hdt".to_string(),
        };
        assert!(output.validate().is_err());
    }

    #[test]
    fn test_initialize_params_validate() {
        // Valid params
        let params = InitializeParams {
            plugin_version: "1.0.0".to_string(),
            protocol_version: "1.0.0".to_string(),
        };
        assert!(params.validate().is_ok());

        // Empty plugin version
        let params = InitializeParams {
            plugin_version: "".to_string(),
            protocol_version: "1.0.0".to_string(),
        };
        assert!(params.validate().is_err());

        // Empty protocol version
        let params = InitializeParams {
            plugin_version: "1.0.0".to_string(),
            protocol_version: "".to_string(),
        };
        assert!(params.validate().is_err());
    }

    #[test]
    fn test_plugin_metadata_validate() {
        // Valid metadata
        let metadata = PluginMetadataRpc {
            description: Some("A test plugin".to_string()),
            author: Some("Test Author".to_string()),
            homepage: Some("https://example.com".to_string()),
            license: Some("MIT".to_string()),
            repository: Some("https://github.com/test/repo".to_string()),
            supported_targets: Some(vec!["x86_64-*-*".to_string()]),
            min_hodu_version: Some("0.1.0".to_string()),
        };
        assert!(metadata.validate().is_ok());

        // Description too long
        let metadata = PluginMetadataRpc {
            description: Some("x".repeat(MAX_METADATA_DESCRIPTION_LEN + 1)),
            ..Default::default()
        };
        assert!(metadata.validate().is_err());

        // Author too long
        let metadata = PluginMetadataRpc {
            author: Some("x".repeat(MAX_METADATA_AUTHOR_LEN + 1)),
            ..Default::default()
        };
        assert!(metadata.validate().is_err());

        // Too many targets
        let metadata = PluginMetadataRpc {
            supported_targets: Some((0..MAX_SUPPORTED_TARGETS + 1).map(|i| format!("target{}", i)).collect()),
            ..Default::default()
        };
        assert!(metadata.validate().is_err());
    }

    #[test]
    fn test_plugin_metadata_sanitize() {
        let mut metadata = PluginMetadataRpc {
            description: Some("x".repeat(MAX_METADATA_DESCRIPTION_LEN + 100)),
            author: Some("x".repeat(MAX_METADATA_AUTHOR_LEN + 100)),
            license: Some("x".repeat(MAX_METADATA_LICENSE_LEN + 100)),
            ..Default::default()
        };

        metadata.sanitize();

        assert!(metadata.description.as_ref().unwrap().len() <= MAX_METADATA_DESCRIPTION_LEN);
        assert!(metadata.author.as_ref().unwrap().len() <= MAX_METADATA_AUTHOR_LEN);
        assert!(metadata.license.as_ref().unwrap().len() <= MAX_METADATA_LICENSE_LEN);
    }
}
