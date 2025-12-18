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

/// Validate a path string (non-empty, no null bytes)
fn validate_path(path: &str, field: &str) -> Result<(), ValidationError> {
    if path.is_empty() {
        return Err(ValidationError {
            field: field.to_string(),
            message: "path cannot be empty".to_string(),
        });
    }
    if path.contains('\0') {
        return Err(ValidationError {
            field: field.to_string(),
            message: "path contains null byte".to_string(),
        });
    }
    Ok(())
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

/// Request ID (can be number or string)
///
/// Per JSON-RPC 2.0 spec, IDs can be numbers, strings, or null.
/// We don't support null IDs as they indicate notifications.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(untagged)]
pub enum RequestId {
    /// Numeric request ID
    Number(i64),
    /// String request ID
    String(String),
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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InitializeParams {
    /// CLI's plugin protocol version
    pub plugin_version: String,
    /// JSON-RPC protocol version (should be "1.0.0")
    pub protocol_version: String,
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
            if let Some(ref targets) = meta.supported_targets {
                if targets.len() > MAX_SUPPORTED_TARGETS {
                    return Err(ValidationError {
                        field: "metadata.supported_targets".to_string(),
                        message: format!(
                            "too many supported targets ({} > {})",
                            targets.len(),
                            MAX_SUPPORTED_TARGETS
                        ),
                    });
                }
            }
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
        !self.name.is_empty()
            && !self.name.chars().any(|c| c.is_control())
            && !self.name.contains('/')
            && !self.name.contains('\\')
    }

    /// Validate the tensor input
    pub fn validate(&self) -> Result<(), ValidationError> {
        if !self.is_valid_name() {
            return Err(ValidationError {
                field: "name".to_string(),
                message: "invalid tensor name (empty, contains control chars, or path separators)".to_string(),
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

    /// Validate tensor name (non-empty, no control chars, no path separators)
    pub fn is_valid_name(&self) -> bool {
        !self.name.is_empty()
            && !self.name.chars().any(|c| c.is_control())
            && !self.name.contains('/')
            && !self.name.contains('\\')
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
    pub fn is_valid(&self) -> bool {
        !self.method.is_empty() && !self.method.chars().all(|c| c.is_whitespace())
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
    pub fn progress(percent: Option<u8>, message: impl Into<String>) -> Self {
        // Clamp percent to valid range 0-100
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
    /// # Example
    ///
    /// ```ignore
    /// let error = RpcError::internal_error("Failed to process")
    ///     .with_cause("IO error: file not found");
    /// ```
    pub fn with_cause(mut self, cause: impl Into<String>) -> Self {
        let cause_str = cause.into();
        self.data = Some(match self.data {
            Some(mut data) => {
                if let Some(obj) = data.as_object_mut() {
                    // Chain causes if already exists
                    if let Some(existing) = obj.get("cause") {
                        obj.insert(
                            "cause".to_string(),
                            serde_json::json!(format!("{} <- {}", cause_str, existing)),
                        );
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
    /// # Example
    ///
    /// ```ignore
    /// let error = RpcError::file_not_found("/path/to/model.onnx")
    ///     .with_hint("Check if the file path is correct")
    ///     .with_hint("Ensure the file exists and is readable");
    /// ```
    pub fn with_hint(mut self, hint: impl Into<String>) -> Self {
        let hint_str = hint.into();
        self.data = Some(match self.data {
            Some(mut data) => {
                if let Some(obj) = data.as_object_mut() {
                    if let Some(hints) = obj.get_mut("hints") {
                        if let Some(arr) = hints.as_array_mut() {
                            arr.push(serde_json::json!(hint_str));
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
}
