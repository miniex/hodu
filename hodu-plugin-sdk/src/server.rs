//! Plugin server framework for JSON-RPC communication over stdio
//!
//! This module provides the runtime for plugins to handle JSON-RPC requests.
//!
//! # Example
//!
//! ```ignore
//! use hodu_plugin_sdk::server::PluginServer;
//! use hodu_plugin_sdk::rpc::{RunParams, RunResult, RpcError};
//! use hodu_plugin_sdk::Context;
//!
//! async fn handle_run(ctx: Context, params: RunParams) -> Result<RunResult, RpcError> {
//!     if ctx.is_cancelled() {
//!         return Err(RpcError::cancelled());
//!     }
//!     // ...
//! }
//!
//! #[tokio::main]
//! async fn main() {
//!     PluginServer::new("my-plugin", env!("CARGO_PKG_VERSION"))
//!         .devices(vec!["cpu"])
//!         .method("backend.run", handle_run)
//!         .run()
//!         .await
//!         .unwrap();
//! }
//! ```

use crate::context::{CancellationHandle, Context};
use crate::rpc::{
    error_codes, methods, CancelParams, InitializeParams, InitializeResult, Notification, PluginMetadataRpc, Request,
    RequestId, Response, RpcError, PROTOCOL_VERSION,
};
use crate::PLUGIN_VERSION;
use serde::{de::DeserializeOwned, Serialize};
use std::collections::HashMap;
use std::future::Future;
use std::io::{BufRead, BufReader, Write};
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;

/// Maximum allowed request size (1MB)
const MAX_REQUEST_SIZE: usize = 1024 * 1024;

/// Maximum allowed batch request count (prevents DoS)
const MAX_BATCH_SIZE: usize = 100;

/// Default request timeout (5 minutes)
const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(300);

/// Reserved method name prefixes that plugins cannot register
const RESERVED_PREFIXES: &[&str] = &["$/", "rpc.", "system."];

// ============================================================================
// Param Deserialization Helper
// ============================================================================

/// Deserialize params from JSON-RPC request with proper error handling
///
/// Returns `RpcError::invalid_params` if params are missing or malformed.
fn deserialize_params<P: DeserializeOwned>(params: Option<serde_json::Value>) -> Result<P, RpcError> {
    match params {
        Some(v) => serde_json::from_value(v).map_err(|e| RpcError::invalid_params(e.to_string())),
        None => Err(RpcError::invalid_params("Missing params")),
    }
}

/// Try to deserialize params, returning None if params are missing or invalid
///
/// Useful for optional/notification handlers where errors should be silently ignored.
fn try_deserialize_params<P: DeserializeOwned>(params: Option<serde_json::Value>) -> Option<P> {
    params.and_then(|v| serde_json::from_value(v).ok())
}

// ============================================================================
// Plugin Metadata
// ============================================================================

/// Plugin metadata for registration and discovery
#[derive(Clone, Default)]
pub struct PluginMetadata {
    /// Plugin description
    pub description: Option<String>,
    /// Author name or organization
    pub author: Option<String>,
    /// Plugin homepage URL
    pub homepage: Option<String>,
    /// License identifier (e.g., "MIT", "Apache-2.0")
    pub license: Option<String>,
    /// Source repository URL
    pub repository: Option<String>,
    /// Supported target triples (e.g., "x86_64-*-*", "aarch64-apple-darwin")
    pub supported_targets: Option<Vec<String>>,
    /// Minimum required hodu version (semver, e.g., "0.1.0")
    pub min_hodu_version: Option<String>,
}

// ============================================================================
// Shutdown Handler
// ============================================================================

/// Type for shutdown cleanup callback
type ShutdownCallback = Box<dyn FnOnce() + Send + 'static>;

// ============================================================================
// Middleware/Hooks
// ============================================================================

/// Information about an incoming request (for hooks)
#[derive(Clone)]
pub struct RequestInfo {
    /// The method being called
    pub method: String,
    /// The request ID
    pub id: RequestId,
    /// The raw params (if any)
    pub params: Option<serde_json::Value>,
}

/// Information about an outgoing response (for hooks)
#[derive(Clone)]
pub struct ResponseInfo {
    /// The method that was called
    pub method: String,
    /// The request ID
    pub id: RequestId,
    /// Whether the request succeeded
    pub success: bool,
    /// Error code if failed
    pub error_code: Option<i32>,
    /// Execution duration
    pub duration: std::time::Duration,
}

/// Pre-request hook return value
pub enum PreRequestAction {
    /// Continue processing the request normally
    Continue,
    /// Skip the handler and return this error
    Reject(RpcError),
}

/// Type for pre-request hook
type PreRequestHook = Box<dyn Fn(&RequestInfo) -> PreRequestAction + Send + Sync>;

/// Type for post-request hook
type PostRequestHook = Box<dyn Fn(&ResponseInfo) + Send + Sync>;

// ============================================================================
// Debug/Profiling Options
// ============================================================================

/// Debug logging options for development
#[derive(Clone, Default)]
pub struct DebugOptions {
    /// Log all incoming requests to stderr
    pub log_requests: bool,
    /// Log all outgoing responses to stderr
    pub log_responses: bool,
    /// Log handler execution times to stderr
    pub log_profiling: bool,
}

impl DebugOptions {
    /// Enable all debug logging
    pub fn all() -> Self {
        Self {
            log_requests: true,
            log_responses: true,
            log_profiling: true,
        }
    }

    /// Enable only profiling (execution time logging)
    pub fn profiling_only() -> Self {
        Self {
            log_requests: false,
            log_responses: false,
            log_profiling: true,
        }
    }
}

// ============================================================================
// Notification helpers (can be called from handlers)
// ============================================================================

/// Send a progress notification to the CLI (fire-and-forget)
///
/// # Arguments
/// * `percent` - Progress percentage (0-100), None for indeterminate. Values > 100 are clamped to 100.
/// * `message` - Progress message
pub fn notify_progress(percent: Option<u8>, message: &str) {
    if let Err(e) = try_notify_progress(percent, message) {
        eprintln!("Warning: Failed to send progress notification: {}", e);
    }
}

/// Send a progress notification to the CLI with error handling
///
/// Returns an error if the notification fails to send, allowing the caller to handle it.
pub fn try_notify_progress(percent: Option<u8>, message: &str) -> Result<(), std::io::Error> {
    use std::io::Write;
    // Clamp percent to 0-100
    let percent = percent.map(|p| p.min(100));
    let notification = Notification::progress(percent, message);
    let json =
        serde_json::to_string(&notification).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    writeln!(std::io::stdout(), "{}", json)?;
    std::io::stdout().flush()
}

/// Valid log levels
const VALID_LOG_LEVELS: &[&str] = &["error", "warn", "info", "debug", "trace"];

/// Send a log notification to the CLI (fire-and-forget)
///
/// # Arguments
/// * `level` - Log level: "error", "warn", "info", "debug", "trace". Invalid levels default to "info".
/// * `message` - Log message
pub fn notify_log(level: &str, message: &str) {
    if let Err(e) = try_notify_log(level, message) {
        eprintln!("Warning: Failed to send log notification: {}", e);
    }
}

/// Send a log notification to the CLI with error handling
///
/// Returns an error if the notification fails to send, allowing the caller to handle it.
pub fn try_notify_log(level: &str, message: &str) -> Result<(), std::io::Error> {
    use std::io::Write;
    // Validate log level, default to "info" if invalid
    let level = if VALID_LOG_LEVELS.contains(&level) {
        level
    } else {
        "info"
    };
    let notification = Notification::log(level, message);
    let json =
        serde_json::to_string(&notification).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    writeln!(std::io::stdout(), "{}", json)?;
    std::io::stdout().flush()
}

/// Convenience functions for different log levels
pub fn log_error(message: &str) {
    notify_log("error", message);
}
pub fn log_warn(message: &str) {
    notify_log("warn", message);
}
pub fn log_info(message: &str) {
    notify_log("info", message);
}
pub fn log_debug(message: &str) {
    notify_log("debug", message);
}

// ============================================================================
// Streaming Response Support
// ============================================================================

/// A streaming response writer for sending chunked data
///
/// Allows sending data in chunks during long-running operations,
/// using JSON-RPC notifications as the transport mechanism.
///
/// # Example
///
/// ```ignore
/// async fn handle_export(ctx: Context, params: ExportParams) -> Result<ExportResult, RpcError> {
///     let mut stream = StreamWriter::new("export.chunk");
///
///     for chunk in data.chunks(1024) {
///         stream.write_chunk(chunk)?;
///     }
///
///     stream.finish()?;
///     Ok(ExportResult { /* ... */ })
/// }
/// ```
pub struct StreamWriter {
    method: String,
    chunk_index: usize,
}

impl StreamWriter {
    /// Create a new stream writer
    ///
    /// # Arguments
    /// * `method` - Notification method name for chunks (e.g., "export.chunk")
    pub fn new(method: impl Into<String>) -> Self {
        Self {
            method: method.into(),
            chunk_index: 0,
        }
    }

    /// Write a chunk of bytes
    pub fn write_chunk(&mut self, data: &[u8]) -> Result<(), std::io::Error> {
        self.write_chunk_with_metadata(data, None)
    }

    /// Write a chunk with optional metadata
    pub fn write_chunk_with_metadata(
        &mut self,
        data: &[u8],
        metadata: Option<serde_json::Value>,
    ) -> Result<(), std::io::Error> {
        use std::io::Write;

        let encoded = base64_encode(data);
        let mut params = serde_json::json!({
            "index": self.chunk_index,
            "data": encoded,
            "size": data.len(),
        });

        if let Some(meta) = metadata {
            if let Some(obj) = params.as_object_mut() {
                obj.insert("metadata".to_string(), meta);
            }
        }

        let notification = Notification::new(&self.method, Some(params));
        let json = serde_json::to_string(&notification).map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Failed to serialize chunk {}: {}", self.chunk_index, e),
            )
        })?;

        writeln!(std::io::stdout(), "{}", json)?;
        std::io::stdout().flush()?;

        self.chunk_index += 1;
        Ok(())
    }

    /// Write a JSON chunk (for structured data streaming)
    pub fn write_json(&mut self, value: &serde_json::Value) -> Result<(), std::io::Error> {
        use std::io::Write;

        let params = serde_json::json!({
            "index": self.chunk_index,
            "json": value,
        });

        let notification = Notification::new(&self.method, Some(params));
        let json = serde_json::to_string(&notification).map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Failed to serialize JSON chunk {}: {}", self.chunk_index, e),
            )
        })?;

        writeln!(std::io::stdout(), "{}", json)?;
        std::io::stdout().flush()?;

        self.chunk_index += 1;
        Ok(())
    }

    /// Signal that streaming is complete
    pub fn finish(&self) -> Result<(), std::io::Error> {
        use std::io::Write;

        let params = serde_json::json!({
            "finished": true,
            "total_chunks": self.chunk_index,
        });

        let notification = Notification::new(&self.method, Some(params));
        let json = serde_json::to_string(&notification).map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Failed to serialize finish notification: {}", e),
            )
        })?;

        writeln!(std::io::stdout(), "{}", json)?;
        std::io::stdout().flush()?;

        Ok(())
    }

    /// Get the number of chunks written so far
    pub fn chunks_written(&self) -> usize {
        self.chunk_index
    }
}

/// Simple base64 encoding for chunk data
fn base64_encode(data: &[u8]) -> String {
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    let mut result = Vec::with_capacity(data.len().div_ceil(3) * 4);

    for chunk in data.chunks(3) {
        let b0 = chunk[0] as usize;
        let b1 = chunk.get(1).copied().unwrap_or(0) as usize;
        let b2 = chunk.get(2).copied().unwrap_or(0) as usize;

        result.push(ALPHABET[b0 >> 2]);
        result.push(ALPHABET[((b0 & 0x03) << 4) | (b1 >> 4)]);

        if chunk.len() > 1 {
            result.push(ALPHABET[((b1 & 0x0f) << 2) | (b2 >> 6)]);
        } else {
            result.push(b'=');
        }

        if chunk.len() > 2 {
            result.push(ALPHABET[b2 & 0x3f]);
        } else {
            result.push(b'=');
        }
    }

    // SAFETY: base64 encoding only produces ASCII characters, so this is always valid UTF-8
    String::from_utf8(result).expect("base64 encoding produces only ASCII characters")
}

// ============================================================================
// Handler Types
// ============================================================================

/// Type-erased async handler function
type BoxedHandlerFn = Box<
    dyn Fn(
            Context,
            Option<serde_json::Value>,
        ) -> Pin<Box<dyn Future<Output = Result<serde_json::Value, RpcError>> + Send>>
        + Send
        + Sync,
>;

/// Handler with optional timeout
struct Handler {
    func: BoxedHandlerFn,
    timeout: Option<Duration>,
}

// ============================================================================
// Plugin Server
// ============================================================================

/// Plugin server that handles JSON-RPC requests over stdio
///
/// # Example
///
/// ```ignore
/// use hodu_plugin_sdk::server::PluginServer;
/// use hodu_plugin_sdk::rpc::{RunParams, RunResult, RpcError};
/// use hodu_plugin_sdk::Context;
///
/// async fn handle_run(ctx: Context, params: RunParams) -> Result<RunResult, RpcError> {
///     for i in 0..10 {
///         if ctx.is_cancelled() {
///             return Err(RpcError::cancelled());
///         }
///         ctx.progress(Some(i * 10), "Processing...");
///         // do work...
///     }
///     Ok(RunResult { outputs: vec![] })
/// }
///
/// #[tokio::main]
/// async fn main() {
///     PluginServer::new("my-plugin", env!("CARGO_PKG_VERSION"))
///         .devices(vec!["cpu"])
///         .method("backend.run", handle_run)
///         .run()
///         .await
///         .unwrap();
/// }
/// ```
pub struct PluginServer {
    name: String,
    version: String,
    capabilities: Vec<String>,
    model_extensions: Option<Vec<String>>,
    tensor_extensions: Option<Vec<String>>,
    devices: Option<Vec<String>>,
    handlers: HashMap<String, Handler>,
    initialized: bool,
    /// Active requests that can be cancelled
    active_requests: Arc<Mutex<HashMap<RequestId, CancellationHandle>>>,
    /// Plugin metadata
    metadata: PluginMetadata,
    /// Shutdown cleanup callback
    shutdown_callback: Option<ShutdownCallback>,
    /// Shared state across handlers
    state: Option<Arc<dyn std::any::Any + Send + Sync>>,
    /// Default timeout for handlers (None = no timeout)
    default_timeout: Option<Duration>,
    /// Pre-request hook
    pre_request_hook: Option<PreRequestHook>,
    /// Post-request hook
    post_request_hook: Option<PostRequestHook>,
    /// Debug/profiling options
    debug_options: DebugOptions,
    /// Build-time validation errors (reported on run())
    build_errors: Vec<String>,
    /// Flag to indicate graceful shutdown was requested
    shutdown_requested: bool,
}

impl PluginServer {
    /// Create a new plugin server
    pub fn new(name: impl Into<String>, version: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            version: version.into(),
            capabilities: Vec::new(),
            model_extensions: None,
            tensor_extensions: None,
            devices: None,
            handlers: HashMap::new(),
            initialized: false,
            active_requests: Arc::new(Mutex::new(HashMap::new())),
            metadata: PluginMetadata::default(),
            shutdown_callback: None,
            state: None,
            default_timeout: Some(DEFAULT_REQUEST_TIMEOUT),
            pre_request_hook: None,
            post_request_hook: None,
            debug_options: DebugOptions::default(),
            build_errors: Vec::new(),
            shutdown_requested: false,
        }
    }

    /// Enable debug logging for development
    ///
    /// Logs requests, responses, and/or profiling info to stderr.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Enable all debug logging
    /// PluginServer::new("my-plugin", "1.0.0")
    ///     .debug(DebugOptions::all())
    ///     .run()
    ///     .await
    ///
    /// // Enable only profiling
    /// PluginServer::new("my-plugin", "1.0.0")
    ///     .debug(DebugOptions::profiling_only())
    ///     .run()
    ///     .await
    /// ```
    pub fn debug(mut self, options: DebugOptions) -> Self {
        self.debug_options = options;
        self
    }

    /// Enable automatic profiling (execution time measurement)
    ///
    /// Logs handler execution times to stderr.
    /// Shortcut for `.debug(DebugOptions::profiling_only())`.
    pub fn enable_profiling(mut self) -> Self {
        self.debug_options.log_profiling = true;
        self
    }

    /// Set default timeout for all handlers
    ///
    /// Handlers will be automatically cancelled if they exceed this duration.
    ///
    /// # Example
    ///
    /// ```ignore
    /// PluginServer::new("my-plugin", "1.0.0")
    ///     .timeout(Duration::from_secs(30))
    ///     .method("backend.run", handler)
    ///     .run()
    ///     .await
    /// ```
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.default_timeout = Some(timeout);
        self
    }

    /// Set shared state that will be available to all handlers
    ///
    /// The state is wrapped in an `Arc` and can be accessed via `ctx.state::<T>()` in handlers.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use std::sync::atomic::{AtomicU64, Ordering};
    ///
    /// struct MyState {
    ///     request_count: AtomicU64,
    /// }
    ///
    /// async fn handler(ctx: Context, params: Params) -> Result<Response, RpcError> {
    ///     if let Some(state) = ctx.state::<MyState>() {
    ///         let count = state.request_count.fetch_add(1, Ordering::SeqCst);
    ///         ctx.log_info(&format!("Request #{}", count));
    ///     }
    ///     // ...
    /// }
    ///
    /// #[tokio::main]
    /// async fn main() {
    ///     let state = MyState {
    ///         request_count: AtomicU64::new(0),
    ///     };
    ///
    ///     PluginServer::new("my-plugin", "1.0.0")
    ///         .with_state(state)
    ///         .method("backend.run", handler)
    ///         .run()
    ///         .await
    ///         .unwrap();
    /// }
    /// ```
    pub fn with_state<S: Send + Sync + 'static>(mut self, state: S) -> Self {
        self.state = Some(Arc::new(state));
        self
    }

    /// Set plugin description
    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.metadata.description = Some(desc.into());
        self
    }

    /// Set plugin author
    pub fn author(mut self, author: impl Into<String>) -> Self {
        self.metadata.author = Some(author.into());
        self
    }

    /// Set plugin homepage URL
    pub fn homepage(mut self, url: impl Into<String>) -> Self {
        self.metadata.homepage = Some(url.into());
        self
    }

    /// Set plugin license
    pub fn license(mut self, license: impl Into<String>) -> Self {
        self.metadata.license = Some(license.into());
        self
    }

    /// Set plugin repository URL
    pub fn repository(mut self, url: impl Into<String>) -> Self {
        self.metadata.repository = Some(url.into());
        self
    }

    /// Set supported target triples (OS/arch patterns)
    ///
    /// Patterns can use wildcards: "x86_64-*-*", "aarch64-apple-darwin"
    ///
    /// # Example
    ///
    /// ```ignore
    /// PluginServer::new("my-plugin", "1.0.0")
    ///     .supported_targets(vec!["x86_64-*-*", "aarch64-apple-darwin"])
    ///     .run()
    ///     .await
    /// ```
    pub fn supported_targets(mut self, targets: Vec<&str>) -> Self {
        self.metadata.supported_targets = Some(targets.into_iter().map(String::from).collect());
        self
    }

    /// Set minimum required hodu version (semver)
    ///
    /// # Example
    ///
    /// ```ignore
    /// PluginServer::new("my-plugin", "1.0.0")
    ///     .min_hodu_version("0.1.0")
    ///     .run()
    ///     .await
    /// ```
    pub fn min_hodu_version(mut self, version: impl Into<String>) -> Self {
        self.metadata.min_hodu_version = Some(version.into());
        self
    }

    /// Set shutdown cleanup callback
    ///
    /// The callback is called before the server exits (on `$/shutdown` request).
    ///
    /// # Example
    ///
    /// ```ignore
    /// PluginServer::new("my-plugin", "1.0.0")
    ///     .on_shutdown(|| {
    ///         println!("Cleaning up resources...");
    ///         // Close connections, save state, etc.
    ///     })
    ///     .run()
    ///     .await
    /// ```
    pub fn on_shutdown<F>(mut self, callback: F) -> Self
    where
        F: FnOnce() + Send + 'static,
    {
        self.shutdown_callback = Some(Box::new(callback));
        self
    }

    /// Set pre-request hook (called before each handler)
    ///
    /// Can be used for logging, authentication, rate limiting, etc.
    /// Return `PreRequestAction::Reject(error)` to skip the handler.
    ///
    /// # Example
    ///
    /// ```ignore
    /// PluginServer::new("my-plugin", "1.0.0")
    ///     .on_request(|req| {
    ///         println!("Received: {} (id: {:?})", req.method, req.id);
    ///         PreRequestAction::Continue
    ///     })
    ///     .run()
    ///     .await
    /// ```
    pub fn on_request<F>(mut self, hook: F) -> Self
    where
        F: Fn(&RequestInfo) -> PreRequestAction + Send + Sync + 'static,
    {
        self.pre_request_hook = Some(Box::new(hook));
        self
    }

    /// Set post-request hook (called after each handler)
    ///
    /// Can be used for logging, metrics, profiling, etc.
    ///
    /// # Example
    ///
    /// ```ignore
    /// PluginServer::new("my-plugin", "1.0.0")
    ///     .on_response(|resp| {
    ///         println!("{} completed in {:?} (success: {})",
    ///             resp.method, resp.duration, resp.success);
    ///     })
    ///     .run()
    ///     .await
    /// ```
    pub fn on_response<F>(mut self, hook: F) -> Self
    where
        F: Fn(&ResponseInfo) + Send + Sync + 'static,
    {
        self.post_request_hook = Some(Box::new(hook));
        self
    }

    /// Set supported file extensions for model format plugins
    pub fn model_extensions(mut self, exts: Vec<&str>) -> Self {
        self.model_extensions = Some(exts.into_iter().map(String::from).collect());
        self
    }

    /// Set supported file extensions for tensor format plugins
    pub fn tensor_extensions(mut self, exts: Vec<&str>) -> Self {
        self.tensor_extensions = Some(exts.into_iter().map(String::from).collect());
        self
    }

    /// Set supported devices (for backend plugins)
    pub fn devices(mut self, devs: Vec<&str>) -> Self {
        self.devices = Some(devs.into_iter().map(String::from).collect());
        self
    }

    /// Register an async method handler with context
    ///
    /// The handler receives a `Context` for cancellation support.
    ///
    /// # Example
    ///
    /// ```ignore
    /// async fn handle_run(ctx: Context, params: RunParams) -> Result<RunResult, RpcError> {
    ///     if ctx.is_cancelled() {
    ///         return Err(RpcError::cancelled());
    ///     }
    ///     // ...
    /// }
    ///
    /// server.method("backend.run", handle_run)
    /// ```
    pub fn method<F, Fut, P, R>(self, name: &str, handler: F) -> Self
    where
        F: Fn(Context, P) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<R, RpcError>> + Send + 'static,
        P: DeserializeOwned + Send + 'static,
        R: Serialize + 'static,
    {
        let handler = Arc::new(handler);
        let boxed: BoxedHandlerFn = Box::new(move |ctx, params| {
            let handler = handler.clone();
            Box::pin(async move {
                let params: P = deserialize_params(params)?;
                let result = handler(ctx, params).await?;
                serde_json::to_value(result).map_err(|e| RpcError::internal_error(e.to_string()))
            })
        });
        self.register_handler(name, boxed, None)
    }

    /// Register an async method handler with custom timeout
    ///
    /// Overrides the default timeout for this specific method.
    ///
    /// # Example
    ///
    /// ```ignore
    /// server
    ///     .timeout(Duration::from_secs(30))  // default
    ///     .method_with_timeout("backend.run", handler, Duration::from_secs(120))  // override
    /// ```
    pub fn method_with_timeout<F, Fut, P, R>(self, name: &str, handler: F, timeout: Duration) -> Self
    where
        F: Fn(Context, P) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<R, RpcError>> + Send + 'static,
        P: DeserializeOwned + Send + 'static,
        R: Serialize + 'static,
    {
        let handler = Arc::new(handler);
        let boxed: BoxedHandlerFn = Box::new(move |ctx, params| {
            let handler = handler.clone();
            Box::pin(async move {
                let params: P = deserialize_params(params)?;
                let result = handler(ctx, params).await?;
                serde_json::to_value(result).map_err(|e| RpcError::internal_error(e.to_string()))
            })
        });
        self.register_handler(name, boxed, Some(timeout))
    }

    /// Register an async method handler without params
    pub fn method_no_params<F, Fut, R>(self, name: &str, handler: F) -> Self
    where
        F: Fn(Context) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<R, RpcError>> + Send + 'static,
        R: Serialize + 'static,
    {
        let handler = Arc::new(handler);
        let boxed: BoxedHandlerFn = Box::new(move |ctx, _params| {
            let handler = handler.clone();
            Box::pin(async move {
                let result = handler(ctx).await?;
                serde_json::to_value(result).map_err(|e| RpcError::internal_error(e.to_string()))
            })
        });
        self.register_handler(name, boxed, None)
    }

    /// Internal helper to register a handler with capability tracking
    ///
    /// Validation errors are collected and reported when `run()` is called.
    /// Invalid handler names include:
    /// - Empty names
    /// - Names containing control characters
    /// - Names using reserved prefixes (`$/`, `rpc.`, `system.`)
    fn register_handler(mut self, name: &str, func: BoxedHandlerFn, timeout: Option<Duration>) -> Self {
        // Validate handler name - collect errors for reporting at run()
        if name.is_empty() {
            self.build_errors.push("Handler name cannot be empty".to_string());
            return self;
        }
        if name.chars().any(|c| c.is_control()) {
            self.build_errors
                .push(format!("Handler name '{}' contains control characters", name));
            return self;
        }
        if RESERVED_PREFIXES.iter().any(|p| name.starts_with(p)) {
            self.build_errors
                .push(format!("Handler name '{}' uses reserved prefix", name));
            return self;
        }

        // Warn on duplicate handler registration
        if self.handlers.contains_key(name) {
            eprintln!("Warning: Handler '{}' is being overwritten", name);
        }

        // Auto-register capability for format/backend methods
        if (name.starts_with("format.") || name.starts_with("backend."))
            && !self.capabilities.contains(&name.to_string())
        {
            self.capabilities.push(name.to_string());
        }
        self.handlers.insert(name.to_string(), Handler { func, timeout });
        self
    }

    /// Run the server
    ///
    /// Starts the JSON-RPC server loop, reading from stdin and writing to stdout.
    /// Supports cancellation via `$/cancel` requests and batch requests per JSON-RPC 2.0 spec.
    ///
    /// # Errors
    /// Returns error if there were validation errors during server construction
    /// (e.g., invalid handler names).
    pub async fn run(mut self) -> Result<(), Box<dyn std::error::Error>> {
        // Check for build-time validation errors
        if !self.build_errors.is_empty() {
            let errors = self.build_errors.join("; ");
            return Err(format!("Plugin server configuration errors: {}", errors).into());
        }

        let stdin = std::io::stdin();
        let mut stdout = std::io::stdout();
        let reader = BufReader::new(stdin.lock());

        for line in reader.lines() {
            let line = line?;
            if line.is_empty() {
                continue;
            }

            // Check request size limit
            if line.len() > MAX_REQUEST_SIZE {
                let resp = Response::error(
                    RequestId::Number(0),
                    RpcError::invalid_request(format!(
                        "Request too large: {} bytes (max: {} bytes)",
                        line.len(),
                        MAX_REQUEST_SIZE
                    )),
                );
                let json = serde_json::to_string(&resp)?;
                writeln!(stdout, "{}", json)?;
                stdout.flush()?;
                continue;
            }

            // Check if batch request (starts with '[')
            let trimmed = line.trim_start();
            if trimmed.starts_with('[') {
                // Batch request
                let responses = self.handle_batch(&line).await;
                if !responses.is_empty() {
                    let json = serde_json::to_string(&responses)?;
                    writeln!(stdout, "{}", json)?;
                    stdout.flush()?;
                }
            } else {
                // Single request
                let response = self.handle_message(&line).await;
                if let Some(resp) = response {
                    let json = serde_json::to_string(&resp)?;
                    writeln!(stdout, "{}", json)?;
                    stdout.flush()?;
                }
            }

            // Check for graceful shutdown after processing the request
            if self.shutdown_requested {
                break;
            }
        }

        Ok(())
    }

    /// Handle a batch of JSON-RPC requests
    async fn handle_batch(&mut self, line: &str) -> Vec<Response> {
        // Parse as array of requests
        let requests: Vec<serde_json::Value> = match serde_json::from_str(line) {
            Ok(reqs) => reqs,
            Err(e) => {
                return vec![Response::error(
                    RequestId::Number(0),
                    RpcError::parse_error(e.to_string()),
                )];
            },
        };

        if requests.is_empty() {
            return vec![Response::error(
                RequestId::Number(0),
                RpcError::invalid_request("Empty batch"),
            )];
        }

        // Check batch size limit to prevent DoS
        if requests.len() > MAX_BATCH_SIZE {
            return vec![Response::error(
                RequestId::Number(0),
                RpcError::invalid_request(format!(
                    "Batch too large: {} requests (max: {})",
                    requests.len(),
                    MAX_BATCH_SIZE
                )),
            )];
        }

        let mut responses = Vec::new();
        for req_value in requests {
            // Convert Value directly to Request (avoids double-parsing)
            if let Some(resp) = self.handle_request_value(req_value).await {
                responses.push(resp);
            }
        }
        responses
    }

    /// Handle a pre-parsed JSON value (used by batch handler to avoid double-parsing)
    async fn handle_request_value(&mut self, value: serde_json::Value) -> Option<Response> {
        // Debug: log raw request
        if self.debug_options.log_requests {
            eprintln!("[DEBUG] Request: {}", value);
        }

        // Parse request from Value directly
        let request: Request = match serde_json::from_value(value) {
            Ok(req) => req,
            Err(e) => {
                return Some(Response::error(
                    RequestId::Number(0),
                    RpcError::parse_error(e.to_string()),
                ));
            },
        };

        self.handle_request(request).await
    }

    async fn handle_message(&mut self, line: &str) -> Option<Response> {
        // Debug: log raw request
        if self.debug_options.log_requests {
            eprintln!("[DEBUG] Request: {}", line);
        }

        // Parse request
        let request: Request = match serde_json::from_str(line) {
            Ok(req) => req,
            Err(e) => {
                return Some(Response::error(
                    RequestId::Number(0),
                    RpcError::parse_error(e.to_string()),
                ));
            },
        };

        self.handle_request(request).await
    }

    /// Core request handling logic (used by both handle_message and handle_request_value)
    async fn handle_request(&mut self, request: Request) -> Option<Response> {
        let Request { id, method, params, .. } = request;
        // Wrap id in Arc to reduce cloning cost (especially for String IDs)
        let id = Arc::new(id);
        let start_time = std::time::Instant::now();

        // Debug: log parsed request info
        if self.debug_options.log_requests {
            eprintln!("[DEBUG] → {} (id: {:?})", method, id);
        }

        // Check if hooks should be called (skip for internal methods)
        let call_hooks = !method.starts_with("$/") && method != methods::INITIALIZE && method != methods::SHUTDOWN;

        // Call pre-request hook
        if call_hooks {
            if let Some(ref hook) = self.pre_request_hook {
                let request_info = RequestInfo {
                    method: method.clone(),
                    id: (*id).clone(),
                    params: params.clone(),
                };
                if let PreRequestAction::Reject(error) = hook(&request_info) {
                    // Call post-request hook on rejection
                    if let Some(ref post_hook) = self.post_request_hook {
                        post_hook(&ResponseInfo {
                            method,
                            id: (*id).clone(),
                            success: false,
                            error_code: Some(error.code),
                            duration: start_time.elapsed(),
                        });
                    }
                    return Some(Response::error(Arc::unwrap_or_clone(id), error));
                }
            }
        }

        // Handle based on method
        let result = match method.as_str() {
            methods::INITIALIZE => self.handle_initialize(params),
            methods::SHUTDOWN => {
                // Call cleanup callback if set
                if let Some(callback) = self.shutdown_callback.take() {
                    callback();
                }
                // Signal graceful shutdown (run loop will exit after sending response)
                self.shutdown_requested = true;
                Ok(serde_json::json!(null))
            },
            methods::CANCEL => {
                self.handle_cancel(params).await;
                return None; // Cancel is a notification, no response
            },
            "$/ping" => {
                // Health check endpoint
                Ok(serde_json::json!({ "status": "ok" }))
            },
            _ => {
                if !self.initialized {
                    Err(RpcError::new(error_codes::INVALID_REQUEST, "Server not initialized"))
                } else if let Some(handler) = self.handlers.get(&method) {
                    // Create context with cancellation token and shared state
                    let ctx = match &self.state {
                        Some(state) => Context::with_state_dyn((*id).clone(), state.clone()),
                        None => Context::new((*id).clone()),
                    };
                    let cancel_handle = CancellationHandle::new(&ctx);

                    // Register active request
                    self.active_requests
                        .lock()
                        .await
                        .insert((*id).clone(), cancel_handle.clone());

                    // Determine effective timeout (handler-specific overrides default)
                    let effective_timeout = handler.timeout.or(self.default_timeout);

                    // Execute handler with optional timeout
                    let result = match effective_timeout {
                        Some(timeout_duration) => {
                            match tokio::time::timeout(timeout_duration, (handler.func)(ctx, params)).await {
                                Ok(result) => result,
                                Err(_elapsed) => {
                                    cancel_handle.cancel();
                                    Err(RpcError::new(
                                        error_codes::REQUEST_CANCELLED,
                                        format!("Request timed out after {:?}", timeout_duration),
                                    ))
                                },
                            }
                        },
                        None => (handler.func)(ctx, params).await,
                    };

                    // Unregister active request
                    self.active_requests.lock().await.remove(&*id);

                    result
                } else {
                    Err(RpcError::method_not_found(&method))
                }
            },
        };

        let duration = start_time.elapsed();

        // Debug: log profiling info
        if self.debug_options.log_profiling {
            let status = if result.is_ok() { "OK" } else { "ERR" };
            eprintln!("[PROFILE] {} {:?} - {} ({:?})", method, id, status, duration);
        }

        // Call post-request hook
        if call_hooks {
            if let Some(ref hook) = self.post_request_hook {
                hook(&ResponseInfo {
                    method,
                    id: (*id).clone(),
                    success: result.is_ok(),
                    error_code: result.as_ref().err().map(|e| e.code),
                    duration,
                });
            }
        }

        // Unwrap Arc for final response (no more clones needed)
        let id = Arc::unwrap_or_clone(id);
        let response = match result {
            Ok(value) => Response::success(id, value),
            Err(error) => Response::error(id, error),
        };

        // Debug: log response
        if self.debug_options.log_responses {
            if let Ok(json) = serde_json::to_string(&response) {
                eprintln!("[DEBUG] ← {}", json);
            }
        }

        Some(response)
    }

    async fn handle_cancel(&self, params: Option<serde_json::Value>) {
        let Some(params) = try_deserialize_params::<CancelParams>(params) else {
            return;
        };

        let active = self.active_requests.lock().await;
        if let Some(handle) = active.get(&params.id) {
            handle.cancel();
            log_debug(&format!("Request {:?} cancelled", params.id));
        }
    }

    fn handle_initialize(&mut self, params: Option<serde_json::Value>) -> Result<serde_json::Value, RpcError> {
        if self.initialized {
            return Err(RpcError::new(error_codes::INVALID_REQUEST, "Already initialized"));
        }

        let _params: InitializeParams = deserialize_params(params)?;

        self.initialized = true;

        // Convert local metadata to RPC metadata
        let metadata = if self.metadata.description.is_some()
            || self.metadata.author.is_some()
            || self.metadata.homepage.is_some()
            || self.metadata.license.is_some()
            || self.metadata.repository.is_some()
            || self.metadata.supported_targets.is_some()
            || self.metadata.min_hodu_version.is_some()
        {
            Some(PluginMetadataRpc {
                description: self.metadata.description.clone(),
                author: self.metadata.author.clone(),
                homepage: self.metadata.homepage.clone(),
                license: self.metadata.license.clone(),
                repository: self.metadata.repository.clone(),
                supported_targets: self.metadata.supported_targets.clone(),
                min_hodu_version: self.metadata.min_hodu_version.clone(),
            })
        } else {
            None
        };

        let result = InitializeResult {
            name: self.name.clone(),
            version: self.version.clone(),
            protocol_version: PROTOCOL_VERSION.to_string(),
            plugin_version: PLUGIN_VERSION.to_string(),
            capabilities: self.capabilities.clone(),
            model_extensions: self.model_extensions.clone(),
            tensor_extensions: self.tensor_extensions.clone(),
            devices: self.devices.clone(),
            metadata,
        };

        serde_json::to_value(result).map_err(|e| RpcError::internal_error(e.to_string()))
    }
}
