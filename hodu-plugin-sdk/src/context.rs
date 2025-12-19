//! Request context for plugin handlers
//!
//! Provides cancellation support, request metadata, and shared state access.

use crate::rpc::RequestId;
use std::any::Any;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

/// Context passed to async handlers
///
/// Contains cancellation token, request metadata, and optional shared state.
///
/// # Example
///
/// ```ignore
/// async fn handle_run(ctx: Context, params: RunParams) -> Result<RunResult, RpcError> {
///     // Access shared state
///     if let Some(config) = ctx.state::<MyConfig>() {
///         println!("Using config: {:?}", config);
///     }
///
///     for i in 0..100 {
///         // Check for cancellation periodically
///         if ctx.is_cancelled() {
///             return Err(RpcError::cancelled());
///         }
///
///         // Do work...
///         ctx.progress(Some(i as u8), &format!("Processing step {}", i));
///     }
///
///     Ok(result)
/// }
/// ```
#[derive(Clone)]
pub struct Context {
    request_id: RequestId,
    cancellation_token: CancellationToken,
    state: Option<Arc<dyn Any + Send + Sync>>,
}

impl Context {
    /// Create a new context
    pub(crate) fn new(request_id: RequestId) -> Self {
        Self {
            request_id,
            cancellation_token: CancellationToken::new(),
            state: None,
        }
    }

    /// Create a new context with a dynamic shared state
    pub(crate) fn with_state_dyn(request_id: RequestId, state: Arc<dyn Any + Send + Sync>) -> Self {
        Self {
            request_id,
            cancellation_token: CancellationToken::new(),
            state: Some(state),
        }
    }

    /// Get the shared state
    ///
    /// Returns `None` if no state was configured or if the type doesn't match.
    ///
    /// # Example
    ///
    /// ```ignore
    /// struct MyState {
    ///     counter: AtomicU64,
    /// }
    ///
    /// async fn handler(ctx: Context, params: Params) -> Result<Response, RpcError> {
    ///     if let Some(state) = ctx.state::<MyState>() {
    ///         state.counter.fetch_add(1, Ordering::SeqCst);
    ///     }
    ///     // ...
    /// }
    /// ```
    pub fn state<S: Send + Sync + 'static>(&self) -> Option<Arc<S>> {
        self.state.as_ref().and_then(|s| match s.clone().downcast::<S>() {
            Ok(state) => Some(state),
            Err(_) => {
                // Log in all builds - this indicates a programming error
                eprintln!(
                    "Warning: State downcast failed for type '{}' (request_id: {:?})",
                    std::any::type_name::<S>(),
                    self.request_id
                );
                None
            },
        })
    }

    /// Get the request ID
    pub fn request_id(&self) -> &RequestId {
        &self.request_id
    }

    /// Get the cancellation token (for advanced use)
    pub fn cancellation_token(&self) -> &CancellationToken {
        &self.cancellation_token
    }

    /// Check if the request has been cancelled
    ///
    /// Call this periodically in long-running handlers.
    pub fn is_cancelled(&self) -> bool {
        self.cancellation_token.is_cancelled()
    }

    /// Wait until cancelled
    ///
    /// Useful with `tokio::select!` to race against cancellation.
    ///
    /// # Example
    ///
    /// ```ignore
    /// tokio::select! {
    ///     result = do_work() => { /* work completed */ }
    ///     _ = ctx.cancelled() => { return Err(RpcError::cancelled()); }
    /// }
    /// ```
    pub async fn cancelled(&self) {
        self.cancellation_token.cancelled().await
    }

    /// Send a progress notification
    ///
    /// # Arguments
    /// * `percent` - Progress percentage (0-100), None for indeterminate. Values > 100 are clamped to 100.
    /// * `message` - Progress message
    pub fn progress(&self, percent: Option<u8>, message: &str) {
        crate::notify_progress(percent, message);
    }

    /// Send a progress notification with error handling
    ///
    /// Returns an error if the notification fails to send.
    pub fn try_progress(&self, percent: Option<u8>, message: &str) -> Result<(), std::io::Error> {
        crate::try_notify_progress(percent, message)
    }

    /// Send a log message
    ///
    /// # Arguments
    /// * `level` - Log level: "error", "warn", "info", "debug", "trace". Invalid levels default to "info".
    /// * `message` - Log message
    pub fn log(&self, level: &str, message: &str) {
        crate::notify_log(level, message);
    }

    /// Send a log message with error handling
    ///
    /// Returns an error if the notification fails to send.
    pub fn try_log(&self, level: &str, message: &str) -> Result<(), std::io::Error> {
        crate::try_notify_log(level, message)
    }

    /// Log an info message
    pub fn log_info(&self, message: &str) {
        self.log("info", message);
    }

    /// Log a warning message
    pub fn log_warn(&self, message: &str) {
        self.log("warn", message);
    }

    /// Log an error message
    pub fn log_error(&self, message: &str) {
        self.log("error", message);
    }

    /// Log a debug message
    pub fn log_debug(&self, message: &str) {
        self.log("debug", message);
    }
}

/// Handle for cancelling a request from outside
#[derive(Clone)]
pub(crate) struct CancellationHandle {
    token: CancellationToken,
}

impl CancellationHandle {
    pub fn new(ctx: &Context) -> Self {
        Self {
            token: ctx.cancellation_token.clone(),
        }
    }

    pub fn cancel(&self) {
        self.token.cancel();
    }
}
