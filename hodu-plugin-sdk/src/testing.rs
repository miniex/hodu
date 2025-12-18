//! Testing utilities for plugin development
//!
//! This module provides tools for testing plugins without running a full server.
//!
//! # Example
//!
//! ```ignore
//! use hodu_plugin_sdk::testing::{MockClient, TestHarness};
//! use hodu_plugin_sdk::rpc::{RunParams, RunResult, RpcError};
//!
//! #[tokio::test]
//! async fn test_my_handler() {
//!     let harness = TestHarness::new();
//!
//!     let params = RunParams { /* ... */ };
//!     let result: RunResult = harness.call("backend.run", params).await.unwrap();
//!
//!     assert!(!result.outputs.is_empty());
//! }
//! ```

use crate::context::Context;
use crate::rpc::{Request, RequestId, RpcError};
use serde::{de::DeserializeOwned, Serialize};
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

// ============================================================================
// Mock Client
// ============================================================================

/// A mock client for simulating CLI requests to plugin handlers
///
/// Useful for unit testing individual handlers without the full server infrastructure.
///
/// # Example
///
/// ```ignore
/// use hodu_plugin_sdk::testing::MockClient;
///
/// async fn my_handler(ctx: Context, params: MyParams) -> Result<MyResult, RpcError> {
///     // handler implementation
/// }
///
/// #[tokio::test]
/// async fn test_handler() {
///     let client = MockClient::new();
///     let result = client.call_handler(my_handler, MyParams { /* ... */ }).await;
///     assert!(result.is_ok());
/// }
/// ```
pub struct MockClient {
    request_counter: std::sync::atomic::AtomicI64,
}

impl MockClient {
    /// Create a new mock client
    pub fn new() -> Self {
        Self {
            request_counter: std::sync::atomic::AtomicI64::new(1),
        }
    }

    /// Get the next request ID
    fn next_id(&self) -> RequestId {
        RequestId::Number(self.request_counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst))
    }

    /// Call a handler function directly with typed params
    ///
    /// This bypasses the JSON-RPC serialization and calls the handler directly.
    pub async fn call_handler<F, Fut, P, R>(&self, handler: F, params: P) -> Result<R, RpcError>
    where
        F: FnOnce(Context, P) -> Fut,
        Fut: Future<Output = Result<R, RpcError>>,
    {
        let ctx = Context::new(self.next_id());
        handler(ctx, params).await
    }

    /// Call a handler with JSON params and get JSON result
    ///
    /// Simulates the full JSON-RPC request/response cycle.
    pub async fn call_json<F, Fut, P, R>(
        &self,
        handler: F,
        params: serde_json::Value,
    ) -> Result<serde_json::Value, RpcError>
    where
        F: FnOnce(Context, P) -> Fut,
        Fut: Future<Output = Result<R, RpcError>>,
        P: DeserializeOwned,
        R: Serialize,
    {
        let typed_params: P = serde_json::from_value(params).map_err(|e| RpcError::invalid_params(e.to_string()))?;

        let ctx = Context::new(self.next_id());
        let result = handler(ctx, typed_params).await?;

        serde_json::to_value(result).map_err(|e| RpcError::internal_error(e.to_string()))
    }

    /// Create a mock JSON-RPC request
    pub fn create_request<P: Serialize>(&self, method: &str, params: P) -> Result<Request, serde_json::Error> {
        let params_value = serde_json::to_value(params)?;
        Ok(Request::new(method, Some(params_value), self.next_id()))
    }
}

impl Default for MockClient {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Test Harness
// ============================================================================

/// Type-erased async handler for the test harness
type BoxedTestHandler = Box<
    dyn Fn(Context, serde_json::Value) -> Pin<Box<dyn Future<Output = Result<serde_json::Value, RpcError>> + Send>>
        + Send
        + Sync,
>;

/// Integration test harness for plugin testing
///
/// Allows registering handlers and calling them as if through JSON-RPC.
///
/// # Example
///
/// ```ignore
/// use hodu_plugin_sdk::testing::TestHarness;
///
/// #[tokio::test]
/// async fn test_plugin() {
///     let mut harness = TestHarness::new()
///         .handler("backend.run", handle_run)
///         .handler("backend.build", handle_build);
///
///     // Call handlers
///     let result: RunResult = harness.call("backend.run", params).await.unwrap();
/// }
/// ```
pub struct TestHarness {
    handlers: HashMap<String, BoxedTestHandler>,
    request_counter: std::sync::atomic::AtomicI64,
    /// Captured logs during test execution
    pub logs: Arc<std::sync::Mutex<Vec<LogEntry>>>,
}

/// A captured log entry
#[derive(Debug, Clone)]
pub struct LogEntry {
    pub level: String,
    pub message: String,
}

impl TestHarness {
    /// Create a new test harness
    pub fn new() -> Self {
        Self {
            handlers: HashMap::new(),
            request_counter: std::sync::atomic::AtomicI64::new(1),
            logs: Arc::new(std::sync::Mutex::new(Vec::new())),
        }
    }

    /// Register a handler
    pub fn handler<F, Fut, P, R>(mut self, method: &str, handler: F) -> Self
    where
        F: Fn(Context, P) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<R, RpcError>> + Send + 'static,
        P: DeserializeOwned + Send + 'static,
        R: Serialize + 'static,
    {
        let handler = Arc::new(handler);
        let boxed: BoxedTestHandler = Box::new(move |ctx, params| {
            let handler = handler.clone();
            Box::pin(async move {
                let typed_params: P =
                    serde_json::from_value(params).map_err(|e| RpcError::invalid_params(e.to_string()))?;
                let result = handler(ctx, typed_params).await?;
                serde_json::to_value(result).map_err(|e| RpcError::internal_error(e.to_string()))
            })
        });
        self.handlers.insert(method.to_string(), boxed);
        self
    }

    /// Get the next request ID
    fn next_id(&self) -> RequestId {
        RequestId::Number(self.request_counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst))
    }

    /// Call a registered handler with typed params and result
    pub async fn call<P, R>(&self, method: &str, params: P) -> Result<R, RpcError>
    where
        P: Serialize,
        R: DeserializeOwned,
    {
        let handler = self
            .handlers
            .get(method)
            .ok_or_else(|| RpcError::method_not_found(method))?;

        let params_value = serde_json::to_value(params).map_err(|e| RpcError::invalid_params(e.to_string()))?;

        let ctx = Context::new(self.next_id());
        let result = handler(ctx, params_value).await?;

        serde_json::from_value(result).map_err(|e| RpcError::internal_error(e.to_string()))
    }

    /// Call a registered handler with JSON params
    pub async fn call_json(&self, method: &str, params: serde_json::Value) -> Result<serde_json::Value, RpcError> {
        let handler = self
            .handlers
            .get(method)
            .ok_or_else(|| RpcError::method_not_found(method))?;

        let ctx = Context::new(self.next_id());
        handler(ctx, params).await
    }

    /// Get captured logs
    ///
    /// Returns logs even if the lock was poisoned (e.g., due to a panic in another test).
    pub fn get_logs(&self) -> Vec<LogEntry> {
        self.logs
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    /// Clear captured logs
    ///
    /// Clears logs even if the lock was poisoned.
    pub fn clear_logs(&self) {
        self.logs
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clear();
    }

    /// Assert that a specific log message was captured
    pub fn assert_logged(&self, level: &str, message_contains: &str) {
        let logs = self.logs.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let found = logs
            .iter()
            .any(|log| log.level == level && log.message.contains(message_contains));
        assert!(
            found,
            "Expected log with level '{}' containing '{}' not found. Logs: {:?}",
            level, message_contains, *logs
        );
    }
}

impl Default for TestHarness {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Assertion Helpers
// ============================================================================

/// Assert that an RpcError has a specific error code
pub fn assert_error_code(result: &Result<impl std::fmt::Debug, RpcError>, expected_code: i32) {
    match result {
        Ok(value) => panic!("Expected error with code {}, got Ok({:?})", expected_code, value),
        Err(e) => assert_eq!(
            e.code, expected_code,
            "Expected error code {}, got {} ({})",
            expected_code, e.code, e.message
        ),
    }
}

/// Assert that an RpcError message contains a substring
pub fn assert_error_contains(result: &Result<impl std::fmt::Debug, RpcError>, substring: &str) {
    match result {
        Ok(value) => panic!("Expected error containing '{}', got Ok({:?})", substring, value),
        Err(e) => assert!(
            e.message.contains(substring),
            "Expected error message to contain '{}', got '{}'",
            substring,
            e.message
        ),
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    async fn echo_handler(_ctx: Context, params: String) -> Result<String, RpcError> {
        Ok(format!("Echo: {}", params))
    }

    #[tokio::test]
    async fn test_mock_client() {
        let client = MockClient::new();
        let result = client.call_handler(echo_handler, "hello".to_string()).await;
        assert_eq!(result.unwrap(), "Echo: hello");
    }

    #[tokio::test]
    async fn test_harness() {
        let harness = TestHarness::new().handler("test.echo", echo_handler);

        let result: String = harness.call("test.echo", "world".to_string()).await.unwrap();
        assert_eq!(result, "Echo: world");
    }

    #[tokio::test]
    async fn test_harness_method_not_found() {
        let harness = TestHarness::new();
        let result: Result<String, _> = harness.call("unknown.method", "test".to_string()).await;
        assert_error_code(&result, crate::rpc::error_codes::METHOD_NOT_FOUND);
    }
}
