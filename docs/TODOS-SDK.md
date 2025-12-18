# TODOS-SDK.md

## hodu-plugin-sdk

**Core Features:** (🔴 Critical)
- [x] Implement cancellation handling - process `$/cancel` requests and allow handlers to check cancellation status
- [x] Implement async handler support - `async fn` handlers with tokio runtime
- [x] Implement graceful shutdown - cleanup callback before exit (`on_shutdown`)

**Validation:** (🔴 Critical)
- [x] Add percent validation (0-100) in notify_progress - values > 100 are now clamped
- [x] Add log level validation in notify_log - invalid levels default to "info"
- [x] Handle notification failures instead of silent discard - now logs to stderr on failure

**Server Improvements:** (🟡 Important)
- [x] Implement context/state passing - shared state across handlers (e.g., config, connections)
- [x] Implement JSON-RPC batch requests - handle array of requests per spec
- [x] Implement per-handler timeout - configurable timeout with auto-cancellation
- [x] Implement middleware/hooks - pre/post request processing (logging, auth, etc.)

**Code Deduplication:** (🟡 Important)
- [x] Extract param deserialization helper - added `deserialize_params()` and `try_deserialize_params()` helpers
- [x] Extract handler registration logic - added `register_handler()` helper method
- [x] Reduce excessive cloning in request processing - destructure request, use `call_hooks` flag, simplify control flow

**Plugin Metadata:** (🟡 Important)
- [x] Add plugin metadata fields - description, author, homepage, license, repository
- [x] Add supported OS/arch declaration - target triple patterns
- [x] Add minimum hodu version requirement - semver compatibility

**Code Quality:** (🟡 Important)
- [x] Fix DType fallback masking in tensor.rs:52 - now panics with descriptive message for unknown dtypes
- [x] Remove dead code `with_state()` in context.rs:54 - removed unused method

**Developer Experience:** (🟢 Nice-to-have)
- [x] Implement health check endpoint - `$/ping` method for liveness probe
- [x] Implement automatic profiling - `enable_profiling()` logs execution times to stderr
- [x] Implement streaming response - `StreamWriter` for chunked data via notifications
- [x] ~~Implement hot reload support~~ - skipped (not useful for compiled plugins)
- [x] Add `#[derive(PluginMethod)]` macro - `PluginMethod` derive, `plugin_handler` attr, `define_params/result` macros

**Error Handling:** (🟢 Nice-to-have)
- [x] Implement error chain/cause - `with_cause()`, `with_error_cause()` methods
- [x] Implement error recovery hints - `with_hint()`, `with_hints()` methods
- [x] Add structured error data - `with_field()`, `with_details()`, `with_context()` methods

**Testing:** (🟢 Nice-to-have)
- [x] Add mock server for testing - `MockClient` for unit testing handlers
- [x] Add integration test harness - `TestHarness` for E2E plugin testing
- [x] Add request/response logging mode - `DebugOptions` and `.debug()` method

---

## Newly Discovered Issues (2nd Analysis)

**Error Handling:** (🟡 Important)
- [x] Handle Mutex poisoning - `testing.rs:250,255,260` now uses `unwrap_or_else(|poisoned| poisoned.into_inner())`
- [x] Handle notification flush failures - `server.rs:192,216` now logs warning to stderr on failure
- [x] Preserve StreamWriter error context - `server.rs:302,322,342` now includes chunk index in error messages

**Code Quality:** (🟢 Nice-to-have)
- [x] Use Clippy div_ceil - `server.rs:360` `(data.len() + 2) / 3 * 4` → `.div_ceil(3) * 4`
- [ ] Enable doc tests - multiple files use `/// ```ignore`, need actual tests

**Validation:** (🟢 Nice-to-have)
- [ ] Validate handler names - `server.rs:815` allows special characters/reserved names
- [ ] Improve empty batch handling - `server.rs:878` JSON-RPC 2.0 spec unclear

**Performance:** (🟢 Nice-to-have)
- [ ] Use RwLock for logs - `testing.rs:179` use RwLock instead of Mutex for concurrent reads

**Missing Features:** (🟢 Nice-to-have)
- [ ] Add request size limit - `server.rs` large JSON payloads can exhaust memory
- [ ] Add default request timeout - `server.rs` requests can wait indefinitely
