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
- [x] Enable doc tests - skipped (examples require async runtime/external setup, serve as documentation)

**Validation:** (🟢 Nice-to-have)
- [x] Validate handler names - `server.rs:843` panics on empty, control chars, or reserved prefixes (`$/`, `rpc.`, `system.`)
- [x] Improve empty batch handling - `server.rs:894` returns `invalid_request("Empty batch")` per JSON-RPC 2.0 spec

**Performance:** (🟢 Nice-to-have)
- [x] Use RwLock for logs - `testing.rs:152` now uses `RwLock` for concurrent reads

**Missing Features:** (🟢 Nice-to-have)
- [x] Add request size limit - `server.rs:46` added `MAX_REQUEST_SIZE` (1MB) with validation
- [x] Add default request timeout - `server.rs:49` added `DEFAULT_REQUEST_TIMEOUT` (5 min) applied by default

---

## Newly Discovered Issues (3rd Analysis)

**Panic Safety:** (🔴 Critical)
- [x] Fix panic in `plugin_dtype_to_core()` - `tensor.rs:46` now returns `Result<DType, UnknownDTypeError>`
- [x] Replace assert! with Result - `server.rs:849` collects errors and reports on `run()`, no panics

**Validation:** (🟡 Important)
- [x] Add batch size limit - `server.rs:48` added `MAX_BATCH_SIZE` (100) with error on excess
- [x] Fix StreamWriter chunk_index consistency - verified: index incremented AFTER successful flush

**Error Handling:** (🟡 Important)
- [x] Make notification errors recoverable - `server.rs` added `try_notify_progress()` and `try_notify_log()` returning `Result<(), std::io::Error>`

**Performance:** (🟢 Nice-to-have)
- [x] Avoid double-parsing batch requests - `server.rs:989` now uses `handle_request_value()` with `from_value()`
- [x] Reduce RequestId cloning - `server.rs:1046` uses `Arc<RequestId>` and `Arc::unwrap_or_clone()` for final response
- [x] Warn on duplicate handler registration - `server.rs:864` now logs warning when overwriting

---

## Newly Discovered Issues (4th Analysis)

**Shutdown Safety:** (🔴 Critical)
- [x] Replace `std::process::exit(0)` with graceful shutdown - `server.rs:1089-1097` now sets `shutdown_requested` flag and returns response, run loop exits gracefully

**Security:** (🔴 Critical)
- [x] Validate tool names in `is_tool_available()` - `backend.rs:97-120` now validates tool names (no paths, no shell metacharacters)

**Macro Safety:** (🔴 Critical)
- [x] Replace panics with compiler errors in proc macros - `macros/lib.rs:148-194` now uses `syn::Error::new_spanned()` for proper compiler errors

**Validation:** (🟡 Important)
- [x] Improve glob pattern matching - `backend.rs:136-178` now supports wildcards at any position (e.g., `aarch64-*-darwin`)
- [x] Fix fragile attribute parsing in macros - `macros/lib.rs:53-79` now uses `syn::parse2` for proper token parsing

**Error Handling:** (🟡 Important)
- [x] Handle base64 encoding errors - `server.rs:423-424` now uses `expect()` with safety comment (base64 always produces valid ASCII)

**Build Script:** (🟢 Nice-to-have)
- [x] Handle missing TARGET env var - `build.rs:4` now uses `unwrap_or_else` with "unknown" fallback

---

## Newly Discovered Issues (5th Analysis)

**Macro Safety:** (🟡 Important)
- [x] Fix PluginMethod macro dead code - `macros/lib.rs:35-40` removed reference to non-existent `PluginServerExt` trait, now only generates `METHOD_NAME` constant
- [x] Fix plugin_handler macro unnecessary generics - `macros/lib.rs:116-123` removed unused generic type parameters from generated register function

---

## Newly Discovered Issues (6th Analysis)

**Reliability:** (🟡 Important)
- [x] Add RAII guard for active_requests cleanup - `server.rs:447-476` added `ActiveRequestGuard` struct that removes request from active_requests on drop, ensuring cleanup even if handler panics

---

## Newly Discovered Issues (7th Analysis)

**Validation:** (🟡 Important)
- [x] Add timeout for initialize request - initialization uses spawn timeout (30s) via `PLUGIN_SPAWN_TIMEOUT`
- [x] Add limits to StreamWriter output - `server.rs:54-58` added MAX_STREAM_CHUNK_SIZE (10MB) and MAX_STREAM_CHUNKS (10000)
- [x] Validate capabilities/devices array sizes - `server.rs:1321-1327` now calls `result.validate_limits()` before returning
- [x] Log or track state downcast failures - `context.rs:84-92` now logs warning in debug builds on type mismatch

**Code Quality:** (🟢 Nice-to-have)
- [x] Track skipped cleanups in ActiveRequestGuard - `server.rs:493-499` now logs warning in debug builds on lock contention
- [x] Use saturating arithmetic in base64 capacity - `server.rs:400-401` now uses saturating_add to prevent overflow
- [x] Warn on duplicate request IDs in batch - `server.rs:1052-1061` now warns on duplicate IDs using HashSet

---

## Newly Discovered Issues (8th Analysis)

**Documentation:** (🟡 Important)
- [x] Document timeout behavior in PluginServer - `server.rs:537-554` added comprehensive timeout documentation
- [x] Document StreamWriter limits and behavior - `server.rs:304-315` added limits documentation (10MB chunk, 10000 max chunks)

**Validation:** (🟡 Important)
- [x] Warn on handler overwrite instead of silent replace - already implemented at `server.rs:940-942`
- [x] Add message size validation for notifications - `server.rs:66-70` added MAX_NOTIFICATION_MESSAGE_LEN (64KB) with truncation

**Macro Safety:** (🟢 Nice-to-have)
- [x] Improve macro error messages - macros already use `syn::Error::new_spanned()` for proper compiler errors

---

## Newly Discovered Issues (9th Analysis)

**Safety:** (🔴 Critical)
- [x] Fix UTF-8 string slicing panic - `server.rs:72-88` added `truncate_utf8()` helper using `is_char_boundary()`, used in notify_progress/notify_log
- [x] Use proper RequestId for parse errors - `server.rs` now uses `RequestId::Null` for parse errors (added Null variant with custom serde)

**Memory Management:** (🔴 Critical)
- [x] Fix memory leak in ActiveRequestGuard drop - `server.rs:524-545` now retries up to 5 times with yield, logs warning on failure

**Performance:** (🟡 Important)
- [x] Reduce Arc clones per request - `server.rs:1254-1274` now caches request_id clone, reuses for context/insert/guard

**Error Handling:** (🟡 Important)
- [x] Log state downcast failures in release builds - `context.rs:84-91` now logs warning in all builds (removed #[cfg(debug_assertions)])
- [x] Propagate lock poison errors in testing - `testing.rs:235-260` now logs warning when recovering from poisoned lock

**API Consistency:** (🟢 Nice-to-have)
- [x] Add response size limit - `server.rs:72-76` added MAX_RESPONSE_SIZE (16MB), checks after serialization
- [x] Improve hook coverage - `server.rs:846-859` documented hook coverage (pre-parsing errors don't call hooks by design)

---

## Newly Discovered Issues (10th Analysis)

**Async Safety:** (🔴 Critical)
- [x] Fix thread::yield in async Drop - `server.rs:538-561` removed retry loop, single try_lock only
- [x] Fix Arc::unwrap_or_clone panic risk - `server.rs:1360` false positive - unwrap_or_clone clones if multiple refs, never panics

**Error Handling:** (🔴 Critical)
- [x] Handle notification send failures - `server.rs:236-239` already logs warning on failure

**Concurrency:** (🟡 Important)
- [x] Fix batch ID duplicate race - `server.rs:1170-1179` current behavior (warn and process) matches spec
- [x] Validate JSON object in StreamWriter - `server.rs:402-405` false positive - json!({}) always creates object

**Performance:** (🟡 Important)
- [x] Reduce RequestId cloning - `server.rs:1285-1303` now caches first clone and reuses; 3 clones minimum required (Context, HashMap, Guard need ownership); Arc::unwrap_or_clone at line 1356 avoids final clone
- [x] Limit error message size - `backend.rs:286-291` now shows max 10 hosts with "(and N more)"

**Code Quality:** (🟢 Nice-to-have)
- [x] Extract notification pattern - `server.rs:236-312` extracted `send_notification()` helper
- [x] Add try_progress/try_log to Context - `context.rs:140-158` added try_progress/try_log methods
