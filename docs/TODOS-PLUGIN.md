# TODOS-PLUGIN.md

## hodu_plugin

**Error Handling:** (🔴 Critical)
- [x] Replace `.unwrap()` with proper error handling in rpc.rs:343,351 (Notification::progress, Notification::log)

**API Consistency:** (🟡 Important)
- [x] Add `BACKEND_SUPPORTED_DEVICES` to capabilities module in rpc.rs (exists in methods but missing in capabilities)
- [x] Standardize RpcError factory methods - all now use `impl Into<String>`
- [x] Make PluginDType FromStr error implement std::error::Error - added ParseDTypeError type

**Validation:** (🟡 Important)
- [x] Add `TensorData::new_checked()` constructor with validation in tensor.rs
- [x] Consider Response struct validation (must have either result OR error per JSON-RPC spec) - added `is_valid()`, `is_success()`, `is_error()`

**Error Chain:** (🟡 Important)
- [x] Preserve error chain in `From<io::Error>` for PluginError - now uses Arc to preserve source

**Dead Code:** (🟡 Important)
- [x] Remove or use `capabilities` module - removed unused module

**Testing:** (🟢 Nice-to-have)
- [x] Add unit tests for PluginDType FromStr parsing
- [x] Add unit tests for TensorData validation (is_valid, is_scalar)
- [x] Add unit tests for JSON-RPC serialization/deserialization
- [x] Add unit tests for device parsing logic - tests in backend.rs

**Documentation:** (🟢 Nice-to-have)
- [x] Add rustdoc for public struct fields in rpc.rs - all params/result structs documented
- [x] Add rustdoc for public functions in rpc.rs - Request::new, Response::*, Notification::*, RpcError::* documented

---

## Newly Discovered Issues (2nd Analysis)

**Arithmetic Safety:** (🔴 Critical)
- [x] Prevent shape product overflow - `tensor.rs:211,224` uses checked_mul via `checked_numel()`

**Validation:** (🔴 Critical)
- [x] Remove expect() calls - `rpc.rs:483,500` now uses `serde_json::json!` macro which is infallible
- [x] Validate percent range - `rpc.rs:475` clamps percent to 0-100
- [x] Validate BuildTarget - `backend.rs:92` added `new_checked()` with validation
- [x] Validate zero dimension shapes - `tensor.rs:224` rejects shapes with zero dimensions

**API Design:** (🟡 Important)
- [x] Strengthen Response state validation - `rpc.rs:527` added `validate()` method returning Result
- [x] Validate Request method - `rpc.rs:451` added `new_checked()` with validation

**Parsing:** (🟢 Nice-to-have)
- [x] Strengthen device ID parsing - `backend.rs:25` now rejects malformed input like `cuda::0::extra`
- [x] Remove unnecessary unwrap_or - `backend.rs:48` changed to `unwrap()` with comment

**Documentation:** (🟢 Nice-to-have)
- [x] Document parse_device_id - `backend.rs:14` added doc comment with examples
- [x] Document device_type - `backend.rs:35` added doc comment with examples
- [x] Document BuildTarget fields - `backend.rs:76-80` fields now have doc comments
