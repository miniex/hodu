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

---

## Newly Discovered Issues (3rd Analysis)

**Validation:** (🟡 Important)
- [x] Validate empty device string - `backend.rs:47` `device_type()` now returns `Option<&str>`, `None` for empty
- [x] Add parameter struct validation - `rpc.rs` added `validate()` methods to LoadModelParams, SaveModelParams, LoadTensorParams, SaveTensorParams, RunParams, BuildParams, TensorInput with `ValidationError` type
- [x] Limit collection sizes - `rpc.rs` added `MAX_*` constants and `validate_limits()` / `is_within_limits()` methods to InitializeResult

**API Consistency:** (🟢 Nice-to-have)
- [x] Add missing error factories - `rpc.rs:658-682` added `device_not_available()`, `model_error()`, `tensor_error()`, `plugin_error()`, `invalid_format()`
- [x] Validate LogParams level - `rpc.rs:382` added `is_valid_level()`, `normalized_level()`, and `VALID_LOG_LEVELS`
- [x] Validate TensorInput/Output names - `rpc.rs:314,348` added `is_valid_name()` methods

---

## Newly Discovered Issues (4th Analysis)

**Validation:** (🟢 Nice-to-have)
- [x] Strengthen BuildTarget triple validation - `backend.rs:113-124` now requires 3+ hyphen-separated parts (arch-vendor-os)
- [x] Strengthen method name validation - `rpc.rs:751-757` now requires at least one alphanumeric character

**Code Quality:** (🟢 Nice-to-have)
- [x] Refactor duplicate validation logic - `rpc.rs:300-382` `is_within_limits()` now calls `validate_limits().is_ok()`
- [x] Consider Result for `numel()` overflow - `tensor.rs:252` now returns `Option<usize>`, `None` on overflow; added `numel_unchecked()` for backwards compatibility
