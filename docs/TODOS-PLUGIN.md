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

---

## Newly Discovered Issues (5th Analysis)

**Validation:** (🟡 Important)
- [x] Add validate() to InitializeParams - `rpc.rs:243-248` added validation for plugin_version and protocol_version
- [x] Add validate() to ProgressParams - `rpc.rs:616-628` added validation for percent (0-100) and message
- [x] Add validate() to LogParams - `rpc.rs:654-667` added validation for level and message
- [x] Add validate() to TensorOutput - `rpc.rs:558-567` added validation matching TensorInput pattern

---

## Newly Discovered Issues (6th Analysis)

**Bug Fix:** (🔴 Critical)
- [x] Fix with_hint() non-array handling - `rpc.rs:1081-1088` now converts existing non-array hints value to array when adding new hint
- [x] Fix with_cause() non-string cause handling - `rpc.rs:1030-1035` now properly extracts string from existing cause value before chaining

**Validation:** (🟡 Important)
- [x] Strengthen Request.is_valid() - `rpc.rs:812-826` now validates method name contains only valid chars (alphanumeric, '.', '_', '/', '$') and at least one alphanumeric
- [x] Add whitespace-only check to validate_path - `rpc.rs:57-62` now rejects paths containing only whitespace

**Code Quality:** (🟢 Nice-to-have)
- [x] Remove unnecessary unwrap in device_type - `backend.rs:51-52` now returns `split().next()` directly without wrapping in Some+unwrap

---

## Newly Discovered Issues (7th Analysis)

**Validation:** (🟡 Important)
- [x] Add size limit to hints array - `rpc.rs:33,1124-1127` added MAX_HINTS (20) and enforces limit in with_hint()
- [x] Add path traversal check to validate_path - `rpc.rs:75-81` now checks for ".." traversal sequences
- [x] Validate field key names in with_field - `rpc.rs:35-36,1173-1180` warns in debug builds if using reserved field names

**Code Quality:** (🟢 Nice-to-have)
- [x] Extract shared is_valid_name() helper - `rpc.rs:85-91` added `is_valid_tensor_name()` helper, used by both TensorInput and TensorOutput
- [x] Align Notification::progress() with ProgressParams validation - `rpc.rs:922-925` documented the intentional difference (convenience clamping vs strict validation)
- [x] Document numel_unchecked() overflow behavior - `tensor.rs:260-272` added Warning section and example in docstring

---

## Newly Discovered Issues (8th Analysis)

**Validation:** (🟡 Important)
- [x] with_hint() silently drops hints beyond MAX_HINTS - `rpc.rs:1158-1168` now logs debug warning when dropping hints
- [x] Inconsistent MAX_HINTS constant usage - all code paths now use MAX_HINTS consistently
- [x] Add tensor name length limit - `rpc.rs:35-36` added MAX_TENSOR_NAME_LEN (255 bytes), used in `is_valid_tensor_name()`

**Path Validation:** (🟡 Important)
- [x] Strengthen path traversal detection - `rpc.rs:95-107` now checks for "~" home expansion and control characters
- [x] Validate absolute vs relative paths - `rpc.rs:61-67` documented that validation accepts both, caller determines expectation

**API Consistency:** (🟢 Nice-to-have)
- [x] Document percent clamping vs validation strategy - `rpc.rs:956-970` added table documenting clamping vs validation strategy
- [x] Add length limits to error message strings - `rpc.rs:38-39` added MAX_ERROR_STRING_LEN (64KB), used in with_cause() and with_hint()

---

## Newly Discovered Issues (9th Analysis)

**Safety:** (🔴 Critical)
- [x] Fix UTF-8 string slicing panic - `rpc.rs:48-60` added `truncate_utf8_owned()` helper, used in `with_cause()` and `with_hint()`

**Validation:** (🟡 Important)
- [x] Add metadata string length limits - `rpc.rs:44-60` added MAX_METADATA_* constants, `PluginMetadataRpc::validate()` and `sanitize()` methods
- [x] Strengthen path traversal detection - `rpc.rs:129-158` now checks URL-encoded forms (%2e, %25, %5c, %2f)

**Testing:** (🟡 Important)
- [x] Add validation method tests - `rpc.rs:1738-1986` added 11 tests for all param validation methods

**Documentation:** (🟢 Nice-to-have)
- [x] Document InitializeParams version format - `rpc.rs:411-462` added semver format, compatibility rules, example
- [x] Make hint truncation warnings consistent - `rpc.rs:1502-1509` with_field() now logs in all builds (removed #[cfg(debug_assertions)])
