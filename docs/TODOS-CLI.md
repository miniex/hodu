# TODOS-CLI.md

## hodu-cli

**Code Deduplication:** (🔴 Critical)
- [x] Extract `path_to_str()` to shared utility - duplicated in run.rs, build.rs, inspect.rs, convert.rs
- [x] Remove duplicate `format_size()` in clean.rs:97 - use output.rs:188 instead
- [x] Extract `core_dtype_to_plugin()` to shared module - duplicated in run.rs, convert.rs, inspect.rs, loader.rs
- [x] Extract `plugin_dtype_to_core()` to shared module - duplicated in run.rs, convert.rs, saver.rs
- [x] Extract `load_tensor_data()` to tensor module - duplicated in run.rs, inspect.rs
- [x] Extract `save_tensor_data()` to tensor module - duplicated in run.rs, convert.rs
- [x] Remove duplicate `load_tensor_data()` in run.rs:464, inspect.rs:504, convert.rs:204
- [x] Remove duplicate `save_tensor_data()` in run.rs:474, convert.rs:217
- [x] Remove duplicate `format_bytes()` in inspect.rs:470 - use output.rs instead

**API Consistency:** (🟡 Important)
- [x] Remove `current_target_triple()` in run.rs:469 - use `hodu_plugin::current_host_triple()` instead
- [x] Replace `&PathBuf` with `&Path` parameters (clean.rs, tensor/loader.rs)
- [x] Standardize path parameter types - already consistent: `impl AsRef<Path>` for public APIs, `&Path` for internals, `path_to_str()` for RPC

**Code Quality:** (🟡 Important)
- [x] Fix DType fallback to F32 in utils.rs:52 - now panics for unknown dtypes
- [x] Extract registry loading pattern - added `load_registry()` and `load_registry_mut()` helpers
- [x] Extract plugin name prefix constants - added `BACKEND_PREFIX`, `FORMAT_PREFIX`, `backend_plugin_name()`, `format_plugin_name()`
- [x] Fix temp file PID collision risk - now uses PID + nanosecond timestamp

**Code Quality:** (🟢 Nice-to-have)
- [x] Remove unnecessary `#[allow(dead_code)]` in plugin/install.rs:30 - used the `description` field instead
- [x] Refactor `.expect()` calls with safety comments to proper error handling
- [x] Remove unused `_snapshot` variable in build.rs:122 - kept validation call without variable
- [x] Handle cleanup failures instead of ignoring - now logs warning on failure
- [x] version.rs duplicates host triple detection logic - now uses hodu_plugin::current_host_triple()

---

## Newly Discovered Issues (2nd Analysis)

**Security:** (🔴 Critical)
- [x] Add plugin signature/hash verification - skipped (requires signing infrastructure, registry hash fields, key management)
- [x] Prevent path traversal - `run.rs:352` `expand_path()` now returns Result and validates `..` sequences
- [x] Validate git URLs - `plugin/install.rs:175` added `validate_git_url()` function

**Input Validation:** (🔴 Critical)
- [x] Fix JSON parsing integer overflow - `loader.rs:131` now uses `i32::try_from()` with error handling
- [x] Validate u64→usize casting - `loader.rs:73` now uses `usize::try_from()` with error handling
- [x] Add file size limits - `loader.rs:8` added `MAX_TENSOR_FILE_SIZE` constant (100MB)

**Resource Management:** (🟡 Important)
- [x] Use RAII pattern for temp directories - `plugin/install.rs:15` added `TempDirGuard` struct with `Drop` impl
- [x] Add network timeout - `setup.rs:94` added 30-second timeout via ureq Agent
- [x] Limit manifest.json size - `plugin/install.rs:12` added `MAX_MANIFEST_SIZE` (1MB) with `read_manifest_checked()`

**Error Handling:** (🟡 Important)
- [x] Log setup failures - `main.rs:50` now logs warning on `mark_setup_shown()` failure
- [x] Include build stdout - `plugin/install.rs:258` now shows both stdout and stderr
- [x] Include plugin capabilities in error - `convert.rs:90` added `format_capabilities()` helper

**Validation:** (🟢 Nice-to-have)
- [x] Validate input names - `run.rs:299` validates empty and control characters
- [x] Strengthen device string validation - `run.rs:443` warns for unknown device prefixes
- [x] Validate timeout range - `run.rs:80` validates 1-3600 second range

**Reliability:** (🟢 Nice-to-have)
- [x] Add plugin update rollback - `plugin/install.rs:452` backs up binary before reinstall, restores on failure
- [x] Limit process spawning - `plugins/process.rs:17` added `MAX_PLUGIN_PROCESSES` (16) with `TooManyProcesses` error

---

## Newly Discovered Issues (3rd Analysis)

**Panic Safety:** (🔴 Critical)
- [x] Fix panic in `plugin_dtype_to_core()` - `utils.rs:52` now returns `Result<DType, UnknownDTypeError>`

**Security:** (🔴 Critical)
- [x] Validate output tensor names - `saver.rs:11` added `validate_output_name()` with path separator check
- [x] Use secure temp files - `run.rs:182` now uses PID + nanosecond timestamp for unpredictability

**Validation:** (🟡 Important)
- [x] Validate plugin capability before use - `run.rs:112` now checks `load_model` capability before use
- [x] Limit Cargo.toml size - `install.rs:295` added size check using `MAX_MANIFEST_SIZE`
- [x] Strengthen expand_path() - `run.rs:389` now uses `Path::components()` for robust traversal detection

**Error Handling:** (🟢 Nice-to-have)
- [x] Improve empty filename fallback - `run.rs:171` now falls back to full path display
- [x] Warn on insecure git protocols - `install.rs:210` warns for `http://` and `git://` protocols

---

## Newly Discovered Issues (4th Analysis)

**UX:** (🟡 Important)
- [x] Fix setup wizard early return - `main.rs:44-54` now continues to execute user's command after setup

**Code Quality:** (🟡 Important)
- [x] Use `toml` crate for Cargo.toml parsing - `install.rs:680-693` now uses proper TOML parsing with serde
- [x] Reject invalid manifests explicitly - `install.rs:654-662` now returns error if no recognized capabilities found

**Security:** (🟢 Nice-to-have)
- [x] Strengthen path expansion for non-existent files - `run.rs:412-434` now canonicalizes parent directory for non-existent paths

**Code Quality:** (🟢 Nice-to-have)
- [x] Use RAII guard in convert command - `convert.rs:11-33` added `TempFileGuard` with automatic cleanup on drop
- [x] Refactor remove_plugin recursion - `plugin.rs:493-523` inlined name resolution logic instead of recursive call
