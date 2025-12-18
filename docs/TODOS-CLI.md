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
- [ ] Add plugin signature/hash verification - `plugin/install.rs` no integrity check for downloaded plugins (requires infrastructure)
- [x] Prevent path traversal - `run.rs:352` `expand_path()` now returns Result and validates `..` sequences
- [x] Validate git URLs - `plugin/install.rs:175` added `validate_git_url()` function

**Input Validation:** (🔴 Critical)
- [x] Fix JSON parsing integer overflow - `loader.rs:131` now uses `i32::try_from()` with error handling
- [x] Validate u64→usize casting - `loader.rs:73` now uses `usize::try_from()` with error handling
- [x] Add file size limits - `loader.rs:8` added `MAX_TENSOR_FILE_SIZE` constant (100MB)

**Resource Management:** (🟡 Important)
- [ ] Use RAII pattern for temp directories - `plugin/install.rs:169` not cleaned up on panic
- [x] Add network timeout - `setup.rs:94` added 30-second timeout via ureq Agent
- [ ] Limit manifest.json size - `plugin/install.rs:414` large manifest can cause DoS

**Error Handling:** (🟡 Important)
- [x] Log setup failures - `main.rs:50` now logs warning on `mark_setup_shown()` failure
- [x] Include build stdout - `plugin/install.rs:258` now shows both stdout and stderr
- [x] Include plugin capabilities in error - `convert.rs:90` added `format_capabilities()` helper

**Validation:** (🟢 Nice-to-have)
- [ ] Validate input names - `run.rs:274` allows special characters/empty strings
- [ ] Strengthen device string validation - `run.rs:400` no warning for unknown devices
- [ ] Validate timeout range - `run.rs:62` no min/max bounds check

**Reliability:** (🟢 Nice-to-have)
- [ ] Add plugin update rollback - `plugin/update.rs` no restore to previous version on failure
- [ ] Limit process spawning - `plugins/process.rs:66` unlimited process creation possible
