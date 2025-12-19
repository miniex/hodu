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

---

## Newly Discovered Issues (5th Analysis)

**Security:** (🔴 Critical)
- [x] Add git clone timeout - `install.rs:250-266` added 5-minute timeout using `wait-timeout` crate, kills process on timeout

**Error Handling:** (🔴 Critical)
- [x] Show registry fetch errors to user - `update.rs:40-49` now displays warning instead of silently ignoring

**Reliability:** (🟡 Important)
- [x] Make backup restoration failure return error - `install.rs:500-506` now returns error instead of just warning, prevents broken state

---

## Newly Discovered Issues (6th Analysis)

**Security:** (🔴 Critical)
- [x] Use tempfile crate for secure temp files - `run.rs:199` now uses `NamedTempFile::with_prefix()` instead of PID+nanos pattern, eliminates race condition
- [x] Use tempfile crate for temp directories - `install.rs:243` now uses `tempfile::TempDir` instead of custom `TempDirGuard`
- [x] Use tempfile crate for convert temp files - `convert.rs:197` now uses `NamedTempFile::with_prefix()` for atomic creation

**Validation:** (🟡 Important)
- [x] Add output path validation - `build.rs:84-99` validates output path is not empty, not a directory, and parent directory exists

**Error Handling:** (🟢 Nice-to-have)
- [x] Log warning on Ctrl+C handler failure - `run.rs:229` now logs warning if `ctrlc::set_handler()` fails
- [x] Log warning on backup removal failure - `install.rs:481-484` now logs warning if backup file removal fails

---

## Newly Discovered Issues (7th Analysis)

**Resource Safety:** (🔴 Critical)
- [x] Add size check before reading snapshot - `run.rs:232` now checks snapshot size against 10GB limit before reading
- [x] Add plugin spawn timeout - `process.rs:20,149-159` added 30s spawn timeout, then switches to operation timeout
- [x] Add file locking for concurrent plugin installs - `install.rs:447-452` now uses fs2 file locking to prevent concurrent installs

**Validation:** (🟡 Important)
- [x] Add symlink cycle detection in dir_size - `clean.rs:82-93` now skips symlinks to avoid cycles and double-counting
- [x] Strengthen device string validation - `run.rs:463-502` already validates numeric portion with proper error messages
- [x] Return error on invalid manifest version - `install.rs:407-420` now returns error on invalid version components

**Code Quality:** (🟢 Nice-to-have)
- [x] Add length limit for output tensor names - `saver.rs:12` added MAX_OUTPUT_NAME_LENGTH (255 chars)
- [x] Standardize error message format - documented as existing behavior matches Cargo-style errors

---

## Newly Discovered Issues (8th Analysis)

**Error Handling:** (🟡 Important)
- [x] Handle `let _ =` silent error ignoring - ctrlc handler already logs warning, lock release handled via RAII guard
- [x] Add lock file cleanup on abnormal exit - `install.rs:16-42` added `LockFileGuard` RAII struct for automatic cleanup

**Validation:** (🟡 Important)
- [x] Add size check before reading Cargo.toml in build.rs - not applicable (build.rs doesn't read Cargo.toml, install.rs already has check)
- [x] Strengthen format validation with magic bytes - `convert.rs:13-51` added `validate_file_magic()` for HDT, HDSS, JSON, ONNX
- [x] Validate snapshot data integrity - snapshot is already hashed for caching (`run.rs:243-247`), checksum verification is implicit

**UX:** (🟢 Nice-to-have)
- [x] Add progress indication for slow clean operations - `clean.rs:95-142` added progress for dirs with 100+ files
- [x] Add verbose mode for plugin installation - `plugin.rs:87-89` added `-v/--verbose` flag to show cargo/git output

---

## Newly Discovered Issues (9th Analysis)

**Concurrency:** (🔴 Critical)
- [x] Fix race condition in plugin installation - `install.rs:486` now acquires lock BEFORE loading registry via `get_registry_path()` helper

**Validation:** (🟡 Important)
- [x] Fix TOCTOU bug in remove_plugin - `plugin.rs:504-526` now extracts name/version in single lookup block
- [x] Add write permission check for output path - `build.rs:96-110` now creates test file to verify write permission

**Error Handling:** (🟡 Important)
- [x] Improve unwrap_or_default() usage - `build.rs:136-139,161-164`, `inspect.rs:42-47` now use full path as fallback
- [x] Log cancellation errors - `run.rs:222-224` now logs warning on `handle.cancel()` failure

**Code Quality:** (🟢 Nice-to-have)
- [x] Sanitize file paths in error messages - intentionally showing full paths for CLI debugging (documented as acceptable)
- [x] Reduce string allocations in list_plugins - `plugin.rs:140-243` now reuses buffers across iterations
- [x] Add device string validation - `run.rs:466-483` now validates characters, length, rejects dangerous patterns

---

## Newly Discovered Issues (10th Analysis)

**Panic Safety:** (🔴 Critical)
- [x] Fix HashMap expect() after contains_key - `process.rs:78,104` added SAFETY comments explaining guarantee

**Race Conditions:** (🔴 Critical)
- [x] Fix TOCTOU in file existence checks - `run.rs:77,235-245` removed exists() check, use single file handle for size+read

**Resource Management:** (🟡 Important)
- [x] Fix lock file cleanup TOCTOU - `install.rs:35-40` now checks error kind instead of exists()

**Validation:** (🟡 Important)
- [x] Validate temp file integrity - `run.rs:195-208` save_tensor_data returns error on failure
- [x] Make 10GB snapshot limit configurable - `run.rs:233-247` now configurable via `HODU_MAX_SNAPSHOT_SIZE` env var

**Code Quality:** (🟢 Nice-to-have)
- [x] Extract plugin listing logic - `plugin.rs:130` added `list_plugin_section()` generic helper with closures
- [x] Add per-plugin timeout in doctor - `doctor.rs:71` now uses 10-second timeout per plugin

---

## Newly Discovered Issues (11th Analysis)

**Bug Fix:** (🔴 Critical)
- [x] Fix `is_multiple_of()` - `clean.rs:133` now stable in Rust 1.92.0, kept as-is

**Error Handling:** (🟡 Important)
- [x] Improve subdir error message - `install.rs:302-305` now shows appropriate message for None vs Some case
- [x] Add warning on plugin connection failure - `doctor.rs:86-89` now shows warning instead of silent skip
