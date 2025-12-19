//! Backend plugin types
//!
//! Types for backend plugins that execute models on various devices.

use std::fmt;

/// Target device for plugin execution
///
/// Using String for extensibility - plugins can define custom devices.
/// Convention: lowercase with `::` separator for device index.
/// Common values: "cpu", "cuda::0", "metal", "vulkan", "webgpu", "rocm::0"
pub type Device = String;

/// Parse device ID from device string (e.g., "cuda::0" -> 0)
///
/// Returns `Some(id)` if the device string contains a valid numeric ID after `::`.
/// Returns `None` for devices without an ID (e.g., "cpu", "metal") or invalid formats.
///
/// # Examples
/// ```
/// use hodu_plugin::parse_device_id;
/// assert_eq!(parse_device_id("cuda::0"), Some(0));
/// assert_eq!(parse_device_id("cpu"), None);
/// ```
pub fn parse_device_id(device: &str) -> Option<usize> {
    let parts: Vec<&str> = device.split("::").collect();
    // Only accept exactly 2 parts (type::id)
    if parts.len() == 2 {
        parts[1].parse().ok()
    } else {
        None
    }
}

/// Get device type from device string (e.g., "cuda::0" -> "cuda")
///
/// Returns the device type portion before the `::` separator, or the entire
/// string if no separator is present. Returns `None` for empty strings.
///
/// # Examples
/// ```
/// use hodu_plugin::device_type;
/// assert_eq!(device_type("cuda::0"), Some("cuda"));
/// assert_eq!(device_type("cpu"), Some("cpu"));
/// assert_eq!(device_type(""), None);
/// ```
pub fn device_type(device: &str) -> Option<&str> {
    if device.is_empty() {
        return None;
    }
    // split().next() always returns Some for non-empty string, but we avoid unwrap for cleaner code
    device.split("::").next()
}

/// Error type for BuildTarget validation
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BuildTargetError {
    /// Triple string is empty
    EmptyTriple,
    /// Device string is empty
    EmptyDevice,
    /// Triple format is invalid (should contain at least one hyphen)
    InvalidTripleFormat,
}

impl fmt::Display for BuildTargetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyTriple => write!(f, "Target triple cannot be empty"),
            Self::EmptyDevice => write!(f, "Device cannot be empty"),
            Self::InvalidTripleFormat => write!(f, "Invalid triple format (expected 'arch-vendor-os' pattern)"),
        }
    }
}

impl std::error::Error for BuildTargetError {}

/// Build target specification for AOT compilation
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BuildTarget {
    /// Target triple (e.g., "x86_64-unknown-linux-gnu", "aarch64-apple-darwin")
    pub triple: String,
    /// Target device (e.g., "cpu", "metal", "cuda::0")
    pub device: String,
}

impl BuildTarget {
    /// Create a new build target without validation
    pub fn new(triple: impl Into<String>, device: impl Into<String>) -> Self {
        Self {
            triple: triple.into(),
            device: device.into(),
        }
    }

    /// Create a new build target with validation
    ///
    /// Returns an error if:
    /// - Triple is empty or has invalid format
    /// - Device is empty
    ///
    /// # Triple Format
    ///
    /// Valid triple formats: `<arch>-<vendor>-<os>` or `<arch>-<vendor>-<os>-<env>`
    ///
    /// Examples: `x86_64-unknown-linux-gnu`, `aarch64-apple-darwin`
    ///
    /// # Special Case: "unknown"
    ///
    /// The literal string `"unknown"` is allowed as a special case for scenarios
    /// where the target triple is not yet determined or when operating in a
    /// platform-agnostic mode. This is useful for:
    /// - Build scripts that defer target detection
    /// - Testing environments where the target doesn't matter
    /// - Fallback cases when `current_host_triple()` fails
    pub fn new_checked(triple: impl Into<String>, device: impl Into<String>) -> Result<Self, BuildTargetError> {
        let triple = triple.into();
        let device = device.into();

        if triple.is_empty() {
            return Err(BuildTargetError::EmptyTriple);
        }

        // Allow "unknown" as special case, otherwise require valid triple format
        if triple != "unknown" {
            let parts: Vec<&str> = triple.split('-').collect();
            // Must have at least 3 parts: arch-vendor-os (optionally -env)
            if parts.len() < 3 {
                return Err(BuildTargetError::InvalidTripleFormat);
            }
            // Each part must be non-empty and alphanumeric (with underscores allowed)
            for part in &parts {
                if part.is_empty() || !part.chars().all(|c| c.is_alphanumeric() || c == '_') {
                    return Err(BuildTargetError::InvalidTripleFormat);
                }
            }
        }

        if device.is_empty() {
            return Err(BuildTargetError::EmptyDevice);
        }

        Ok(Self { triple, device })
    }

    /// Create a build target for the current host system
    pub fn host(device: impl Into<String>) -> Self {
        Self::new(current_host_triple(), device)
    }
}

/// Get the current host triple
pub fn current_host_triple() -> &'static str {
    #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
    return "x86_64-unknown-linux-gnu";
    #[cfg(all(target_arch = "aarch64", target_os = "linux"))]
    return "aarch64-unknown-linux-gnu";
    #[cfg(all(target_arch = "x86_64", target_os = "macos"))]
    return "x86_64-apple-darwin";
    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    return "aarch64-apple-darwin";
    #[cfg(all(target_arch = "x86_64", target_os = "windows"))]
    return "x86_64-pc-windows-msvc";
    #[cfg(not(any(
        all(target_arch = "x86_64", target_os = "linux"),
        all(target_arch = "aarch64", target_os = "linux"),
        all(target_arch = "x86_64", target_os = "macos"),
        all(target_arch = "aarch64", target_os = "macos"),
        all(target_arch = "x86_64", target_os = "windows"),
    )))]
    return "unknown";
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_device_id() {
        assert_eq!(parse_device_id("cuda::0"), Some(0));
        assert_eq!(parse_device_id("cuda::1"), Some(1));
        assert_eq!(parse_device_id("rocm::2"), Some(2));
        assert_eq!(parse_device_id("cpu"), None);
        assert_eq!(parse_device_id("metal"), None);
        assert_eq!(parse_device_id("cuda::invalid"), None);
        // Malformed inputs should return None
        assert_eq!(parse_device_id("cuda::0::extra"), None);
        assert_eq!(parse_device_id(""), None);
    }

    #[test]
    fn test_device_type() {
        assert_eq!(device_type("cuda::0"), Some("cuda"));
        assert_eq!(device_type("cuda::1"), Some("cuda"));
        assert_eq!(device_type("rocm::0"), Some("rocm"));
        assert_eq!(device_type("cpu"), Some("cpu"));
        assert_eq!(device_type("metal"), Some("metal"));
        assert_eq!(device_type("webgpu"), Some("webgpu"));
        // Empty string should return None
        assert_eq!(device_type(""), None);
    }

    #[test]
    fn test_build_target_new() {
        let target = BuildTarget::new("x86_64-unknown-linux-gnu", "cuda::0");
        assert_eq!(target.triple, "x86_64-unknown-linux-gnu");
        assert_eq!(target.device, "cuda::0");
    }

    #[test]
    fn test_build_target_host() {
        let target = BuildTarget::host("cpu");
        assert_eq!(target.device, "cpu");
        assert!(!target.triple.is_empty());
    }

    #[test]
    fn test_current_host_triple() {
        let triple = current_host_triple();
        assert!(!triple.is_empty());
        // Should contain architecture and OS
        assert!(triple.contains('-'));
    }

    #[test]
    fn test_build_target_new_checked_valid() {
        let result = BuildTarget::new_checked("x86_64-unknown-linux-gnu", "cuda::0");
        assert!(result.is_ok());
        let target = result.unwrap();
        assert_eq!(target.triple, "x86_64-unknown-linux-gnu");
        assert_eq!(target.device, "cuda::0");
    }

    #[test]
    fn test_build_target_new_checked_empty_triple() {
        let result = BuildTarget::new_checked("", "cpu");
        assert_eq!(result.unwrap_err(), BuildTargetError::EmptyTriple);
    }

    #[test]
    fn test_build_target_new_checked_empty_device() {
        let result = BuildTarget::new_checked("x86_64-unknown-linux-gnu", "");
        assert_eq!(result.unwrap_err(), BuildTargetError::EmptyDevice);
    }

    #[test]
    fn test_build_target_new_checked_invalid_triple() {
        let result = BuildTarget::new_checked("invalid", "cpu");
        assert_eq!(result.unwrap_err(), BuildTargetError::InvalidTripleFormat);
    }

    #[test]
    fn test_build_target_new_checked_unknown_triple() {
        // "unknown" is allowed as a special case
        let result = BuildTarget::new_checked("unknown", "cpu");
        assert!(result.is_ok());
    }

    #[test]
    fn test_build_target_error_display() {
        assert_eq!(
            BuildTargetError::EmptyTriple.to_string(),
            "Target triple cannot be empty"
        );
        assert_eq!(BuildTargetError::EmptyDevice.to_string(), "Device cannot be empty");
    }
}
