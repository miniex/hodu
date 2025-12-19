//! Backend plugin types
//!
//! Types for backend plugins that execute models on various devices.
//! Common types (Device, BuildTarget, current_host_triple) are re-exported from hodu_plugin at crate root.

use hodu_plugin::current_host_triple;
use serde::{Deserialize, Serialize};
use std::process::Command;

// ============================================================================
// Build Target Capability (plugin SDK specific)
// ============================================================================

/// Supported target definition for manifest.json
///
/// # Example manifest.json
/// ```json
/// {
///   "supported_targets": [
///     {
///       "triple": "x86_64-unknown-linux-gnu",
///       "requires": ["clang|gcc"]
///     },
///     {
///       "triple": "aarch64-apple-darwin",
///       "requires": ["clang"],
///       "host_only": ["*-apple-darwin"]
///     }
///   ]
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SupportedTarget {
    /// Target triple (e.g., "x86_64-unknown-linux-gnu")
    pub triple: String,

    /// Required tools (e.g., ["clang", "nvcc", "xcrun"])
    /// Use "|" for alternatives: "clang|gcc" means clang OR gcc
    #[serde(default)]
    pub requires: Vec<String>,

    /// Host triples that can build this target (glob patterns)
    /// Empty means any host can build (with proper toolchain)
    /// e.g., ["*-apple-darwin"] means only macOS hosts
    #[serde(default)]
    pub host_only: Vec<String>,
}

/// Result of checking if a target can be built
#[derive(Debug, Clone)]
pub struct BuildCapability {
    /// Whether the target can be built from current host
    pub can_build: bool,
    /// Tools that were found and will be used
    pub available_tools: Vec<String>,
    /// Missing tools that need to be installed
    pub missing_tools: Vec<String>,
    /// Why the build is not possible (if can_build is false)
    pub reason: Option<String>,
}

impl BuildCapability {
    /// Create a capability indicating build is available
    pub fn available(tools: Vec<String>) -> Self {
        Self {
            can_build: true,
            available_tools: tools,
            missing_tools: Vec::new(),
            reason: None,
        }
    }

    /// Create a capability indicating build is not available
    pub fn unavailable(missing: Vec<String>, reason: impl Into<String>) -> Self {
        Self {
            can_build: false,
            available_tools: Vec::new(),
            missing_tools: missing,
            reason: Some(reason.into()),
        }
    }
}

// ============================================================================
// Host / Tool Detection Utilities
// ============================================================================

/// Check if a tool is available on the system
///
/// Tool names are validated to prevent command injection:
/// - Must not be empty
/// - Must not contain path separators (`/`, `\`)
/// - Must not contain shell metacharacters
/// - Must not contain null bytes or control characters
///
/// Returns `false` for invalid tool names without executing anything.
pub fn is_tool_available(tool: &str) -> bool {
    // Validate tool name to prevent command injection
    if tool.is_empty() {
        return false;
    }
    // Reject path separators (prevents executing arbitrary paths)
    if tool.contains('/') || tool.contains('\\') {
        return false;
    }
    // Reject shell metacharacters and control characters
    const FORBIDDEN_CHARS: &[char] = &[
        '\0', '\n', '\r', '\t', ' ', '&', '|', ';', '$', '`', '(', ')', '{', '}', '[', ']', '<', '>', '\'', '"', '!',
        '*', '?', '#', '~', '^',
    ];
    if tool.chars().any(|c| c.is_control() || FORBIDDEN_CHARS.contains(&c)) {
        return false;
    }

    Command::new(tool)
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Check if host matches a pattern (supports glob * wildcards)
///
/// Supports wildcards at any position:
/// - `*` matches everything
/// - `*-apple-darwin` matches suffix
/// - `x86_64-*` matches prefix
/// - `aarch64-*-darwin` matches prefix and suffix with any middle
///
/// # Examples
/// - `host_matches_pattern("aarch64-apple-darwin", "*-apple-darwin")` -> true
/// - `host_matches_pattern("x86_64-linux-gnu", "x86_64-*")` -> true
/// - `host_matches_pattern("aarch64-linux-gnu", "*")` -> true
/// - `host_matches_pattern("aarch64-apple-darwin", "aarch64-*-darwin")` -> true
/// - `host_matches_pattern("x86_64-unknown-linux-gnu", "*-linux-*")` -> true
pub fn host_matches_pattern(host: &str, pattern: &str) -> bool {
    if pattern == "*" {
        return true;
    }

    // Split pattern by '*' and match each part sequentially
    let parts: Vec<&str> = pattern.split('*').collect();

    // Single part means no wildcard - exact match
    if parts.len() == 1 {
        return host == pattern;
    }

    let mut remaining = host;

    for (i, part) in parts.iter().enumerate() {
        if part.is_empty() {
            continue;
        }

        if i == 0 {
            // First part must be a prefix
            if !remaining.starts_with(part) {
                return false;
            }
            remaining = &remaining[part.len()..];
        } else if i == parts.len() - 1 {
            // Last part must be a suffix
            if !remaining.ends_with(part) {
                return false;
            }
        } else {
            // Middle parts can appear anywhere in remaining
            if let Some(pos) = remaining.find(part) {
                remaining = &remaining[pos + part.len()..];
            } else {
                return false;
            }
        }
    }

    true
}

/// Plugin manifest (manifest.json)
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PluginManifest {
    pub name: String,
    pub version: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub license: String,
    #[serde(default)]
    pub plugin_version: String,
    #[serde(default)]
    pub capabilities: Vec<String>,
    #[serde(default)]
    pub devices: Vec<String>,
    #[serde(default)]
    pub extensions: Vec<String>,
    #[serde(default)]
    pub dependencies: Vec<String>,
    #[serde(default)]
    pub supported_targets: Vec<SupportedTarget>,
}

impl PluginManifest {
    /// Load manifest from the standard location (next to executable)
    pub fn load() -> Result<Self, String> {
        let exe_path = std::env::current_exe().map_err(|e| format!("Failed to get executable path: {}", e))?;

        let manifest_path = exe_path
            .parent()
            .map(|p| p.join("manifest.json"))
            .ok_or("No parent directory")?;

        Self::load_from(&manifest_path)
    }

    /// Load manifest from a specific path
    pub fn load_from(path: &std::path::Path) -> Result<Self, String> {
        let content = std::fs::read_to_string(path).map_err(|e| format!("Failed to read manifest: {}", e))?;

        serde_json::from_str(&content).map_err(|e| format!("Failed to parse manifest: {}", e))
    }

    /// Check if a target triple is supported and can be built
    pub fn check_target(&self, triple: &str) -> BuildCapability {
        // Find target in manifest
        let target = match self.supported_targets.iter().find(|t| t.triple == triple) {
            Some(t) => t,
            None => {
                return BuildCapability::unavailable(
                    vec![],
                    format!(
                        "Target '{}' is not in supported_targets. Available: {}",
                        triple,
                        self.supported_targets
                            .iter()
                            .map(|t| t.triple.as_str())
                            .collect::<Vec<_>>()
                            .join(", ")
                    ),
                );
            },
        };

        check_build_capability(target)
    }

    /// Get list of buildable targets from current host
    pub fn buildable_targets(&self) -> Vec<(&SupportedTarget, BuildCapability)> {
        self.supported_targets
            .iter()
            .map(|t| (t, check_build_capability(t)))
            .collect()
    }

    /// Format supported targets with build status
    pub fn format_targets(&self) -> String {
        let mut result = String::new();
        let host = current_host_triple();

        result.push_str(&format!("Host: {}\n\n", host));

        for target in &self.supported_targets {
            let cap = check_build_capability(target);
            let status = if cap.can_build { "✓" } else { "✗" };
            result.push_str(&format!("  {} {}\n", status, target.triple));
        }

        result
    }
}

/// Check build capability for a supported target
///
/// This checks:
/// 1. If current host is allowed to build this target (host_only)
/// 2. If required tools are available
pub fn check_build_capability(target: &SupportedTarget) -> BuildCapability {
    let host = current_host_triple();

    // Check host restriction
    if !target.host_only.is_empty() {
        let host_allowed = target.host_only.iter().any(|p| host_matches_pattern(host, p));
        if !host_allowed {
            // Limit hosts list to prevent huge error messages
            const MAX_HOSTS_SHOWN: usize = 10;
            let hosts_display = if target.host_only.len() > MAX_HOSTS_SHOWN {
                format!(
                    "{} (and {} more)",
                    target.host_only[..MAX_HOSTS_SHOWN].join(", "),
                    target.host_only.len() - MAX_HOSTS_SHOWN
                )
            } else {
                target.host_only.join(", ")
            };
            return BuildCapability::unavailable(
                vec![],
                format!(
                    "Host '{}' cannot build target '{}'. Allowed hosts: {}",
                    host, target.triple, hosts_display
                ),
            );
        }
    }

    // Check required tools
    let mut missing = Vec::new();
    let mut available = Vec::new();

    for req in &target.requires {
        // Handle alternatives (e.g., "clang|gcc")
        let alternatives: Vec<&str> = req.split('|').collect();
        let mut found = None;

        for alt in &alternatives {
            if is_tool_available(alt) {
                found = Some(alt.to_string());
                break;
            }
        }

        if let Some(tool) = found {
            available.push(tool);
        } else {
            missing.push(req.clone());
        }
    }

    if !missing.is_empty() {
        return BuildCapability::unavailable(
            missing.clone(),
            format!("Missing required tools: {}", missing.join(", ")),
        );
    }

    BuildCapability::available(available)
}
