//! Plugin installation logic

use crate::output;
use crate::plugins::{
    detect_plugin_type, get_registry_path, DetectedPluginType, PluginCapabilities, PluginEntry, PluginRegistry,
    PluginSource, PluginType,
};
use fs2::FileExt;
use hodu_plugin::PLUGIN_VERSION;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::process::Command;
use tempfile::TempDir;
use wait_timeout::ChildExt;

/// RAII guard for lock file cleanup
///
/// Ensures the lock file is removed when installation completes (success or failure).
/// This prevents stale lock files from blocking future installations.
struct LockFileGuard {
    path: PathBuf,
    #[allow(dead_code)]
    file: File, // Keep file open to maintain lock
}

impl LockFileGuard {
    fn new(path: PathBuf, file: File) -> Self {
        Self { path, file }
    }
}

impl Drop for LockFileGuard {
    fn drop(&mut self) {
        // Best effort cleanup - log warning on failure
        if let Err(e) = std::fs::remove_file(&self.path) {
            // Only warn if file exists but couldn't be removed
            if self.path.exists() {
                eprintln!("Warning: Failed to remove lock file: {}", e);
            }
        }
    }
}

/// Maximum manifest.json file size (1MB)
const MAX_MANIFEST_SIZE: u64 = 1024 * 1024;

/// Parsed manifest info: (name, version, plugin_version, plugin_type, capabilities)
type ManifestInfo = (String, String, String, PluginType, PluginCapabilities);

/// Official plugin registry URL
pub const PLUGIN_REGISTRY_URL: &str = "https://raw.githubusercontent.com/daminstudio/hodu-plugins/main/plugins.toml";

/// Version entry in the registry
#[derive(Debug, serde::Deserialize)]
pub struct PluginVersionEntry {
    pub version: String,
    pub tag: String,
    /// Plugin protocol version requirement (e.g., "0.1" means compatible with 0.1.x)
    pub plugin: String,
}

/// Plugin entry in the registry
#[derive(Debug, serde::Deserialize)]
pub struct RegistryPlugin {
    pub name: String,
    pub description: Option<String>,
    pub git: String,
    pub path: Option<String>,
    pub versions: Vec<PluginVersionEntry>,
}

/// Registry file structure
#[derive(Debug, serde::Deserialize)]
pub struct PluginRegistryFile {
    pub plugin: Vec<RegistryPlugin>,
}

pub fn install_from_registry(
    name_with_version: &str,
    tag_override: Option<&str>,
    debug: bool,
    force: bool,
    verbose: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    // Parse name@version syntax
    let (name, requested_version) = if let Some(at_pos) = name_with_version.find('@') {
        let n = &name_with_version[..at_pos];
        let v = &name_with_version[at_pos + 1..];
        (n, Some(v))
    } else {
        (name_with_version, None)
    };

    output::fetching(&format!("plugin registry for '{}'", name));

    // Fetch registry
    let body = ureq::get(PLUGIN_REGISTRY_URL)
        .call()
        .map_err(|e| format!("Failed to fetch plugin registry: {}", e))?
        .body_mut()
        .read_to_string()
        .map_err(|e| format!("Failed to read registry: {}", e))?;

    let registry: PluginRegistryFile = toml::from_str(&body).map_err(|e| format!("Failed to parse registry: {}", e))?;

    // Find plugin
    let plugin = registry.plugin.iter().find(|p| p.name == name).ok_or_else(|| {
        let available: Vec<_> = registry
            .plugin
            .iter()
            .map(|p| {
                if let Some(desc) = &p.description {
                    format!("{} - {}", p.name, desc)
                } else {
                    p.name.clone()
                }
            })
            .collect();
        format!(
            "Plugin '{}' not found in registry.\n\nAvailable plugins:\n  {}",
            name,
            available.join("\n  ")
        )
    })?;

    // Get host plugin version (major.minor)
    let host_version_parts: Vec<&str> = PLUGIN_VERSION.split('.').collect();
    let host_major_minor = if host_version_parts.len() >= 2 {
        format!("{}.{}", host_version_parts[0], host_version_parts[1])
    } else {
        PLUGIN_VERSION.to_string()
    };

    // Filter compatible versions (same major.minor)
    let compatible_versions: Vec<_> = plugin
        .versions
        .iter()
        .filter(|v| v.plugin == host_major_minor)
        .collect();

    // Determine the tag to use
    let tag = if let Some(t) = tag_override {
        // --tag flag takes precedence
        Some(t.to_string())
    } else if let Some(ver) = requested_version {
        // @version syntax
        if ver == "latest" {
            // Use the first compatible version (latest)
            if compatible_versions.is_empty() {
                return Err(format!(
                    "No compatible version found for plugin protocol {}.\n\nAvailable versions:\n  {}",
                    host_major_minor,
                    plugin
                        .versions
                        .iter()
                        .map(|v| format!("{} (protocol {})", v.version, v.plugin))
                        .collect::<Vec<_>>()
                        .join("\n  ")
                )
                .into());
            }
            compatible_versions.first().map(|v| v.tag.clone())
        } else {
            // Find specific version
            let version_entry = plugin.versions.iter().find(|v| v.version == ver).ok_or_else(|| {
                let available: Vec<_> = plugin.versions.iter().map(|v| v.version.as_str()).collect();
                format!(
                    "Version '{}' not found for plugin '{}'.\n\nAvailable versions:\n  {}",
                    ver,
                    name,
                    available.join("\n  ")
                )
            })?;
            Some(version_entry.tag.clone())
        }
    } else {
        // No version specified, use latest compatible
        if compatible_versions.is_empty() {
            return Err(format!(
                "No compatible version found for plugin protocol {}.\n\nAvailable versions:\n  {}",
                host_major_minor,
                plugin
                    .versions
                    .iter()
                    .map(|v| format!("{} (protocol {})", v.version, v.plugin))
                    .collect::<Vec<_>>()
                    .join("\n  ")
            )
            .into());
        }
        compatible_versions.first().map(|v| v.tag.clone())
    };

    install_from_git(
        &plugin.git,
        plugin.path.as_deref(),
        tag.as_deref(),
        debug,
        force,
        verbose,
    )
}

/// Validate git URL format
fn validate_git_url(url: &str) -> Result<(), Box<dyn std::error::Error>> {
    // Allow common git URL patterns
    let valid = url.starts_with("https://")
        || url.starts_with("http://")
        || url.starts_with("git@")
        || url.starts_with("git://")
        || url.starts_with("ssh://");

    if !valid {
        return Err(format!(
            "Invalid git URL: {}. Expected https://, git@, git://, or ssh:// URL",
            url
        )
        .into());
    }

    // Warn about insecure protocols (unencrypted)
    if url.starts_with("http://") {
        eprintln!("Warning: Using insecure HTTP protocol. Consider using HTTPS instead.");
    } else if url.starts_with("git://") {
        eprintln!("Warning: Using insecure git:// protocol (unencrypted). Consider using HTTPS or SSH.");
    }

    // Check for suspicious patterns
    if url.contains("..") || url.contains('\0') {
        return Err("Git URL contains suspicious characters".into());
    }

    Ok(())
}

pub fn install_from_git(
    url: &str,
    subdir: Option<&str>,
    tag: Option<&str>,
    debug: bool,
    force: bool,
    verbose: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    // Validate URL before cloning
    validate_git_url(url)?;

    // Create temp directory with tempfile crate for secure, atomic creation
    let temp_dir =
        TempDir::with_prefix("hodu_plugin_").map_err(|e| format!("Failed to create temp directory: {}", e))?;

    // Clone repository with timeout
    let mut git_cmd = Command::new("git");
    git_cmd.arg("clone");
    if !verbose {
        git_cmd.arg("-q"); // Quiet unless verbose
    }
    if tag.is_none() {
        git_cmd.arg("--depth").arg("1");
    }
    git_cmd.arg(url).arg(temp_dir.path());

    const GIT_CLONE_TIMEOUT_SECS: u64 = 300; // 5 minutes
    let mut child = git_cmd.spawn()?;
    let status = match child.wait_timeout(std::time::Duration::from_secs(GIT_CLONE_TIMEOUT_SECS))? {
        Some(status) => status,
        None => {
            // Timeout - kill the process
            let _ = child.kill();
            let _ = child.wait();
            return Err(format!(
                "Git clone timed out after {} seconds. Repository may be too large or network is slow: {}",
                GIT_CLONE_TIMEOUT_SECS, url
            )
            .into());
        },
    };
    if !status.success() {
        return Err(format!("Failed to clone repository: {}", url).into());
    }

    // Checkout tag/branch if specified (quietly)
    if let Some(t) = tag {
        let status = Command::new("git")
            .arg("checkout")
            .arg("-q")
            .arg(t)
            .current_dir(temp_dir.path())
            .status()?;
        if !status.success() {
            return Err(format!("Failed to checkout: {}", t).into());
        }
    }

    // Determine install path (with optional subdir)
    let install_path = match subdir {
        Some(s) => temp_dir.path().join(s),
        None => temp_dir.path().to_path_buf(),
    };

    if !install_path.exists() {
        return Err(format!("Subdirectory '{}' not found in repository", subdir.unwrap_or("")).into());
    }

    // Install from the cloned path
    let source = PluginSource::Git {
        url: url.to_string(),
        tag: tag.map(|t| t.to_string()),
        subdir: subdir.map(|s| s.to_string()),
    };
    install_from_path(&install_path, debug, force, verbose, source)
    // temp_dir is automatically cleaned up when dropped
}

pub fn install_from_path(
    path: &Path,
    debug: bool,
    force: bool,
    verbose: bool,
    source: PluginSource,
) -> Result<(), Box<dyn std::error::Error>> {
    let path = path.canonicalize()?;

    // Check if it's a Cargo project
    let cargo_toml = path.join("Cargo.toml");
    if !cargo_toml.exists() {
        return Err(format!("No Cargo.toml found at {}", path.display()).into());
    }

    // Check Cargo.toml size before reading (reuse manifest size limit)
    let cargo_size = std::fs::metadata(&cargo_toml)?.len();
    if cargo_size > MAX_MANIFEST_SIZE {
        return Err(format!(
            "Cargo.toml too large: {} bytes (max: {} bytes)",
            cargo_size, MAX_MANIFEST_SIZE
        )
        .into());
    }

    // Parse Cargo.toml to get the package name
    let cargo_content = std::fs::read_to_string(&cargo_toml)?;
    let package_name = parse_package_name(&cargo_content)
        .ok_or_else(|| format!("Could not find package name in {}", cargo_toml.display()))?;

    // Build the plugin with cargo (as executable)
    output::compiling(&package_name);
    let mut cargo_cmd = Command::new("cargo");
    cargo_cmd.arg("build");
    cargo_cmd.arg("-p").arg(&package_name);
    if !debug {
        cargo_cmd.arg("--release");
    }
    if !verbose {
        cargo_cmd.arg("-q"); // Quiet unless verbose
    }
    cargo_cmd.current_dir(&path);

    let cmd_output = cargo_cmd.output()?;
    if !cmd_output.status.success() {
        output::error("build failed");
        let stderr = String::from_utf8_lossy(&cmd_output.stderr);
        let stdout = String::from_utf8_lossy(&cmd_output.stdout);
        let mut msg = String::from("Failed to build plugin:");
        if !stderr.is_empty() {
            msg.push_str("\n--- stderr ---\n");
            msg.push_str(&stderr);
        }
        if !stdout.is_empty() {
            msg.push_str("\n--- stdout ---\n");
            msg.push_str(&stdout);
        }
        return Err(msg.into());
    }

    // Find the built executable
    let profile = if debug { "debug" } else { "release" };

    // Try multiple possible target directories
    let possible_target_dirs = vec![
        path.join("target").join(profile),
        path.parent()
            .map(|p| p.join("target").join(profile))
            .unwrap_or_default(),
    ];

    let mut bin_path = None;
    for target_dir in &possible_target_dirs {
        if !target_dir.exists() {
            continue;
        }

        // Look for executable matching the package name
        let candidate = target_dir.join(&package_name);
        if candidate.exists() {
            bin_path = Some(candidate);
            break;
        }

        // On Windows, add .exe
        #[cfg(windows)]
        {
            let candidate = target_dir.join(format!("{}.exe", package_name));
            if candidate.exists() {
                bin_path = Some(candidate);
                break;
            }
        }
    }

    let bin_path = bin_path.ok_or_else(|| {
        format!(
            "No executable found for package '{}'. Checked: {:?}",
            package_name, possible_target_dirs
        )
    })?;

    // Read manifest.json if it exists, or detect from binary
    let manifest_path = path.join("manifest.json");
    let (name, version, plugin_version, plugin_type, capabilities) = if manifest_path.exists() {
        parse_manifest(&manifest_path, &package_name)?
    } else {
        // Try to detect from binary (spawn and initialize)
        let detected = detect_plugin_type(&bin_path)?;
        match detected {
            DetectedPluginType::Backend {
                name,
                version,
                plugin_version,
            } => {
                let capabilities = PluginCapabilities::backend(true, false, vec![], vec![]);
                (name, version, plugin_version, PluginType::Backend, capabilities)
            },
            DetectedPluginType::ModelFormat {
                name,
                version,
                plugin_version,
            } => {
                let capabilities = PluginCapabilities::model_format(true, false, vec![]);
                (name, version, plugin_version, PluginType::ModelFormat, capabilities)
            },
            DetectedPluginType::TensorFormat {
                name,
                version,
                plugin_version,
            } => {
                let capabilities = PluginCapabilities::tensor_format(true, false, vec![]);
                (name, version, plugin_version, PluginType::TensorFormat, capabilities)
            },
        }
    };

    // Check plugin protocol version compatibility
    let host_parts: Vec<u32> = PLUGIN_VERSION
        .split('.')
        .map(|s| {
            s.parse::<u32>()
                .map_err(|_| format!("Invalid host version component: '{}'", s))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let plugin_parts: Vec<u32> = plugin_version
        .split('.')
        .map(|s| {
            s.parse::<u32>().map_err(|_| {
                format!(
                    "Invalid plugin version component: '{}' in version '{}'",
                    s, plugin_version
                )
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    if host_parts.len() >= 2 && plugin_parts.len() >= 2 {
        let (host_major, host_minor) = (host_parts[0], host_parts[1]);
        let (plugin_major, plugin_minor) = (plugin_parts[0], plugin_parts[1]);

        if host_major != plugin_major {
            return Err(format!(
                "Plugin protocol major version mismatch: host={}, plugin={}",
                PLUGIN_VERSION, plugin_version
            )
            .into());
        }

        if host_minor < plugin_minor {
            return Err(format!(
                "Plugin requires newer protocol version: host={}, plugin={}",
                PLUGIN_VERSION, plugin_version
            )
            .into());
        }
    }

    // Acquire lock BEFORE loading registry to prevent race conditions
    let registry_path = get_registry_path()?;
    let lock_path = registry_path.with_extension("lock");
    let lock_file = File::create(&lock_path).map_err(|e| format!("Failed to create lock file: {}", e))?;
    lock_file
        .lock_exclusive()
        .map_err(|e| format!("Failed to acquire lock (another installation in progress?): {}", e))?;
    // RAII guard ensures lock file is cleaned up on function exit (success or failure)
    let _lock_guard = LockFileGuard::new(lock_path, lock_file);
    // Now load registry while holding the lock
    let mut registry = PluginRegistry::load(&registry_path)?;

    // Check if already installed
    if let Some(existing) = registry.find(&name) {
        if !force {
            return Err(format!(
                "Plugin {} v{} is already installed. Use --force to reinstall.",
                existing.name, existing.version
            )
            .into());
        }
    }

    // Copy executable to plugins directory
    let plugins_dir = get_plugins_dir()?;
    let plugin_dir = plugins_dir.join(&name);
    std::fs::create_dir_all(&plugin_dir)?;

    let bin_filename = bin_path
        .file_name()
        .ok_or("Invalid binary path: no filename")?
        .to_string_lossy()
        .to_string();
    let dest_path = plugin_dir.join(&bin_filename);

    // Backup existing binary for rollback (if reinstalling)
    let backup_path = dest_path.with_extension("backup");
    let has_backup = if dest_path.exists() && force {
        std::fs::rename(&dest_path, &backup_path)?;
        true
    } else {
        false
    };

    // Copy new binary (with rollback on failure)
    if let Err(e) = std::fs::copy(&bin_path, &dest_path) {
        if has_backup {
            // Restore backup - fail if restoration fails to avoid broken state
            if let Err(restore_err) = std::fs::rename(&backup_path, &dest_path) {
                return Err(format!(
                    "Failed to copy plugin binary: {}. Additionally, failed to restore backup: {}. \
                     Plugin may be in broken state - manual intervention required.",
                    e, restore_err
                )
                .into());
            }
        }
        return Err(format!("Failed to copy plugin binary: {}", e).into());
    }

    // Remove backup after successful copy
    if has_backup {
        if let Err(e) = std::fs::remove_file(&backup_path) {
            output::warning(&format!(
                "Failed to remove backup file: {}. You may want to delete it manually.",
                e
            ));
        }
    }

    // Make executable on Unix
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(&dest_path)?.permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&dest_path, perms)?;
    }

    // Copy manifest.json if it exists (needed for runtime target checking)
    if manifest_path.exists() {
        let dest_manifest = plugin_dir.join("manifest.json");
        std::fs::copy(&manifest_path, &dest_manifest)?;
    }

    // Parse metadata from manifest if available (with size limit check)
    let (description, license, dependencies) = if manifest_path.exists() {
        let manifest_content = read_manifest_checked(&manifest_path)?;
        let manifest: serde_json::Value = serde_json::from_str(&manifest_content)?;
        let desc = manifest["description"].as_str().map(String::from);
        let lic = manifest["license"].as_str().map(String::from);
        let deps = manifest["dependencies"]
            .as_array()
            .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
            .unwrap_or_default();
        (desc, lic, deps)
    } else {
        (None, None, Vec::new())
    };

    // Create registry entry
    let entry = PluginEntry {
        name: name.clone(),
        version: version.clone(),
        description,
        license,
        plugin_type,
        capabilities,
        binary: bin_filename,
        source,
        installed_at: chrono_now(),
        plugin_version,
        enabled: true,
        dependencies: dependencies.clone(),
    };

    // Update registry
    registry.upsert(entry);
    registry.save(&registry_path)?;

    // Check dependencies after installation
    if !dependencies.is_empty() {
        if let Err(missing) = registry.check_dependencies(&name) {
            eprintln!("Warning: Missing dependencies for {}: {}", name, missing.join(", "));
        }
    }

    output::installed(&format!("{} v{}", name, version));
    Ok(())
}

/// Read manifest file with size limit check
fn read_manifest_checked(path: &Path) -> Result<String, Box<dyn std::error::Error>> {
    let metadata = std::fs::metadata(path)?;
    if metadata.len() > MAX_MANIFEST_SIZE {
        return Err(format!(
            "manifest.json too large: {} bytes (max: {} bytes)",
            metadata.len(),
            MAX_MANIFEST_SIZE
        )
        .into());
    }
    Ok(std::fs::read_to_string(path)?)
}

fn parse_manifest(manifest_path: &Path, package_name: &str) -> Result<ManifestInfo, Box<dyn std::error::Error>> {
    let manifest_content = read_manifest_checked(manifest_path)?;
    let manifest: serde_json::Value = serde_json::from_str(&manifest_content)?;

    let name = manifest["name"].as_str().unwrap_or(package_name).to_string();
    let version = manifest["version"].as_str().unwrap_or("0.1.0").to_string();
    let plugin_version = manifest["plugin_version"]
        .as_str()
        .unwrap_or(PLUGIN_VERSION)
        .to_string();

    let caps = manifest["capabilities"].as_array();
    let is_backend = caps
        .map(|c| {
            c.iter()
                .any(|v| v.as_str().map(|s| s.starts_with("backend.")).unwrap_or(false))
        })
        .unwrap_or(false);

    // Determine plugin type and capabilities from manifest
    let has_model_caps = caps
        .map(|c| {
            c.iter().any(|v| {
                v.as_str()
                    .map(|s| s == "format.load_model" || s == "format.save_model")
                    .unwrap_or(false)
            })
        })
        .unwrap_or(false);
    let has_tensor_caps = caps
        .map(|c| {
            c.iter().any(|v| {
                v.as_str()
                    .map(|s| s == "format.load_tensor" || s == "format.save_tensor")
                    .unwrap_or(false)
            })
        })
        .unwrap_or(false);

    let extensions: Vec<String> = manifest["extensions"]
        .as_array()
        .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
        .unwrap_or_default();

    let (plugin_type, capabilities) = if is_backend {
        let devices: Vec<String> = manifest["devices"]
            .as_array()
            .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
            .unwrap_or_default();
        let runner = caps
            .map(|c| c.iter().any(|v| v.as_str() == Some("backend.run")))
            .unwrap_or(false);
        let builder = caps
            .map(|c| c.iter().any(|v| v.as_str() == Some("backend.build")))
            .unwrap_or(false);
        (
            PluginType::Backend,
            PluginCapabilities::backend(runner, builder, devices, vec![]),
        )
    } else if has_model_caps {
        let load_model = caps
            .map(|c| c.iter().any(|v| v.as_str() == Some("format.load_model")))
            .unwrap_or(false);
        let save_model = caps
            .map(|c| c.iter().any(|v| v.as_str() == Some("format.save_model")))
            .unwrap_or(false);
        (
            PluginType::ModelFormat,
            PluginCapabilities::model_format(load_model, save_model, extensions),
        )
    } else if has_tensor_caps {
        let load_tensor = caps
            .map(|c| c.iter().any(|v| v.as_str() == Some("format.load_tensor")))
            .unwrap_or(false);
        let save_tensor = caps
            .map(|c| c.iter().any(|v| v.as_str() == Some("format.save_tensor")))
            .unwrap_or(false);
        (
            PluginType::TensorFormat,
            PluginCapabilities::tensor_format(load_tensor, save_tensor, extensions),
        )
    } else {
        // No recognized capabilities - reject invalid manifest instead of silent fallback
        return Err("Invalid manifest.json: no recognized capabilities found. \
             Expected one or more of: backend.run, backend.build, format.load_model, \
             format.save_model, format.load_tensor, format.save_tensor"
            .into());
    };

    Ok((name, version, plugin_version, plugin_type, capabilities))
}

pub fn chrono_now() -> String {
    chrono::Utc::now().to_rfc3339()
}

pub fn get_plugins_dir() -> Result<PathBuf, Box<dyn std::error::Error>> {
    let home = dirs::home_dir().ok_or("Could not find home directory")?;
    let plugins_dir = home.join(".hodu").join("plugins");

    if !plugins_dir.exists() {
        std::fs::create_dir_all(&plugins_dir)?;
    }

    Ok(plugins_dir)
}

/// Parse package name from Cargo.toml content using proper TOML parsing
pub fn parse_package_name(content: &str) -> Option<String> {
    #[derive(serde::Deserialize)]
    struct CargoToml {
        package: Option<Package>,
    }
    #[derive(serde::Deserialize)]
    struct Package {
        name: Option<String>,
    }

    let cargo: CargoToml = toml::from_str(content).ok()?;
    cargo.package?.name
}

/// Fetch official registry
pub fn fetch_official_registry() -> Result<PluginRegistryFile, Box<dyn std::error::Error>> {
    let body = ureq::get(PLUGIN_REGISTRY_URL)
        .call()
        .map_err(|e| format!("Failed to fetch plugin registry: {}", e))?
        .body_mut()
        .read_to_string()
        .map_err(|e| format!("Failed to read registry: {}", e))?;

    let registry: PluginRegistryFile = toml::from_str(&body).map_err(|e| format!("Failed to parse registry: {}", e))?;
    Ok(registry)
}
