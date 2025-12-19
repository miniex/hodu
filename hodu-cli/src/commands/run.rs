//! Run command - execute model inference using plugins
//!
//! This command uses JSON-RPC based plugins to load models and run inference.

use crate::output;
use crate::plugins::{backend_plugin_name, load_registry, PluginManager, PluginRegistry};
use crate::tensor::{load_tensor_data, load_tensor_file, save_outputs, save_tensor_data};
use crate::utils::{core_dtype_to_plugin, path_to_str, plugin_dtype_to_core};
use clap::Args;
use hodu_core::snapshot::Snapshot;
use hodu_core::tensor::Tensor;
use hodu_core::types::{Device as CoreDevice, Shape};
use hodu_plugin::rpc::TensorInput;
use hodu_plugin::{current_host_triple, Device, TensorData};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tempfile::NamedTempFile;

/// Minimum timeout in seconds
const MIN_TIMEOUT_SECS: u64 = 1;
/// Maximum timeout in seconds (1 hour)
const MAX_TIMEOUT_SECS: u64 = 3600;
/// Known device prefixes for validation
const KNOWN_DEVICE_PREFIXES: &[&str] = &["cpu", "metal", "cuda", "rocm", "vulkan", "directml"];

#[derive(Args)]
pub struct RunArgs {
    /// Model file (.onnx, .hdss, etc.)
    pub model: PathBuf,

    /// Input tensor (name=path), can be repeated
    #[arg(short, long = "input", value_name = "NAME=PATH")]
    pub input: Vec<String>,

    /// Input tensors (comma-separated: a=path,b=path)
    #[arg(long = "inputs", value_name = "INPUTS", value_delimiter = ',')]
    pub inputs: Vec<String>,

    /// Execution device (cpu, metal, cuda::0)
    #[arg(short, long, default_value = "cpu")]
    pub device: String,

    /// Backend plugin to use (auto-select if not specified)
    #[arg(long)]
    pub backend: Option<String>,

    /// Output format (pretty, json)
    #[arg(short, long, default_value = "pretty")]
    pub format: String,

    /// Save outputs to directory
    #[arg(long)]
    pub save: Option<PathBuf>,

    /// Save format (hdt, json, or format plugin extension)
    #[arg(long, default_value = "hdt")]
    pub save_format: String,

    /// Dry run (show what would be executed)
    #[arg(long)]
    pub dry_run: bool,

    /// Suppress all output
    #[arg(short, long)]
    pub quiet: bool,

    /// Timeout in seconds for plugin operations (default: 300)
    #[arg(long, value_name = "SECONDS")]
    pub timeout: Option<u64>,
}

pub fn execute(args: RunArgs) -> Result<(), Box<dyn std::error::Error>> {
    // Note: We don't check exists() here to avoid TOCTOU race conditions.
    // File operations will fail with descriptive errors if the file doesn't exist.

    // Validate timeout range
    if let Some(timeout) = args.timeout {
        if !(MIN_TIMEOUT_SECS..=MAX_TIMEOUT_SECS).contains(&timeout) {
            return Err(format!(
                "Timeout must be between {} and {} seconds (got: {})",
                MIN_TIMEOUT_SECS, MAX_TIMEOUT_SECS, timeout
            )
            .into());
        }
    }

    // Get model extension
    let extension = args
        .model
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_lowercase());

    // Load plugin registry
    let registry = load_registry()?;

    // Check for model format plugin (for non-builtin formats)
    let format_plugin = match extension.as_deref() {
        Some("hdss") | Some("hdt") | Some("json") => {
            // Builtin formats
            None
        },
        Some(ext) => {
            let plugin = registry.find_model_format_by_extension(ext);
            if plugin.is_none() {
                return Err(friendly_format_error(ext, &registry).into());
            }
            // Validate that the plugin has load_model capability
            if let Some(p) = &plugin {
                if !p.capabilities.load_model.unwrap_or(false) {
                    return Err(format!(
                        "Plugin '{}' doesn't support loading models (missing load_model capability)",
                        p.name
                    )
                    .into());
                }
            }
            plugin
        },
        None => {
            return Err("Model file has no extension. Cannot determine format.".into());
        },
    };

    // Parse device
    let device = parse_device(&args.device)?;

    // Find backend plugin
    let backend_plugin = find_backend_plugin(&args.backend, &device, &registry)?;

    // Combine --input and --inputs arguments
    let all_inputs: Vec<String> = args.input.iter().chain(args.inputs.iter()).cloned().collect();

    if args.dry_run {
        println!(
            "Model format: {} ({})",
            extension.as_deref().unwrap_or("unknown"),
            format_plugin
                .map(|p| format!("{} {}", p.name, p.version))
                .unwrap_or_else(|| "builtin".to_string())
        );
        for input_arg in &all_inputs {
            if let Some((name, path)) = input_arg.split_once('=') {
                let ext = std::path::Path::new(path)
                    .extension()
                    .and_then(|e| e.to_str())
                    .unwrap_or("unknown");
                println!("Input {}: {} (builtin)", name, ext);
            }
        }
        println!(
            "Backend: {} ({} {})",
            device, backend_plugin.name, backend_plugin.version
        );
        println!();
        println!("Would execute with above configuration.");
        return Ok(());
    }

    // Create plugin manager with optional timeout
    let mut manager = match args.timeout {
        Some(secs) => PluginManager::with_timeout(secs)?,
        None => PluginManager::new()?,
    };

    // Load model (using format plugin if needed)
    let model_name = args
        .model
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| args.model.display().to_string());
    let snapshot_path = if let Some(format_entry) = format_plugin {
        // Use format plugin to convert to snapshot
        output::loading(&model_name);
        let client = manager.get_plugin(&format_entry.name)?;
        let result = client.load_model(path_to_str(&args.model)?)?;
        PathBuf::from(result.snapshot_path)
    } else {
        // Builtin format - model is already a snapshot
        args.model.clone()
    };

    // Load the snapshot
    let snapshot = Snapshot::load(&snapshot_path)?;

    // Parse input tensors
    let inputs = parse_inputs(&all_inputs, &snapshot)?;

    // Save input tensors to temp files and create TensorInput refs
    // Use tempfile crate for secure, atomic temp file creation
    let mut input_refs = Vec::new();
    let mut temp_files = Vec::new(); // Keep temp files alive until inference completes
    for (name, tensor_data) in &inputs {
        let temp_file = NamedTempFile::with_prefix(format!("hodu_input_{}_", name))
            .map_err(|e| format!("Failed to create temp file for input '{}': {}", name, e))?;
        let temp_path = temp_file.path().to_path_buf();
        save_tensor_data(tensor_data, &temp_path)?;
        input_refs.push(TensorInput {
            name: name.clone(),
            path: temp_path.to_string_lossy().to_string(),
        });
        temp_files.push(temp_file); // Keep file handle to prevent deletion
    }

    // Run inference using backend plugin
    // First, spawn the backend plugin and get cancellation handle
    let _ = manager.get_plugin(&backend_plugin.name)?; // Ensure plugin is running
    let cancel_handle = manager.get_cancellation_handle(&backend_plugin.name);

    // Set up Ctrl+C handler for cancellation
    let cancelled = Arc::new(AtomicBool::new(false));
    let cancelled_clone = Arc::clone(&cancelled);
    if let Some(handle) = cancel_handle {
        if let Err(e) = ctrlc::set_handler(move || {
            cancelled_clone.store(true, Ordering::SeqCst);
            eprintln!("\nCancelling...");
            if let Err(cancel_err) = handle.cancel() {
                eprintln!("Warning: Failed to send cancellation: {}", cancel_err);
            }
        }) {
            output::warning(&format!(
                "Failed to set Ctrl+C handler: {}. Cancellation may not work.",
                e
            ));
        }
    }

    // Compute cache key from snapshot content (with size limit check)
    // Use a single open file handle to avoid TOCTOU race conditions
    // Limit is configurable via HODU_MAX_SNAPSHOT_SIZE env var (in bytes)
    const DEFAULT_MAX_SNAPSHOT_SIZE: u64 = 10 * 1024 * 1024 * 1024; // 10GB default
    let max_snapshot_size = std::env::var("HODU_MAX_SNAPSHOT_SIZE")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(DEFAULT_MAX_SNAPSHOT_SIZE);
    let snapshot_file = std::fs::File::open(&snapshot_path).map_err(|e| format!("Failed to open snapshot: {}", e))?;
    let snapshot_size = snapshot_file
        .metadata()
        .map_err(|e| format!("Failed to read snapshot metadata: {}", e))?
        .len();
    if snapshot_size > max_snapshot_size {
        return Err(format!(
            "Snapshot file too large: {} bytes (max: {} bytes, set HODU_MAX_SNAPSHOT_SIZE to override)",
            snapshot_size, max_snapshot_size
        )
        .into());
    }
    // Read from the already-opened handle to avoid TOCTOU
    use std::io::Read;
    let mut snapshot_content = Vec::new();
    std::io::BufReader::new(snapshot_file)
        .read_to_end(&mut snapshot_content)
        .map_err(|e| format!("Failed to read snapshot: {}", e))?;
    let mut hasher = Sha256::new();
    hasher.update(&snapshot_content);
    hasher.update(current_host_triple().as_bytes());
    let snapshot_hash = hex::encode(hasher.finalize());

    // Determine library extension and cache path
    let lib_ext = if cfg!(target_os = "macos") {
        "dylib"
    } else if cfg!(target_os = "windows") {
        "dll"
    } else {
        "so"
    };

    let cache_dir = dirs::home_dir()
        .ok_or("Could not determine home directory")?
        .join(".hodu")
        .join("cache")
        .join(&backend_plugin.name);
    std::fs::create_dir_all(&cache_dir)?;
    let library_path = cache_dir.join(format!("{}.{}", snapshot_hash, lib_ext));

    // Build if not cached
    let backend_client = manager.get_plugin(&backend_plugin.name)?;
    let start = std::time::Instant::now();
    if !library_path.exists() {
        output::compiling(&format!("{} ({})", model_name, device));
        backend_client.build(
            path_to_str(&snapshot_path)?,
            current_host_triple(),
            &device,
            "sharedlib",
            path_to_str(&library_path)?,
        )?;
    } else {
        output::cached(&model_name);
    }

    // Run with cached library
    output::running(&format!("{} ({})", model_name, device));
    let result = backend_client.run(
        path_to_str(&library_path)?,
        path_to_str(&snapshot_path)?,
        &device,
        input_refs,
    )?;
    let duration = start.elapsed().as_secs_f64();
    if !args.quiet {
        output::finished(&format!("inference in {}", output::format_duration(duration)));
    }

    // Check if was cancelled
    if cancelled.load(Ordering::SeqCst) {
        return Err("Operation cancelled by user".into());
    }

    // Load output tensors from paths
    let mut outputs: HashMap<String, TensorData> = HashMap::new();
    for output_ref in result.outputs {
        let tensor_data = load_tensor_data(&output_ref.path)?;
        outputs.insert(output_ref.name, tensor_data);
    }

    // Save outputs if requested
    if let Some(save_dir) = &args.save {
        save_outputs(&outputs, save_dir, &args.save_format)?;
    }

    // Output results
    if !args.quiet {
        output_results(&outputs, &args)?;
    }

    Ok(())
}

fn parse_inputs(
    input_args: &[String],
    snapshot: &Snapshot,
) -> Result<HashMap<String, TensorData>, Box<dyn std::error::Error>> {
    let mut inputs = HashMap::new();

    for arg in input_args {
        let parts: Vec<&str> = arg.splitn(2, '=').collect();
        if parts.len() != 2 {
            return Err(format!("Invalid input format: '{}'. Expected: name=path", arg).into());
        }

        let name = parts[0];

        // Validate input name
        if name.is_empty() {
            return Err("Input name cannot be empty".into());
        }
        if name.chars().any(|c| c.is_control() || c == '\0') {
            return Err(format!("Input name '{}' contains invalid characters", name).into());
        }

        let path = expand_path(parts[1])?;

        if !path.exists() {
            return Err(format!("Input file not found: {}", path.display()).into());
        }

        let input_spec = snapshot.inputs.iter().find(|i| i.name == name).ok_or_else(|| {
            format!(
                "Unknown input '{}'. Available: {:?}",
                name,
                snapshot.inputs.iter().map(|i| &i.name).collect::<Vec<_>>()
            )
        })?;

        let tensor_data = load_tensor_file(&path, input_spec.shape.dims(), core_dtype_to_plugin(input_spec.dtype))?;
        inputs.insert(name.to_string(), tensor_data);
    }

    for input_spec in &snapshot.inputs {
        if !inputs.contains_key(&input_spec.name) {
            return Err(format!(
                "Missing required input '{}' ({:?}, {:?})",
                input_spec.name,
                input_spec.shape.dims(),
                input_spec.dtype
            )
            .into());
        }
    }

    Ok(inputs)
}

fn output_results(outputs: &HashMap<String, TensorData>, args: &RunArgs) -> Result<(), Box<dyn std::error::Error>> {
    match args.format.as_str() {
        "json" => {
            let json_outputs: HashMap<String, serde_json::Value> = outputs
                .iter()
                .map(|(name, data)| {
                    (
                        name.clone(),
                        serde_json::json!({
                            "shape": data.shape,
                            "dtype": data.dtype.name(),
                        }),
                    )
                })
                .collect();
            println!("{}", serde_json::to_string_pretty(&json_outputs)?);
        },
        _ => {
            let mut names: Vec<_> = outputs.keys().collect();
            names.sort();
            for name in names {
                let data = &outputs[name];
                let dtype = plugin_dtype_to_core(data.dtype)?;
                let shape = Shape::new(&data.shape);
                let tensor = Tensor::from_bytes(&data.data, shape, dtype, CoreDevice::CPU)?;
                // Colored ">" prefix, white name
                if output::supports_color() {
                    println!("{}>{}  {}", output::colors::BOLD_YELLOW, output::colors::RESET, name);
                } else {
                    println!(">  {}", name);
                }
                println!("{}", tensor);
            }
        },
    }

    Ok(())
}

fn expand_path(path: &str) -> Result<PathBuf, Box<dyn std::error::Error>> {
    use std::path::Component;

    // Expand ~ to home directory
    let expanded = if let Some(stripped) = path.strip_prefix("~/") {
        if let Some(home) = dirs::home_dir() {
            home.join(stripped)
        } else {
            PathBuf::from(path)
        }
    } else {
        PathBuf::from(path)
    };

    // Canonicalize to resolve .. and symlinks, preventing path traversal
    if expanded.exists() {
        return Ok(expanded.canonicalize()?);
    }

    // For non-existent paths, strictly validate components
    for component in expanded.components() {
        match component {
            Component::ParentDir => {
                return Err("Path traversal (.. components) not allowed".into());
            },
            Component::Normal(s) => {
                let s_str = s.to_string_lossy();
                if s_str.contains('\0') {
                    return Err("Path contains null byte".into());
                }
            },
            _ => {},
        }
    }

    // For non-existent files, canonicalize the existing parent directory
    // This resolves symlinks in the parent path, preventing symlink-based traversal
    if let Some(parent) = expanded.parent() {
        if parent.exists() {
            let canonical_parent = parent.canonicalize()?;
            if let Some(file_name) = expanded.file_name() {
                return Ok(canonical_parent.join(file_name));
            }
        }
    }

    Ok(expanded)
}

fn parse_device(device_str: &str) -> Result<Device, Box<dyn std::error::Error>> {
    // Validate device string doesn't contain dangerous characters
    if device_str.is_empty() {
        return Err("Device string cannot be empty".into());
    }
    if device_str.len() > 64 {
        return Err(format!("Device string too long: {} chars (max 64)", device_str.len()).into());
    }
    // Only allow alphanumeric, underscore, colon, and hyphen
    if !device_str
        .chars()
        .all(|c| c.is_alphanumeric() || c == '_' || c == ':' || c == '-')
    {
        return Err(format!(
            "Device string contains invalid characters: '{}'. Only alphanumeric, '_', ':', '-' allowed",
            device_str
        )
        .into());
    }

    // Normalize to lowercase with :: separator for device index
    let device = device_str.to_lowercase();

    // Handle common patterns
    match device.as_str() {
        "cpu" | "metal" | "vulkan" | "webgpu" => Ok(device),
        s if s.starts_with("cuda") => {
            if s == "cuda" {
                Ok("cuda::0".to_string())
            } else if let Some(idx_str) = s.strip_prefix("cuda::") {
                // Validate index
                idx_str
                    .parse::<usize>()
                    .map_err(|_| format!("Invalid CUDA device index: {}", idx_str))?;
                Ok(device)
            } else if let Some(idx_str) = s.strip_prefix("cuda:") {
                // Convert cuda:N to cuda::N
                idx_str
                    .parse::<usize>()
                    .map_err(|_| format!("Invalid CUDA device index: {}", idx_str))?;
                Ok(format!("cuda::{}", idx_str))
            } else {
                Err(format!("Invalid device format: {}", device_str).into())
            }
        },
        s if s.starts_with("rocm") => {
            if s == "rocm" {
                Ok("rocm::0".to_string())
            } else if let Some(idx_str) = s.strip_prefix("rocm::") {
                idx_str
                    .parse::<usize>()
                    .map_err(|_| format!("Invalid ROCm device index: {}", idx_str))?;
                Ok(device)
            } else {
                Err(format!("Invalid device format: {}", device_str).into())
            }
        },
        // Allow any other device string for extensibility, but warn if unknown
        _ => {
            let is_known = KNOWN_DEVICE_PREFIXES.iter().any(|prefix| device.starts_with(prefix));
            if !is_known {
                eprintln!(
                    "Warning: Unknown device '{}'. Known devices: {}",
                    device,
                    KNOWN_DEVICE_PREFIXES.join(", ")
                );
            }
            Ok(device)
        },
    }
}

fn find_backend_plugin<'a>(
    backend_name: &Option<String>,
    device: &Device,
    registry: &'a PluginRegistry,
) -> Result<&'a crate::plugins::PluginEntry, Box<dyn std::error::Error>> {
    if let Some(name) = backend_name {
        if let Some(plugin) = registry.find(name) {
            return Ok(plugin);
        }
        let prefixed = backend_plugin_name(name);
        if let Some(plugin) = registry.find(&prefixed) {
            return Ok(plugin);
        }
        return Err(format!("Backend '{}' not found.", name).into());
    }

    if let Some(plugin) = registry.find_backend_by_device(device) {
        return Ok(plugin);
    }

    Err(friendly_backend_error(device, registry).into())
}

fn friendly_format_error(extension: &str, registry: &PluginRegistry) -> String {
    let mut msg = format!("No model format plugin found for '.{}' format.\n\n", extension);
    let formats: Vec<_> = registry.model_formats().collect();
    if !formats.is_empty() {
        msg.push_str("Installed model format plugins:\n");
        for p in formats {
            msg.push_str(&format!(
                "  {} - {}\n",
                p.name,
                p.capabilities.model_extensions.join(", ")
            ));
        }
    }
    msg.push_str("\nBuiltin formats: .hdss");
    msg
}

fn friendly_backend_error(device: &Device, registry: &PluginRegistry) -> String {
    let mut msg = format!("No backend plugin found for device '{}'.\n\n", device);

    let backends: Vec<_> = registry.backends().collect();
    if !backends.is_empty() {
        msg.push_str("Installed backend plugins:\n");
        for p in backends {
            msg.push_str(&format!(
                "  {} - devices: {}\n",
                p.name,
                p.capabilities.devices.join(", ")
            ));
        }
    }

    msg
}
