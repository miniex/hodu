//! Build command - AOT compile models using backend plugins
//!
//! This command uses JSON-RPC based plugins to compile models.

use crate::output;
use crate::plugins::{load_registry, PluginManager, PluginRegistry};
use crate::utils::path_to_str;
use clap::Args;
use hodu_core::snapshot::Snapshot;
use hodu_plugin::BuildTarget;
use std::path::{Path, PathBuf};

#[derive(Args)]
pub struct BuildArgs {
    /// Model file (.onnx, .hdss, etc.)
    pub model: Option<PathBuf>,

    /// Output file path
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Target triple (default: current system)
    #[arg(short, long)]
    pub target: Option<String>,

    /// Target device (cpu, metal, cuda::0)
    #[arg(short, long, default_value = "cpu")]
    pub device: String,

    /// Backend plugin name (default: auto-detect by device)
    #[arg(short, long)]
    pub backend: Option<String>,

    /// Output format (sharedlib, staticlib, object, metallib, ptx)
    #[arg(short, long)]
    pub format: Option<String>,

    /// Optimization level (0-3)
    #[arg(short = 'O', long, default_value = "2")]
    pub opt_level: u8,

    /// Generate standalone executable
    #[arg(long)]
    pub standalone: bool,

    /// Verbose output
    #[arg(short, long)]
    pub verbose: bool,

    /// List supported build targets for the backend
    #[arg(long)]
    pub list_targets: bool,

    /// Timeout in seconds for plugin operations (default: 300)
    #[arg(long, value_name = "SECONDS")]
    pub timeout: Option<u64>,
}

pub fn execute(args: BuildArgs) -> Result<(), Box<dyn std::error::Error>> {
    let registry = load_registry()?;

    // Normalize device (lowercase)
    let device = args.device.to_lowercase();

    // Find backend: explicit --backend or auto-detect by device
    let backend_name = match &args.backend {
        Some(name) => find_backend_by_name(name, &registry)?.name.clone(),
        None => find_builder_backend(&device, &registry)?.name.clone(),
    };

    // Handle --list-targets
    if args.list_targets {
        return list_targets(&backend_name);
    }

    // For normal build, model and output are required
    let model = args.model.ok_or("Model file is required for building")?;
    let output = args.output.ok_or("Output path is required (use -o/--output)")?;

    if !model.exists() {
        return Err(format!("Model file not found: {}", model.display()).into());
    }

    // Validate output path
    if output.as_os_str().is_empty() {
        return Err("Output path cannot be empty".into());
    }
    if output.is_dir() {
        return Err(format!("Output path is a directory: {}", output.display()).into());
    }
    if let Some(parent) = output.parent() {
        if !parent.as_os_str().is_empty() && !parent.exists() {
            return Err(format!("Output directory does not exist: {}", parent.display()).into());
        }
        // Check write permission on parent directory
        if !parent.as_os_str().is_empty() && parent.exists() {
            let test_path = parent.join(".hodu_write_test");
            match std::fs::File::create(&test_path) {
                Ok(_) => {
                    let _ = std::fs::remove_file(&test_path);
                },
                Err(_) => {
                    return Err(format!("Cannot write to output directory: {}", parent.display()).into());
                },
            }
        }
    }

    let extension = model.extension().and_then(|e| e.to_str()).map(|e| e.to_lowercase());

    // Find model format plugin if needed
    let format_plugin = match extension.as_deref() {
        Some("hdss") => None,
        Some(ext) => {
            let plugin = registry.find_model_format_by_extension(ext);
            if plugin.is_none() {
                return Err(format!("No model format plugin found for .{}", ext).into());
            }
            plugin
        },
        None => return Err("Model file has no extension".into()),
    };

    // Create plugin manager with optional timeout
    let mut manager = match args.timeout {
        Some(secs) => PluginManager::with_timeout(secs)?,
        None => PluginManager::new()?,
    };

    // Load model (using format plugin if needed)
    let snapshot_path = if let Some(format_entry) = format_plugin {
        let display_name = model
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| model.display().to_string());
        output::loading(&display_name);
        let client = manager.get_plugin(&format_entry.name)?;
        let result = client.load_model(path_to_str(&model)?)?;
        PathBuf::from(result.snapshot_path)
    } else {
        model.clone()
    };

    // Validate snapshot is loadable before building
    Snapshot::load(&snapshot_path)?;

    // Determine build format from arg or output extension
    let format = determine_format(&args.format, &output);

    // Determine build target
    let build_target = match &args.target {
        Some(triple) => BuildTarget::new(triple.clone(), device.clone()),
        None => BuildTarget::host(device.clone()),
    };

    // Build message
    let model_name = model
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| model.display().to_string());
    output::compiling(&format!("{} ({}, {})", model_name, build_target.triple, device));

    // Call backend.build via JSON-RPC
    let start = std::time::Instant::now();
    let client = manager.get_plugin(&backend_name)?;
    client.build(
        path_to_str(&snapshot_path)?,
        &build_target.triple,
        &build_target.device,
        &format,
        path_to_str(&output)?,
    )?;

    let duration = start.elapsed().as_secs_f64();
    output::finished(&format!(
        "{} target(s) in {}",
        format,
        output::format_duration(duration)
    ));

    Ok(())
}

fn list_targets(backend_name: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut manager = PluginManager::new()?;
    let client = manager.get_plugin(backend_name)?;

    let result = client.list_targets()?;

    println!("Supported build targets for {} backend:\n", backend_name);
    println!("{}", result.formatted);

    Ok(())
}

fn find_backend_by_name<'a>(
    name: &str,
    registry: &'a PluginRegistry,
) -> Result<&'a crate::plugins::PluginEntry, Box<dyn std::error::Error>> {
    // Try exact match first
    if let Some(plugin) = registry.find(name) {
        if plugin.capabilities.builder == Some(true) {
            return Ok(plugin);
        }
    }

    // Try with "aot-" prefix
    let prefixed = format!("aot-{}", name);
    if let Some(plugin) = registry.find(&prefixed) {
        if plugin.capabilities.builder == Some(true) {
            return Ok(plugin);
        }
    }

    Err(format!(
        "Backend '{}' not found or does not support building.\n\nInstalled backends:\n{}",
        name,
        registry
            .backends()
            .map(|p| format!("  {} - builder: {}", p.name, p.capabilities.builder.unwrap_or(false)))
            .collect::<Vec<_>>()
            .join("\n")
    )
    .into())
}

fn find_builder_backend<'a>(
    device: &str,
    registry: &'a PluginRegistry,
) -> Result<&'a crate::plugins::PluginEntry, Box<dyn std::error::Error>> {
    for plugin in registry.backends() {
        if plugin.capabilities.builder == Some(true)
            && plugin.capabilities.devices.iter().any(|d| d.to_lowercase() == device)
        {
            return Ok(plugin);
        }
    }

    Err(format!(
        "No builder backend found for device '{}'\n\nInstalled backends:\n{}",
        device,
        registry
            .backends()
            .map(|p| format!(
                "  {} - devices: {}, builder: {}",
                p.name,
                p.capabilities.devices.join(", "),
                p.capabilities.builder.unwrap_or(false)
            ))
            .collect::<Vec<_>>()
            .join("\n")
    )
    .into())
}

fn determine_format(format_arg: &Option<String>, output: &Path) -> String {
    if let Some(fmt) = format_arg {
        return fmt.to_lowercase();
    }

    let ext = output.extension().and_then(|e| e.to_str()).unwrap_or("");

    match ext {
        "" | "exe" | "bin" => "executable".to_string(),
        "so" | "dylib" | "dll" => "sharedlib".to_string(),
        "a" | "lib" => "staticlib".to_string(),
        "o" | "obj" => "object".to_string(),
        "metallib" => "metallib".to_string(),
        "ptx" => "ptx".to_string(),
        "cubin" => "cubin".to_string(),
        "ll" => "llvmir".to_string(),
        "bc" => "llvmbitcode".to_string(),
        "wgsl" => "wgsl".to_string(),
        "spv" => "spirv".to_string(),
        _ => "sharedlib".to_string(),
    }
}
