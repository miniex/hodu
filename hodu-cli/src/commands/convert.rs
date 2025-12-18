//! Convert command - convert models and tensors between formats

use crate::output;
use crate::plugins::{load_registry, PluginManager, PluginRegistry};
use crate::tensor::{load_tensor_data, save_tensor_data};
use crate::utils::{core_dtype_to_plugin, path_to_str, plugin_dtype_to_core};
use clap::Args;
use std::path::PathBuf;

/// RAII guard for temporary files - automatically cleans up on drop
struct TempFileGuard {
    path: PathBuf,
}

impl TempFileGuard {
    fn new(path: PathBuf) -> Self {
        Self { path }
    }

    fn path(&self) -> &PathBuf {
        &self.path
    }
}

impl Drop for TempFileGuard {
    fn drop(&mut self) {
        if self.path.exists() {
            if let Err(e) = std::fs::remove_file(&self.path) {
                output::warning(&format!("Failed to remove temp file: {}", e));
            }
        }
    }
}

#[derive(Args)]
pub struct ConvertArgs {
    /// Input file
    pub input: PathBuf,

    /// Output file
    #[arg(short, long)]
    pub output: PathBuf,

    /// Verbose output
    #[arg(short, long)]
    pub verbose: bool,
}

pub fn execute(args: ConvertArgs) -> Result<(), Box<dyn std::error::Error>> {
    if !args.input.exists() {
        return Err(format!("Input file not found: {}", args.input.display()).into());
    }

    let input_ext = args
        .input
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_lowercase())
        .ok_or("Input file has no extension")?;

    let output_ext = args
        .output
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_lowercase())
        .ok_or("Output file has no extension")?;

    let registry = load_registry()?;

    // Determine conversion type (model or tensor)
    let is_model = is_model_format(&input_ext) || is_model_format(&output_ext);

    if args.verbose {
        println!("Input: {} (.{})", args.input.display(), input_ext);
        println!("Output: {} (.{})", args.output.display(), output_ext);
        println!("Type: {}", if is_model { "model" } else { "tensor" });
    }

    let mut manager = PluginManager::new()?;

    output::converting(&format!(
        "{} -> .{}",
        args.input.file_name().unwrap_or_default().to_string_lossy(),
        output_ext
    ));

    if is_model {
        convert_model(&args, &input_ext, &output_ext, &registry, &mut manager)
    } else {
        convert_tensor(&args, &input_ext, &output_ext, &registry, &mut manager)
    }
}

fn is_model_format(ext: &str) -> bool {
    matches!(ext, "hdss" | "onnx" | "pb" | "tflite" | "mlmodel")
}

fn convert_model(
    args: &ConvertArgs,
    input_ext: &str,
    output_ext: &str,
    registry: &PluginRegistry,
    manager: &mut PluginManager,
) -> Result<(), Box<dyn std::error::Error>> {
    // Step 1: Load input model to snapshot
    let snapshot_path = if input_ext == "hdss" {
        // Already a snapshot
        args.input.clone()
    } else {
        // Use model format plugin to load
        let plugin = registry
            .find_model_format_by_extension(input_ext)
            .ok_or_else(|| format!("No model format plugin for .{}", input_ext))?;

        if !plugin.capabilities.load_model.unwrap_or(false) {
            let caps = format_capabilities(&plugin.capabilities);
            return Err(format!(
                "Plugin {} doesn't support loading models.\nAvailable capabilities: {}",
                plugin.name, caps
            )
            .into());
        }

        let client = manager.get_plugin(&plugin.name)?;
        let result = client.load_model(path_to_str(&args.input)?)?;
        PathBuf::from(result.snapshot_path)
    };

    // Step 2: Save to output format
    if output_ext == "hdss" {
        // Just copy the snapshot
        std::fs::copy(&snapshot_path, &args.output)?;
    } else {
        // Use model format plugin to save
        let plugin = registry
            .find_model_format_by_extension(output_ext)
            .ok_or_else(|| format!("No model format plugin for .{}", output_ext))?;

        if !plugin.capabilities.save_model.unwrap_or(false) {
            let caps = format_capabilities(&plugin.capabilities);
            return Err(format!(
                "Plugin {} doesn't support saving models.\nAvailable capabilities: {}",
                plugin.name, caps
            )
            .into());
        }

        let client = manager.get_plugin(&plugin.name)?;
        client.save_model(path_to_str(&snapshot_path)?, path_to_str(&args.output)?)?;
    }

    output::finished(&format!(
        "{} -> {}",
        args.input.file_name().unwrap_or_default().to_string_lossy(),
        args.output.file_name().unwrap_or_default().to_string_lossy()
    ));
    Ok(())
}

fn convert_tensor(
    args: &ConvertArgs,
    input_ext: &str,
    output_ext: &str,
    registry: &PluginRegistry,
    manager: &mut PluginManager,
) -> Result<(), Box<dyn std::error::Error>> {
    use hodu_core::format::json;
    use hodu_core::tensor::Tensor;
    use hodu_core::types::{Device as CoreDevice, Shape};
    use hodu_plugin::TensorData;

    // Step 1: Load input tensor
    let tensor_data = match input_ext {
        "hdt" => load_tensor_data(&args.input)?,
        "json" => {
            let tensor = json::load(&args.input).map_err(|e| e.to_string())?;
            let shape = tensor.shape().dims().to_vec();
            let dtype = core_dtype_to_plugin(tensor.dtype());
            let data = tensor.to_bytes().map_err(|e| e.to_string())?;
            TensorData::new(data, shape, dtype)
        },
        _ => {
            // Use tensor format plugin
            let plugin = registry
                .find_tensor_format_by_extension(input_ext)
                .ok_or_else(|| format!("No tensor format plugin for .{}", input_ext))?;

            if !plugin.capabilities.load_tensor.unwrap_or(false) {
                return Err(format!("Plugin {} doesn't support loading tensors", plugin.name).into());
            }

            let client = manager.get_plugin(&plugin.name)?;
            let result = client.load_tensor(path_to_str(&args.input)?)?;
            load_tensor_data(&result.tensor_path)?
        },
    };

    // Step 2: Save to output format
    match output_ext {
        "hdt" => {
            save_tensor_data(&tensor_data, &args.output)?;
        },
        "json" => {
            let shape = Shape::new(&tensor_data.shape);
            let dtype = plugin_dtype_to_core(tensor_data.dtype)?;
            let tensor =
                Tensor::from_bytes(&tensor_data.data, shape, dtype, CoreDevice::CPU).map_err(|e| e.to_string())?;
            json::save(&tensor, &args.output).map_err(|e| e.to_string())?;
        },
        _ => {
            // Use tensor format plugin
            let plugin = registry
                .find_tensor_format_by_extension(output_ext)
                .ok_or_else(|| format!("No tensor format plugin for .{}", output_ext))?;

            if !plugin.capabilities.save_tensor.unwrap_or(false) {
                return Err(format!("Plugin {} doesn't support saving tensors", plugin.name).into());
            }

            // Save to temp hdt first (use PID + timestamp for uniqueness)
            // TempFileGuard ensures cleanup even on error
            let temp_guard = TempFileGuard::new(std::env::temp_dir().join(format!(
                "hodu_convert_{}_{}.hdt",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_nanos())
                    .unwrap_or(0)
            )));
            save_tensor_data(&tensor_data, temp_guard.path())?;

            let client = manager.get_plugin(&plugin.name)?;
            client.save_tensor(path_to_str(temp_guard.path())?, path_to_str(&args.output)?)?;
            // temp_guard automatically cleans up on drop
        },
    }

    output::finished(&format!(
        "{} -> {}",
        args.input.file_name().unwrap_or_default().to_string_lossy(),
        args.output.file_name().unwrap_or_default().to_string_lossy()
    ));
    Ok(())
}

/// Format plugin capabilities as a comma-separated string
fn format_capabilities(caps: &crate::plugins::PluginCapabilities) -> String {
    let mut list = Vec::new();
    if caps.load_model.unwrap_or(false) {
        list.push("load_model");
    }
    if caps.save_model.unwrap_or(false) {
        list.push("save_model");
    }
    if caps.load_tensor.unwrap_or(false) {
        list.push("load_tensor");
    }
    if caps.save_tensor.unwrap_or(false) {
        list.push("save_tensor");
    }
    if caps.runner.unwrap_or(false) {
        list.push("runner");
    }
    if caps.builder.unwrap_or(false) {
        list.push("builder");
    }
    if list.is_empty() {
        "none".to_string()
    } else {
        list.join(", ")
    }
}
