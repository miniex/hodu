//! Inspect command - examine model and tensor files
//!
//! This command inspects model and tensor files, optionally using format plugins.

use crate::output::{self, colors};
use crate::plugins::{load_registry, PluginManager};
use crate::tensor::load_tensor_data;
use crate::utils::path_to_str;
use clap::Args;
use hodu_core::format::hdt;
use hodu_core::ops::Op;
use hodu_core::snapshot::Snapshot;
use hodu_core::tensor::Tensor;
use std::path::{Path, PathBuf};

#[derive(Args)]
pub struct InspectArgs {
    /// File to inspect (.hdss, .hdt, .json, .onnx, etc.)
    pub file: PathBuf,

    /// Verbose output
    #[arg(short, long)]
    pub verbose: bool,

    /// Output format (pretty, json)
    #[arg(short, long, default_value = "pretty")]
    pub format: String,
}

pub fn execute(args: InspectArgs) -> Result<(), Box<dyn std::error::Error>> {
    if !args.file.exists() {
        return Err(format!("File not found: {}", args.file.display()).into());
    }

    let ext = args
        .file
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    let display_name = args
        .file
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| args.file.display().to_string());
    output::inspecting(&display_name);

    match ext.as_str() {
        "hdss" => inspect_hdss(&args),
        "hdt" => inspect_hdt(&args),
        "json" => inspect_json_tensor(&args),
        _ => inspect_with_plugin(&args, &ext),
    }
}

fn inspect_hdss(args: &InspectArgs) -> Result<(), Box<dyn std::error::Error>> {
    let use_color = output::supports_color();
    let snapshot = Snapshot::load(&args.file).map_err(|e| format!("Failed to load snapshot: {}", e))?;

    if args.format == "json" {
        println!("{}", serde_json::to_string_pretty(&snapshot)?);
        return Ok(());
    }

    // Header
    let model_name = snapshot.name.as_deref().unwrap_or("unnamed");
    if use_color {
        println!(
            "{}{}{} {}",
            colors::BOLD,
            model_name,
            colors::RESET,
            format_file_size(&args.file)
        );
    } else {
        println!("{} {}", model_name, format_file_size(&args.file));
    }
    println!();

    // Inputs
    print_section_header("Inputs", snapshot.inputs.len(), use_color);
    for input in &snapshot.inputs {
        print_tensor_row(
            &input.name,
            input.shape.dims(),
            &format!("{:?}", input.dtype),
            use_color,
        );
    }
    println!();

    // Constants (if any)
    if !snapshot.constants.is_empty() {
        print_section_header("Constants", snapshot.constants.len(), use_color);
        for constant in &snapshot.constants {
            let name = constant.name.as_deref().unwrap_or("(unnamed)");
            let size_str = output::format_size(constant.data.len());
            if use_color {
                println!(
                    "  {}•{} {:<16} {:?} {}{:?}{} {}{}{}",
                    colors::CYAN,
                    colors::RESET,
                    name,
                    constant.shape.dims(),
                    colors::CYAN,
                    constant.dtype,
                    colors::RESET,
                    colors::YELLOW,
                    size_str,
                    colors::RESET
                );
            } else {
                println!(
                    "  • {:<16} {:?} {:?} {}",
                    name,
                    constant.shape.dims(),
                    constant.dtype,
                    size_str
                );
            }
        }
        println!();
    }

    // Outputs
    print_section_header("Outputs", snapshot.targets.len(), use_color);
    for target in &snapshot.targets {
        if use_color {
            println!("  {}→{} {}", colors::GREEN, colors::RESET, target.name);
        } else {
            println!("  → {}", target.name);
        }
    }
    println!();

    // Nodes
    print_section_header("Graph", snapshot.nodes.len(), use_color);
    if args.verbose {
        for (i, node) in snapshot.nodes.iter().enumerate() {
            let op_str = format_op(&node.op);
            if use_color {
                println!(
                    "  {}[{:3}]{} {} {}→ {:?}{}",
                    colors::CYAN,
                    i,
                    colors::RESET,
                    op_str,
                    colors::YELLOW,
                    node.output_id,
                    colors::RESET
                );
            } else {
                println!("  [{:3}] {} → {:?}", i, op_str, node.output_id);
            }
        }
    } else {
        // Group by op category
        let mut categories = std::collections::HashMap::new();
        for node in &snapshot.nodes {
            let cat = get_op_category(&node.op);
            *categories.entry(cat).or_insert(0) += 1;
        }

        let mut sorted: Vec<_> = categories.into_iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1));

        for (cat, count) in sorted {
            if use_color {
                println!(
                    "  {}•{} {:<20} {}×{}{}",
                    colors::CYAN,
                    colors::RESET,
                    cat,
                    colors::YELLOW,
                    count,
                    colors::RESET
                );
            } else {
                println!("  • {:<20} ×{}", cat, count);
            }
        }
    }

    Ok(())
}

fn inspect_hdt(args: &InspectArgs) -> Result<(), Box<dyn std::error::Error>> {
    let tensor = hdt::load(&args.file).map_err(|e| format!("Failed to load HDT: {}", e))?;
    print_tensor_info(&tensor, &args.file, args.format == "json")
}

fn inspect_json_tensor(args: &InspectArgs) -> Result<(), Box<dyn std::error::Error>> {
    use hodu_core::format::json;
    let tensor = json::load(&args.file).map_err(|e| format!("Failed to load JSON tensor: {}", e))?;
    print_tensor_info(&tensor, &args.file, args.format == "json")
}

fn print_tensor_info(tensor: &Tensor, path: &Path, as_json: bool) -> Result<(), Box<dyn std::error::Error>> {
    let use_color = output::supports_color();
    let tensor_shape = tensor.shape();
    let shape = tensor_shape.dims();
    let dtype = tensor.dtype();
    let numel: usize = shape.iter().product();
    let size_bytes = tensor.to_bytes().map(|b| b.len()).unwrap_or(0);

    if as_json {
        println!(
            "{}",
            serde_json::json!({
                "file": path.display().to_string(),
                "shape": shape,
                "dtype": format!("{:?}", dtype),
                "numel": numel,
                "size_bytes": size_bytes
            })
        );
    } else {
        let filename = path.file_name().unwrap_or_default().to_string_lossy();
        if use_color {
            println!("{}{}{}", colors::BOLD, filename, colors::RESET);
            println!();
            println!("  {}Shape{} {:?}", colors::CYAN, colors::RESET, shape);
            println!("  {}DType{} {:?}", colors::CYAN, colors::RESET, dtype);
            println!("  {}Elements{} {}", colors::CYAN, colors::RESET, format_number(numel));
            println!(
                "  {}Size{} {}",
                colors::CYAN,
                colors::RESET,
                output::format_size(size_bytes)
            );
        } else {
            println!("{}", filename);
            println!();
            println!("  Shape    {:?}", shape);
            println!("  DType    {:?}", dtype);
            println!("  Elements {}", format_number(numel));
            println!("  Size     {}", output::format_size(size_bytes));
        }
    }

    Ok(())
}

fn inspect_with_plugin(args: &InspectArgs, ext: &str) -> Result<(), Box<dyn std::error::Error>> {
    let use_color = output::supports_color();

    // Create plugin manager and get format plugin by extension
    let mut manager = PluginManager::new()?;
    let client = manager.get_format_for_extension(ext).map_err(|_| {
        format!(
            "No plugin found for '.{}' format.\n\nBuiltin formats: .hdss, .hdt, .json\n\nInstall a format plugin:\n  hodu plugin install --git <url>",
            ext
        )
    })?;

    // Get plugin entry from registry for capability check
    let registry = load_registry()?;

    // Try model format first, then tensor format
    let plugin_entry = registry
        .find_model_format_by_extension(ext)
        .or_else(|| registry.find_tensor_format_by_extension(ext))
        .ok_or("Plugin not found")?;

    // Try to load as model first
    if plugin_entry.capabilities.load_model.unwrap_or(false) {
        let result = client.load_model(path_to_str(&args.file)?)?;

        // Load the snapshot from the temp path
        let snapshot = Snapshot::load(&result.snapshot_path).map_err(|e| format!("Failed to load snapshot: {}", e))?;

        if args.format == "json" {
            println!("{}", serde_json::to_string_pretty(&snapshot)?);
        } else {
            // Header
            let model_name = snapshot.name.as_deref().unwrap_or("unnamed");
            if use_color {
                println!(
                    "{}{}{} {}via {}{}",
                    colors::BOLD,
                    model_name,
                    colors::RESET,
                    colors::CYAN,
                    plugin_entry.name,
                    colors::RESET
                );
            } else {
                println!("{} via {}", model_name, plugin_entry.name);
            }
            println!();

            // Inputs
            print_section_header("Inputs", snapshot.inputs.len(), use_color);
            for input in &snapshot.inputs {
                print_tensor_row(
                    &input.name,
                    input.shape.dims(),
                    &format!("{:?}", input.dtype),
                    use_color,
                );
            }
            println!();

            // Outputs
            print_section_header("Outputs", snapshot.targets.len(), use_color);
            for target in &snapshot.targets {
                if use_color {
                    println!("  {}→{} {}", colors::GREEN, colors::RESET, target.name);
                } else {
                    println!("  → {}", target.name);
                }
            }
            println!();

            // Nodes summary
            print_section_header("Graph", snapshot.nodes.len(), use_color);
            if args.verbose {
                for (i, node) in snapshot.nodes.iter().enumerate() {
                    let op_str = format_op(&node.op);
                    if use_color {
                        println!("  {}[{:3}]{} {}", colors::CYAN, i, colors::RESET, op_str);
                    } else {
                        println!("  [{:3}] {}", i, op_str);
                    }
                }
            }
        }
        return Ok(());
    }

    // Try to load as tensor
    if plugin_entry.capabilities.load_tensor.unwrap_or(false) {
        let result = client.load_tensor(path_to_str(&args.file)?)?;

        // Load tensor data from path
        let tensor_data = load_tensor_data(&result.tensor_path).map_err(|e| format!("Failed to load tensor: {}", e))?;

        if args.format == "json" {
            println!(
                "{}",
                serde_json::json!({
                    "file": args.file.display().to_string(),
                    "shape": tensor_data.shape,
                    "dtype": tensor_data.dtype.name(),
                    "size_bytes": tensor_data.data.len()
                })
            );
        } else {
            let filename = args.file.file_name().unwrap_or_default().to_string_lossy();
            if use_color {
                println!("{}{}{}", colors::BOLD, filename, colors::RESET);
                println!();
                println!("  {}Shape{} {:?}", colors::CYAN, colors::RESET, tensor_data.shape);
                println!("  {}DType{} {}", colors::CYAN, colors::RESET, tensor_data.dtype.name());
                println!(
                    "  {}Size{} {}",
                    colors::CYAN,
                    colors::RESET,
                    output::format_size(tensor_data.data.len())
                );
            } else {
                println!("{}", filename);
                println!();
                println!("  Shape {:?}", tensor_data.shape);
                println!("  DType {}", tensor_data.dtype.name());
                println!("  Size  {}", output::format_size(tensor_data.data.len()));
            }
        }
        return Ok(());
    }

    Err(format!(
        "Plugin '{}' doesn't support loading models or tensors",
        plugin_entry.name
    )
    .into())
}

// Helper functions

fn print_section_header(title: &str, count: usize, use_color: bool) {
    if use_color {
        println!(
            "{}{}{} {}({}){}",
            colors::BOLD,
            colors::CYAN,
            title,
            colors::RESET,
            count,
            colors::RESET
        );
    } else {
        println!("{} ({})", title, count);
    }
}

fn print_tensor_row(name: &str, shape: &[usize], dtype: &str, use_color: bool) {
    if use_color {
        println!(
            "  {}•{} {:<16} {:?} {}{}{}",
            colors::CYAN,
            colors::RESET,
            name,
            shape,
            colors::CYAN,
            dtype,
            colors::RESET
        );
    } else {
        println!("  • {:<16} {:?} {}", name, shape, dtype);
    }
}

fn format_op(op: &Op) -> String {
    // Format as "Category[variant]" e.g., "Matrix[dot]", "Binary[mul]"
    let debug_str = format!("{:?}", op);

    // Parse "Category(Variant)" or "Category(Variant { ... })"
    if let Some(paren_pos) = debug_str.find('(') {
        let category = &debug_str[..paren_pos];
        let rest = &debug_str[paren_pos + 1..];

        // Extract variant name (before any whitespace or brace)
        let variant = rest
            .trim_end_matches(')')
            .split(|c: char| c.is_whitespace() || c == '{')
            .next()
            .unwrap_or("")
            .to_lowercase();

        if variant.is_empty() {
            category.to_string()
        } else {
            format!("{}[{}]", category, variant)
        }
    } else {
        debug_str
    }
}

fn get_op_category(op: &Op) -> &'static str {
    match op {
        Op::Binary(_) => "Binary",
        Op::BinaryLogical(_) => "BinaryLogical",
        Op::BitwiseBinary(_) => "BitwiseBinary",
        Op::BitwiseUnary(_) => "BitwiseUnary",
        Op::BitwiseUnaryScalar(_) => "BitwiseUnaryScalar",
        Op::Cmp(_) => "Compare",
        Op::CmpScalar(_) => "CmpScalar",
        Op::Unary(_) => "Unary",
        Op::UnaryLogical(_) => "UnaryLogical",
        Op::UnaryScalar(_) => "UnaryScalar",
        Op::Matrix(_) => "Matrix",
        Op::Linalg(_) => "Linalg",
        Op::Reduce(_) => "Reduce",
        Op::Concat(_) => "Concat",
        Op::Split(_) => "Split",
        Op::Indexing(_) => "Indexing",
        Op::Conv(_) => "Conv",
        Op::Windowing(_) => "Windowing",
        Op::Resize(_) => "Resize",
        Op::Padding(_) => "Padding",
        Op::Scan(_) => "Scan",
        Op::Sort(_) => "Sort",
        Op::Einsum(_) => "Einsum",
        Op::Shape(_) => "Shape",
        Op::ShapeScalars(_) => "ShapeScalars",
        Op::ShapeMemory(_) => "ShapeMemory",
        Op::Cast(_) => "Cast",
        Op::Memory(_) => "Memory",
        Op::Dummy => "Dummy",
    }
}

fn format_number(n: usize) -> String {
    if n >= 1_000_000_000 {
        format!("{:.2}B", n as f64 / 1_000_000_000.0)
    } else if n >= 1_000_000 {
        format!("{:.2}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.2}K", n as f64 / 1_000.0)
    } else {
        format!("{}", n)
    }
}

fn format_file_size(path: &Path) -> String {
    std::fs::metadata(path)
        .map(|m| output::format_size(m.len() as usize))
        .unwrap_or_default()
}
