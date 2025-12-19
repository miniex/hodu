//! Plugin command - manage plugins (install, remove, list, etc.)
//!
//! This command manages JSON-RPC based plugins as standalone executables.

mod install;
mod update;

use crate::output;
use crate::plugins::{
    backend_plugin_name, format_plugin_name, load_registry, load_registry_mut, PluginManager, PluginRegistry,
    PluginSource,
};
use clap::{Args, Subcommand};
use std::path::PathBuf;

pub use install::{get_plugins_dir, install_from_git, install_from_path, install_from_registry};
pub use update::update_plugins;

#[derive(Args)]
pub struct PluginArgs {
    #[command(subcommand)]
    pub command: PluginCommands,
}

#[derive(Subcommand)]
pub enum PluginCommands {
    /// List installed plugins
    List,

    /// Show plugin info (spawns plugin to get runtime info)
    Info(InfoArgs),

    /// Install a plugin
    Install(InstallArgs),

    /// Remove a plugin
    Remove(RemoveArgs),

    /// Update plugins
    Update(UpdateArgs),

    /// Enable a plugin
    Enable(EnableArgs),

    /// Disable a plugin
    Disable(DisableArgs),

    /// Verify plugin integrity (check binaries exist, dependencies satisfied)
    Verify,
}

#[derive(Args)]
pub struct InfoArgs {
    /// Plugin name
    pub name: String,
}

#[derive(Args)]
pub struct InstallArgs {
    /// Plugin name (from official registry)
    pub name: Option<String>,

    /// Install from local path
    #[arg(long)]
    pub path: Option<PathBuf>,

    /// Install from git repository
    #[arg(long)]
    pub git: Option<String>,

    /// Subdirectory in git repository
    #[arg(long)]
    pub subdir: Option<String>,

    /// Git tag or branch
    #[arg(long)]
    pub tag: Option<String>,

    /// Force reinstall
    #[arg(long)]
    pub force: bool,

    /// Debug build
    #[arg(long)]
    pub debug: bool,

    /// Show detailed build output
    #[arg(long, short = 'v')]
    pub verbose: bool,
}

#[derive(Args)]
pub struct RemoveArgs {
    /// Plugin name
    pub name: String,
}

#[derive(Args)]
pub struct UpdateArgs {
    /// Plugin name (update all if not specified)
    pub name: Option<String>,
}

#[derive(Args)]
pub struct EnableArgs {
    /// Plugin name
    pub name: String,
}

#[derive(Args)]
pub struct DisableArgs {
    /// Plugin name
    pub name: String,
}

pub fn execute(args: PluginArgs) -> Result<(), Box<dyn std::error::Error>> {
    match args.command {
        PluginCommands::List => list_plugins(),
        PluginCommands::Info(info_args) => info_plugin(info_args),
        PluginCommands::Install(install_args) => do_install(install_args),
        PluginCommands::Remove(remove_args) => remove_plugin(remove_args),
        PluginCommands::Update(update_args) => update_plugins(update_args.name.as_deref()),
        PluginCommands::Enable(enable_args) => enable_plugin(enable_args),
        PluginCommands::Disable(disable_args) => disable_plugin(disable_args),
        PluginCommands::Verify => verify_plugins(),
    }
}

/// Helper to list plugins with a common pattern
fn list_plugin_section<'a, I, F, G>(plugins: I, use_color: bool, get_tags: F, get_extra: G)
where
    I: Iterator<Item = &'a hodu_plugin_runtime::PluginEntry>,
    F: Fn(&hodu_plugin_runtime::PluginCapabilities) -> Vec<&'static str>,
    G: Fn(&hodu_plugin_runtime::PluginCapabilities, &mut String),
{
    let plugins: Vec<_> = plugins.collect();
    if plugins.is_empty() {
        print_empty(use_color);
    } else {
        let mut extra_buf = String::with_capacity(64);
        for plugin in plugins {
            let caps = &plugin.capabilities;
            let tags = get_tags(caps);
            extra_buf.clear();
            get_extra(caps, &mut extra_buf);
            print_plugin_row(
                &plugin.name,
                &plugin.version,
                &tags,
                Some(&extra_buf),
                plugin.enabled,
                use_color,
            );
        }
    }
}

/// Helper to format extensions as ".ext1 .ext2"
fn format_extensions(extensions: &[String], buf: &mut String) {
    for (i, ext) in extensions.iter().enumerate() {
        if i > 0 {
            buf.push(' ');
        }
        buf.push('.');
        buf.push_str(ext);
    }
}

fn list_plugins() -> Result<(), Box<dyn std::error::Error>> {
    let use_color = output::supports_color();

    let registry = load_registry()?;

    // Backend plugins
    print_section("Backend Plugins", use_color);
    list_plugin_section(
        registry.all_backends(),
        use_color,
        |caps| {
            let mut tags = Vec::with_capacity(2);
            if caps.runner.unwrap_or(false) {
                tags.push("run");
            }
            if caps.builder.unwrap_or(false) {
                tags.push("build");
            }
            tags
        },
        |caps, buf| buf.push_str(&caps.devices.join(", ")),
    );
    println!();

    // Model format plugins
    print_section("Model Format Plugins", use_color);
    list_plugin_section(
        registry.all_model_formats(),
        use_color,
        |caps| {
            let mut tags = Vec::with_capacity(2);
            if caps.load_model.unwrap_or(false) {
                tags.push("load");
            }
            if caps.save_model.unwrap_or(false) {
                tags.push("save");
            }
            tags
        },
        |caps, buf| format_extensions(&caps.model_extensions, buf),
    );
    println!();

    // Tensor format plugins
    print_section("Tensor Format Plugins", use_color);
    list_plugin_section(
        registry.all_tensor_formats(),
        use_color,
        |caps| {
            let mut tags = Vec::with_capacity(2);
            if caps.load_tensor.unwrap_or(false) {
                tags.push("load");
            }
            if caps.save_tensor.unwrap_or(false) {
                tags.push("save");
            }
            tags
        },
        |caps, buf| format_extensions(&caps.tensor_extensions, buf),
    );

    Ok(())
}

fn print_section(title: &str, use_color: bool) {
    use output::colors;
    if use_color {
        println!("{}{}{}{}", colors::BOLD, colors::CYAN, title, colors::RESET);
    } else {
        println!("{}", title);
    }
}

fn print_empty(use_color: bool) {
    use output::colors;
    if use_color {
        println!("  {}(none){}", colors::YELLOW, colors::RESET);
    } else {
        println!("  (none)");
    }
}

fn print_plugin_row(name: &str, version: &str, tags: &[&str], extra: Option<&str>, enabled: bool, use_color: bool) {
    use output::colors;

    let status_icon = if enabled { "●" } else { "○" };
    let status_color = if enabled { colors::GREEN } else { colors::YELLOW };

    let tags_str = if tags.is_empty() {
        String::new()
    } else if use_color {
        tags.iter()
            .map(|t| format!("{}[{}]{}", colors::CYAN, t, colors::RESET))
            .collect::<Vec<_>>()
            .join(" ")
    } else {
        tags.iter().map(|t| format!("[{}]", t)).collect::<Vec<_>>().join(" ")
    };

    let extra_str = extra
        .filter(|s| !s.is_empty())
        .map(|s| format!("  {}", s))
        .unwrap_or_default();

    if use_color {
        println!(
            "  {}{}{} {}{:<16}{} {}{}{}",
            status_color,
            status_icon,
            colors::RESET,
            colors::BOLD,
            name,
            colors::RESET,
            version,
            if tags_str.is_empty() {
                String::new()
            } else {
                format!("  {}", tags_str)
            },
            extra_str
        );
    } else {
        println!(
            "  {} {:<16} {}{}{}",
            status_icon,
            name,
            version,
            if tags_str.is_empty() {
                String::new()
            } else {
                format!("  {}", tags_str)
            },
            extra_str
        );
    }
}

fn info_plugin(args: InfoArgs) -> Result<(), Box<dyn std::error::Error>> {
    use output::colors;
    let use_color = output::supports_color();

    let registry = load_registry()?;

    // Find plugin
    let plugin = registry.find(&args.name).or_else(|| {
        let backend_name = backend_plugin_name(&args.name);
        let format_name = format_plugin_name(&args.name);
        registry.find(&backend_name).or_else(|| registry.find(&format_name))
    });

    let plugin = match plugin {
        Some(p) => p,
        None => return Err(format!("Plugin '{}' not found.", args.name).into()),
    };

    // Header
    if use_color {
        println!(
            "{}{}{} {}v{}{}",
            colors::BOLD,
            plugin.name,
            colors::RESET,
            colors::CYAN,
            plugin.version,
            colors::RESET
        );
    } else {
        println!("{} v{}", plugin.name, plugin.version);
    }

    if let Some(desc) = &plugin.description {
        println!("{}", desc);
    }
    println!();

    // Info section
    print_section("Info", use_color);
    print_info_row("Type", &format!("{:?}", plugin.plugin_type), use_color);
    if let Some(lic) = &plugin.license {
        print_info_row("License", lic, use_color);
    }
    print_info_row("Plugin", &plugin.plugin_version, use_color);

    let status = if plugin.enabled {
        if use_color {
            format!("{}enabled{}", colors::GREEN, colors::RESET)
        } else {
            "enabled".to_string()
        }
    } else if use_color {
        format!("{}disabled{}", colors::YELLOW, colors::RESET)
    } else {
        "disabled".to_string()
    };
    print_info_row("Status", &status, use_color);
    println!();

    // Spawn plugin to get runtime info
    let mut manager = PluginManager::new()?;
    let _client = manager.get_plugin(&plugin.name)?;

    let info = manager.get_info(&plugin.name).cloned();
    let has_build_capability = info
        .as_ref()
        .map(|i| i.capabilities.contains(&"backend.build".to_string()))
        .unwrap_or(false);

    if let Some(info) = info {
        // Capabilities section
        print_section("Capabilities", use_color);

        // Show devices for backend plugins
        if let Some(devices) = &info.devices {
            if !devices.is_empty() {
                print_info_row("Devices", &devices.join(", "), use_color);
            }
        }

        // Show extensions for format plugins
        if let Some(extensions) = &info.model_extensions {
            if !extensions.is_empty() {
                let ext_str = extensions
                    .iter()
                    .map(|e| format!(".{}", e))
                    .collect::<Vec<_>>()
                    .join(" ");
                print_info_row("Extensions", &ext_str, use_color);
            }
        }
        if let Some(extensions) = &info.tensor_extensions {
            if !extensions.is_empty() {
                let ext_str = extensions
                    .iter()
                    .map(|e| format!(".{}", e))
                    .collect::<Vec<_>>()
                    .join(" ");
                print_info_row("Extensions", &ext_str, use_color);
            }
        }

        // Format capabilities nicely
        let caps_str = info
            .capabilities
            .iter()
            .map(|c| {
                let short = c.split('.').next_back().unwrap_or(c);
                if use_color {
                    format!("{}[{}]{}", colors::CYAN, short, colors::RESET)
                } else {
                    format!("[{}]", short)
                }
            })
            .collect::<Vec<_>>()
            .join(" ");
        println!("  {}", caps_str);
        println!();
    }

    // Show supported targets for backend plugins with build capability
    if has_build_capability {
        let client = manager.get_plugin(&plugin.name)?;
        match client.list_targets() {
            Ok(targets_result) => {
                print_section("Supported Targets", use_color);

                for line in targets_result.formatted.lines() {
                    let trimmed = line.trim();
                    if trimmed.is_empty() || trimmed.starts_with("Host:") {
                        continue;
                    }
                    if trimmed.starts_with('✓') {
                        let target = trimmed.trim_start_matches('✓').trim();
                        if use_color {
                            println!("  {}✓{} {}", colors::GREEN, colors::RESET, target);
                        } else {
                            println!("  ✓ {}", target);
                        }
                    } else if trimmed.starts_with('✗') {
                        let target = trimmed.trim_start_matches('✗').trim();
                        if use_color {
                            println!("  {}✗{} {}", colors::RED, colors::RESET, target);
                        } else {
                            println!("  ✗ {}", target);
                        }
                    }
                }
            },
            Err(e) => {
                if use_color {
                    println!(
                        "{}Supported Targets:{} {}(failed: {}){}",
                        colors::BOLD,
                        colors::RESET,
                        colors::RED,
                        e,
                        colors::RESET
                    );
                } else {
                    println!("Supported Targets: (failed: {})", e);
                }
            },
        }
    }

    Ok(())
}

fn print_info_row(label: &str, value: &str, use_color: bool) {
    use output::colors;
    if use_color {
        println!("  {}{:<12}{} {}", colors::CYAN, label, colors::RESET, value);
    } else {
        println!("  {:<12} {}", label, value);
    }
}

fn do_install(args: InstallArgs) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(path) = &args.path {
        let source = PluginSource::Local {
            path: path.canonicalize()?.to_string_lossy().to_string(),
        };
        install_from_path(path, args.debug, args.force, args.verbose, source)
    } else if let Some(git) = &args.git {
        install_from_git(
            git,
            args.subdir.as_deref(),
            args.tag.as_deref(),
            args.debug,
            args.force,
            args.verbose,
        )
    } else if let Some(name) = &args.name {
        install_from_registry(name, args.tag.as_deref(), args.debug, args.force, args.verbose)
    } else {
        Err("No plugin specified. Use <name>, --path, or --git.".into())
    }
}

fn remove_plugin(args: RemoveArgs) -> Result<(), Box<dyn std::error::Error>> {
    let (mut registry, registry_path) = load_registry_mut()?;

    // Resolve plugin name and get plugin info in one lookup to avoid TOCTOU
    let (name, version) = {
        // Try exact name first
        if let Some(plugin) = registry.find(&args.name) {
            (plugin.name.clone(), plugin.version.clone())
        } else {
            // Try with prefixes
            let backend_name = backend_plugin_name(&args.name);
            let format_name = format_plugin_name(&args.name);

            if let Some(plugin) = registry.find(&backend_name) {
                (plugin.name.clone(), plugin.version.clone())
            } else if let Some(plugin) = registry.find(&format_name) {
                (plugin.name.clone(), plugin.version.clone())
            } else {
                return Err(format!(
                    "Plugin '{}' not found.\n\nInstalled plugins:\n{}",
                    args.name,
                    list_installed_plugins(&registry)
                )
                .into());
            }
        }
    };

    // Delete the plugin directory
    let plugins_dir = get_plugins_dir()?;
    let plugin_dir = plugins_dir.join(&name);

    output::removing(&format!("{} v{}", name, version));

    if plugin_dir.exists() {
        std::fs::remove_dir_all(&plugin_dir)?;
    }

    // Remove from registry
    registry.remove(&name);
    registry.save(&registry_path)?;

    output::removed(&format!("{} v{}", name, version));
    Ok(())
}

fn enable_plugin(args: EnableArgs) -> Result<(), Box<dyn std::error::Error>> {
    let (mut registry, registry_path) = load_registry_mut()?;

    // Try to find plugin with various name formats
    let name = find_plugin_name(&registry, &args.name)?;

    if registry.enable(&name) {
        registry.save(&registry_path)?;
        output::finished(&format!("enabled {}", name));
    } else {
        return Err(format!("Plugin '{}' not found.", args.name).into());
    }

    Ok(())
}

fn disable_plugin(args: DisableArgs) -> Result<(), Box<dyn std::error::Error>> {
    let (mut registry, registry_path) = load_registry_mut()?;

    // Try to find plugin with various name formats
    let name = find_plugin_name(&registry, &args.name)?;

    // Check if other plugins depend on this one
    let dependents: Vec<String> = registry
        .plugins
        .iter()
        .filter(|p| p.enabled && p.dependencies.contains(&name))
        .map(|p| p.name.clone())
        .collect();

    if !dependents.is_empty() {
        return Err(format!("Cannot disable '{}': required by {}", name, dependents.join(", ")).into());
    }

    if registry.disable(&name) {
        registry.save(&registry_path)?;
        output::finished(&format!("disabled {}", name));
    } else {
        return Err(format!("Plugin '{}' not found.", args.name).into());
    }

    Ok(())
}

fn verify_plugins() -> Result<(), Box<dyn std::error::Error>> {
    let registry = load_registry()?;
    let plugins_dir = get_plugins_dir()?;

    let mut issues = Vec::new();
    let mut ok_count = 0;

    for plugin in &registry.plugins {
        let mut plugin_issues = Vec::new();

        // Check if binary exists
        let binary_path = plugins_dir.join(&plugin.name).join(&plugin.binary);
        if !binary_path.exists() {
            plugin_issues.push(format!("binary not found: {}", binary_path.display()));
        }

        // Check dependencies (only for enabled plugins)
        if plugin.enabled {
            let missing_deps: Vec<&String> = plugin
                .dependencies
                .iter()
                .filter(|dep| registry.find(dep).map(|p| !p.enabled).unwrap_or(true))
                .collect();

            if !missing_deps.is_empty() {
                plugin_issues.push(format!(
                    "missing dependencies: {}",
                    missing_deps.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", ")
                ));
            }
        }

        if plugin_issues.is_empty() {
            ok_count += 1;
        } else {
            let status = if plugin.enabled { "" } else { " (disabled)" };
            issues.push(format!("  {}{}: {}", plugin.name, status, plugin_issues.join("; ")));
        }
    }

    if issues.is_empty() {
        println!("All {} plugins verified OK.", ok_count);
    } else {
        println!(
            "Verified {} plugins, {} with issues:",
            ok_count + issues.len(),
            issues.len()
        );
        for issue in issues {
            println!("{}", issue);
        }
    }

    Ok(())
}

fn find_plugin_name(registry: &PluginRegistry, name: &str) -> Result<String, Box<dyn std::error::Error>> {
    if registry.find(name).is_some() {
        return Ok(name.to_string());
    }

    let backend_name = backend_plugin_name(name);
    if registry.find(&backend_name).is_some() {
        return Ok(backend_name);
    }

    let format_name = format_plugin_name(name);
    if registry.find(&format_name).is_some() {
        return Ok(format_name);
    }

    Err(format!("Plugin '{}' not found.", name).into())
}

fn list_installed_plugins(registry: &PluginRegistry) -> String {
    let mut result = String::new();

    let backends: Vec<_> = registry.backends().collect();
    if !backends.is_empty() {
        result.push_str("  Backend: ");
        result.push_str(&backends.iter().map(|p| p.name.as_str()).collect::<Vec<_>>().join(", "));
        result.push('\n');
    }

    let model_formats: Vec<_> = registry.model_formats().collect();
    if !model_formats.is_empty() {
        result.push_str("  Model format: ");
        result.push_str(
            &model_formats
                .iter()
                .map(|p| p.name.as_str())
                .collect::<Vec<_>>()
                .join(", "),
        );
        result.push('\n');
    }

    let tensor_formats: Vec<_> = registry.tensor_formats().collect();
    if !tensor_formats.is_empty() {
        result.push_str("  Tensor format: ");
        result.push_str(
            &tensor_formats
                .iter()
                .map(|p| p.name.as_str())
                .collect::<Vec<_>>()
                .join(", "),
        );
        result.push('\n');
    }

    if result.is_empty() {
        result.push_str("  (none installed)\n");
    }

    result
}
