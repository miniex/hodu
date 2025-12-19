//! Doctor command - diagnose available devices and buildable targets on this host

use crate::output::{self, colors};
use crate::plugins::{load_registry, PluginManager};
use hodu_plugin::current_host_triple;

pub fn execute() -> Result<(), Box<dyn std::error::Error>> {
    let host = current_host_triple();
    let use_color = output::supports_color();

    // Header
    if use_color {
        println!("{}{}Host{} {}", colors::BOLD, colors::CYAN, colors::RESET, host);
    } else {
        println!("Host: {}", host);
    }
    println!();

    let registry = load_registry()?;

    // Collect backend plugins
    let backends: Vec<_> = registry.backends().collect();

    if backends.is_empty() {
        if use_color {
            println!("{}No backend plugins installed.{}", colors::YELLOW, colors::RESET);
        } else {
            println!("No backend plugins installed.");
        }
        println!();
        println!("Install a backend plugin:");
        println!("  hodu plugin install aot-cpu");
        return Ok(());
    }

    // Show available devices
    print_section_header("Devices", use_color);

    let mut has_devices = false;
    for plugin in &backends {
        if !plugin.enabled {
            continue;
        }
        for device in &plugin.capabilities.devices {
            if use_color {
                println!(
                    "  {}●{} {:<12} {}({}){}",
                    colors::GREEN,
                    colors::RESET,
                    device,
                    colors::CYAN,
                    plugin.name,
                    colors::RESET
                );
            } else {
                println!("  ● {:<12} ({})", device, plugin.name);
            }
            has_devices = true;
        }
    }
    if !has_devices {
        println!("  (none)");
    }
    println!();

    // Show buildable targets per plugin
    print_section_header("Buildable Targets", use_color);

    // Use shorter timeout for doctor diagnostics (10 seconds per plugin operation)
    const DOCTOR_TIMEOUT_SECS: u64 = 10;
    let mut manager = PluginManager::with_timeout(DOCTOR_TIMEOUT_SECS)?;

    for plugin in &backends {
        if !plugin.enabled {
            continue;
        }

        // Check if plugin has build capability
        if !plugin.capabilities.builder.unwrap_or(false) {
            continue;
        }

        // Get plugin client and fetch targets
        let client = match manager.get_plugin(&plugin.name) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("  Warning: Failed to connect to plugin '{}': {}", plugin.name, e);
                continue;
            },
        };

        match client.list_targets() {
            Ok(result) => {
                // Plugin name as subheader
                if use_color {
                    println!("  {}{}{}:", colors::BOLD, plugin.name, colors::RESET);
                } else {
                    println!("  {}:", plugin.name);
                }

                // Parse and show only buildable targets
                for line in result.formatted.lines() {
                    let trimmed = line.trim();
                    if trimmed.starts_with('✓') {
                        let target = trimmed.trim_start_matches('✓').trim();
                        if use_color {
                            println!("    {}✓{} {}", colors::GREEN, colors::RESET, target);
                        } else {
                            println!("    ✓ {}", target);
                        }
                    }
                }
            },
            Err(_) => {
                if use_color {
                    println!(
                        "  {}{}{}: {}(failed to fetch targets){}",
                        colors::BOLD,
                        plugin.name,
                        colors::RESET,
                        colors::RED,
                        colors::RESET
                    );
                } else {
                    println!("  {}: (failed to fetch targets)", plugin.name);
                }
            },
        }
    }

    Ok(())
}

fn print_section_header(title: &str, use_color: bool) {
    if use_color {
        println!("{}{}{}{}", colors::BOLD, colors::CYAN, title, colors::RESET);
    } else {
        println!("{}", title);
    }
}
