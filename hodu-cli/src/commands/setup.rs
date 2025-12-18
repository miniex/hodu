//! First-run setup command - install recommended plugins

use crate::commands::plugin::{get_plugins_dir, install_from_registry};
use crate::output;
use crate::plugins::load_registry;
use inquire::Select;
use serde::Deserialize;
use std::collections::HashMap;

const DEFAULT_PLUGINS_URL: &str =
    "https://raw.githubusercontent.com/daminstudio/hodu-plugins/main/default-plugins.json";

#[derive(Deserialize)]
struct DefaultPlugins {
    presets: HashMap<String, PresetInfo>,
}

#[derive(Deserialize)]
struct PresetInfo {
    description: String,
    plugins: Vec<String>,
}

/// Check if this is the first run (no plugins installed)
pub fn is_first_run() -> bool {
    match load_registry() {
        Ok(registry) => registry.plugins.is_empty(),
        Err(_) => true,
    }
}

/// Run the first-time setup wizard
pub fn run_setup() -> Result<(), Box<dyn std::error::Error>> {
    println!();
    output::info("Welcome to Hodu! No plugins installed yet.");
    println!();

    // Fetch presets from remote
    let presets = match fetch_default_plugins() {
        Ok(p) => p,
        Err(e) => {
            output::warning(&format!("Failed to fetch plugin presets: {}", e));
            output::info("You can install plugins manually with: hodu plugin install <name>");
            return Ok(());
        },
    };

    // Build options list
    let mut options: Vec<String> = presets
        .iter()
        .map(|(name, info)| format!("{:<16} - {}", name, info.description))
        .collect();
    options.sort();
    options.push("skip             - I'll install plugins manually".to_string());

    let selection = Select::new("Install recommended plugins?", options).prompt()?;

    let preset_name = selection.split_whitespace().next().unwrap_or("skip");

    if preset_name == "skip" {
        println!();
        output::info("Skipped. You can install plugins later with:");
        println!("  hodu plugin install <name>");
        println!();
        println!("Available plugins: https://github.com/daminstudio/hodu-plugins");
        return Ok(());
    }

    let plugins = match presets.get(preset_name) {
        Some(info) => &info.plugins,
        None => {
            output::warning("Invalid preset selected");
            return Ok(());
        },
    };

    println!();
    output::info(&format!("Installing {} plugins...", plugins.len()));
    println!();

    for plugin in plugins {
        if let Err(e) = install_from_registry(plugin, None, false, false) {
            output::warning(&format!("Failed to install {}: {}", plugin, e));
        }
    }

    println!();
    output::finished("Setup complete!");

    Ok(())
}

fn fetch_default_plugins() -> Result<HashMap<String, PresetInfo>, Box<dyn std::error::Error>> {
    use std::time::Duration;

    // Configure timeout (30 seconds)
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(30)))
        .build()
        .into();

    let response = agent.get(DEFAULT_PLUGINS_URL).call()?;
    let body = response.into_body().read_to_string()?;
    let data: DefaultPlugins = serde_json::from_str(&body)?;
    Ok(data.presets)
}

/// Mark that setup has been shown (create marker file)
pub fn mark_setup_shown() -> Result<(), Box<dyn std::error::Error>> {
    let plugins_dir = get_plugins_dir()?;
    std::fs::create_dir_all(&plugins_dir)?;

    let marker = plugins_dir.join(".setup_shown");
    std::fs::write(marker, "")?;

    Ok(())
}

/// Check if setup has been shown before
pub fn was_setup_shown() -> bool {
    get_plugins_dir()
        .map(|p| p.join(".setup_shown").exists())
        .unwrap_or(false)
}
