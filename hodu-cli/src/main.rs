use clap::{Parser, Subcommand};
use hodu_cli::commands;
use hodu_cli::output;

#[derive(Parser)]
#[command(name = "hodu")]
#[command(author, version, about = "hodu", long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Run a model
    Run(commands::run::RunArgs),

    /// Build a model to native artifact
    Build(commands::build::BuildArgs),

    /// Convert models and tensors between formats
    Convert(commands::convert::ConvertArgs),

    /// Inspect a model file
    Inspect(commands::inspect::InspectArgs),

    /// Diagnose available devices and buildable targets on this host
    Doctor,

    /// Manage plugins
    Plugin(commands::plugin::PluginArgs),

    /// Clean build cache
    Clean(commands::clean::CleanArgs),

    /// Show version information
    Version,

    /// Generate shell completions
    Completions(commands::completions::CompletionsArgs),
}

fn main() {
    // First-run setup: show plugin installation wizard if no plugins installed
    // Runs before command execution but after parsing, so user's command is preserved
    if commands::setup::is_first_run() && !commands::setup::was_setup_shown() {
        if let Err(e) = commands::setup::run_setup() {
            output::warning(&format!("Setup skipped: {e}"));
        }
        if let Err(e) = commands::setup::mark_setup_shown() {
            output::warning(&format!("Failed to mark setup shown: {e}"));
        }
        // Continue to execute user's command after setup (don't return early)
    }

    let cli = Cli::parse();

    let result = match cli.command {
        Commands::Run(args) => commands::run::execute(args),
        Commands::Build(args) => commands::build::execute(args),
        Commands::Convert(args) => commands::convert::execute(args),
        Commands::Inspect(args) => commands::inspect::execute(args),
        Commands::Doctor => commands::doctor::execute(),
        Commands::Plugin(args) => commands::plugin::execute(args),
        Commands::Clean(args) => commands::clean::execute(args),
        Commands::Version => commands::version::execute(),
        Commands::Completions(args) => commands::completions::execute::<Cli>(args),
    };

    if let Err(e) = result {
        output::error(&format!("{e}"));
        std::process::exit(1);
    }
}
