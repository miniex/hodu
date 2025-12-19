//! Clean command - remove cached build artifacts

use crate::output;
use crate::plugins::BACKEND_PREFIX;
use clap::Args;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

/// Minimum file count to show progress (avoid progress noise for small dirs)
const MIN_FILES_FOR_PROGRESS: usize = 100;

/// Progress update interval (every N files)
const PROGRESS_UPDATE_INTERVAL: usize = 50;

#[derive(Args)]
pub struct CleanArgs {
    /// Only show what would be deleted (dry run)
    #[arg(long)]
    pub dry_run: bool,

    /// Clean specific backend cache only
    #[arg(long)]
    pub backend: Option<String>,

    /// Clean all caches including plugin registry (use with caution)
    #[arg(long)]
    pub all: bool,
}

pub fn execute(args: CleanArgs) -> Result<(), Box<dyn std::error::Error>> {
    let hodu_dir = dirs::home_dir()
        .ok_or("Could not determine home directory")?
        .join(".hodu");

    if !hodu_dir.exists() {
        println!("Nothing to clean.");
        return Ok(());
    }

    let cache_dir = hodu_dir.join("cache");

    if args.all {
        // Clean everything
        clean_directory(&hodu_dir, "all hodu data", args.dry_run)?;
    } else if let Some(backend) = &args.backend {
        // Clean specific backend
        let backend_cache = cache_dir.join(backend);
        if backend_cache.exists() {
            clean_directory(&backend_cache, &format!("{} cache", backend), args.dry_run)?;
        } else {
            // Try with prefix
            let prefixed = format!("{}{}-plugin", BACKEND_PREFIX, backend);
            let backend_cache = cache_dir.join(&prefixed);
            if backend_cache.exists() {
                clean_directory(&backend_cache, &format!("{} cache", backend), args.dry_run)?;
            } else {
                println!("No cache found for backend '{}'", backend);
            }
        }
    } else {
        // Clean all caches (default)
        if cache_dir.exists() {
            clean_directory(&cache_dir, "build cache", args.dry_run)?;
        } else {
            println!("Nothing to clean.");
        }
    }

    Ok(())
}

fn clean_directory(path: &Path, name: &str, dry_run: bool) -> Result<(), Box<dyn std::error::Error>> {
    let (size, file_count) = dir_stats(path)?;
    let size_str = output::format_size(size);

    if dry_run {
        output::skipping(&format!("{} ({}, {} files) - dry run", name, size_str, file_count));
    } else {
        output::cleaning(&format!("{} ({}, {} files)", name, size_str, file_count));

        // Show progress for large directories
        if file_count >= MIN_FILES_FOR_PROGRESS {
            remove_dir_with_progress(path, file_count)?;
        } else {
            std::fs::remove_dir_all(path)?;
        }

        output::removed(name);
    }

    Ok(())
}

/// Remove directory with progress indication for large directories
fn remove_dir_with_progress(path: &Path, total_files: usize) -> Result<(), Box<dyn std::error::Error>> {
    let deleted = AtomicUsize::new(0);
    let start = Instant::now();

    remove_dir_recursive(path, &deleted, total_files)?;

    let elapsed = start.elapsed();
    if elapsed.as_secs() >= 1 {
        eprintln!(
            "\r  Deleted {} files in {:.1}s",
            deleted.load(Ordering::Relaxed),
            elapsed.as_secs_f32()
        );
    }

    Ok(())
}

/// Recursively remove directory contents with progress updates
fn remove_dir_recursive(path: &Path, deleted: &AtomicUsize, total: usize) -> Result<(), Box<dyn std::error::Error>> {
    if path.is_symlink() {
        std::fs::remove_file(path)?;
        return Ok(());
    }

    if path.is_dir() {
        for entry in std::fs::read_dir(path)? {
            let entry = entry?;
            let entry_path = entry.path();

            if entry_path.is_dir() && !entry_path.is_symlink() {
                remove_dir_recursive(&entry_path, deleted, total)?;
            } else {
                std::fs::remove_file(&entry_path)?;
                let count = deleted.fetch_add(1, Ordering::Relaxed) + 1;

                // Print progress periodically
                if count.is_multiple_of(PROGRESS_UPDATE_INTERVAL) {
                    let percent = (count * 100) / total;
                    eprint!("\r  Deleting... {}% ({}/{})", percent, count, total);
                }
            }
        }
        std::fs::remove_dir(path)?;
    } else {
        std::fs::remove_file(path)?;
        deleted.fetch_add(1, Ordering::Relaxed);
    }

    Ok(())
}

/// Get directory size and file count
fn dir_stats(path: &Path) -> Result<(usize, usize), Box<dyn std::error::Error>> {
    let mut size = 0;
    let mut count = 0;
    // Skip symlinks to avoid cycles and double-counting
    if path.is_symlink() {
        return Ok((0, 0));
    }
    if path.is_dir() {
        for entry in std::fs::read_dir(path)? {
            let entry = entry?;
            let path = entry.path();
            // Skip symlinks
            if path.is_symlink() {
                continue;
            }
            if path.is_dir() {
                let (sub_size, sub_count) = dir_stats(&path)?;
                size += sub_size;
                count += sub_count;
            } else {
                size += entry.metadata()?.len() as usize;
                count += 1;
            }
        }
    } else {
        size = std::fs::metadata(path)?.len() as usize;
        count = 1;
    }
    Ok((size, count))
}
