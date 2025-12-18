fn main() {
    println!("cargo::rerun-if-changed=Cargo.toml");
    // TARGET is always set by cargo during build, but provide fallback for edge cases
    let target = std::env::var("TARGET").unwrap_or_else(|_| "unknown".to_string());
    println!("cargo::rustc-env=HOST_TARGET={}", target);
}
