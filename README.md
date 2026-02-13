# Hodu, a Rust ML toolkit

[![Crates.io](https://img.shields.io/crates/v/hodu.svg)](https://crates.io/crates/hodu)
[![Doc.rs](https://docs.rs/hodu/badge.svg)](https://docs.rs/hodu)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](./LICENSE)
[![Discord](https://img.shields.io/discord/1440598688861917196?logo=discord&logoColor=white&label=Discord)](https://discord.gg/eyuKxPw9As)

> **Hodu** (호두) is a Korean word meaning "walnut".

<p align="center">
    <img src="./assets/hodu/sd_logo.png" alt="Hodu Avatar" width="384" />
</p>

**Hodu** is a machine learning library built with user convenience at its core.

### Core Differentiators

Built on **Rust's foundation of memory safety and zero-cost abstractions**, Hodu offers unique advantages:

- **Memory Safety by Design**: Leverage Rust's ownership system to eliminate common ML deployment issues like memory leaks and data races
- **Zero-Cost Abstractions**: High-level APIs that compile down to efficient machine code without runtime overhead
- **Multi-Device Support**: CPU, Metal GPU (macOS), and CUDA GPU acceleration

## Get started

### Requirements

**Required**
- Rust 1.93.0 or later (latest stable version recommended)

**Optional**
- **OpenBLAS 0.3.30+** - For optimized linear algebra operations on CPU
  - macOS: `brew install openblas gfortran`
  - Linux: `sudo apt install libopenblas-dev pkg-config gfortran`
  - Windows: Install via vcpkg or MinGW

- **CUDA Toolkit** - Required when using the `cuda` feature
  - Download from [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)

- **Xcode Command Line Tools** - Required when using the `metal` feature on macOS
  - `xcode-select --install`

### Example

```rust
use hodu::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Set the runtime device (CPU, CUDA, Metal)
    set_runtime_device(Device::CPU);

    // Create random tensors
    let a = Tensor::randn(&[2, 3], 0f32, 1.)?;
    let b = Tensor::randn(&[3, 4], 0f32, 1.)?;

    // Matrix multiplication
    let c = a.matmul(&b)?;

    println!("{}", c);
    println!("{:?}", c);

    Ok(())
}
```

With the `cuda` feature enabled, you can use CUDA with the following setting:

```diff
- set_runtime_device(Device::CPU);
+ set_runtime_device(Device::CUDA(0));
```

## Features

### Default Features
| Feature | Description | Dependencies |
|---------|-------------|--------------|
| `serde` | Serialization/deserialization support | - |

### Optional Features
| Feature | Description | Dependencies |
|---------|-------------|--------------|
| `openblas` | Use OpenBLAS for CPU backend (instead of OS-provided BLAS) | OpenBLAS library |
| `cuda` | NVIDIA CUDA GPU support | CUDA toolkit |
| `metal` | Apple Metal GPU support | Metal framework |

### Optional Data Type Features

By default, Hodu supports these data types: `bool`, `f8e4m3`, `bf16`, `f16`, `f32`, `u8`, `u32`, `i8`, `i32`.

Additional data types can be enabled with feature flags to reduce compilation time:

| Feature | Description |
|---------|-------------|
| `f8e5m2` | Enable 8-bit floating point (E5M2) support |
| `f64` | Enable 64-bit floating point support |
| `u16` | Enable unsigned 16-bit integer support |
| `u64` | Enable unsigned 64-bit integer support |
| `i16` | Enable signed 16-bit integer support |
| `i64` | Enable signed 64-bit integer support |

**Compilation Performance**: Disabling unused data types can reduce compilation time by up to 30-40%. If you don't need these specific data types, consider building without these features.

## Supported Platforms

| Target Triple | Device | Features | Status |
|--------------|--------|----------|--------|
| x86_64-unknown-linux-gnu | CPU | - | Stable |
| | CUDA | `cuda` | Stable |
| aarch64-unknown-linux-gnu | CPU | - | Stable |
| x86_64-apple-darwin | CPU | - | Not planned |
| aarch64-apple-darwin | CPU | - | Stable |
| | Metal | `metal` | Stable |
| x86_64-pc-windows-msvc | CPU | - | Experimental |
| | CUDA | `cuda` | Experimental |

## Contributing

> **Note**: The project structure is still being established. For now, please open an issue first before submitting pull requests. This helps us coordinate work and avoid conflicts during this early stage.

Bug reports, feature requests, and feedback are always welcome!

Check out [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the [BSD-3-Clause License](./LICENSE).

### Assets

All icons, characters, and visual assets are created by **해꿈(sundream)** and © Han Damin. All rights reserved. These assets may not be used, modified, or distributed for any purpose, commercial or non-commercial, without explicit written permission from the copyright holder. See [Assets License](./assets/hodu/LICENSE) for details.

For licensing inquiries, contact: [miniex@daminstudio.net](mailto:miniex@daminstudio.net)
