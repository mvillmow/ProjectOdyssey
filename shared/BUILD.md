# Build System Guide - ML Odyssey Shared Library

This document describes how to build and package the ML Odyssey shared library.

## Quick Start

```bash
# Build development package (fast)
mojo package shared

# Build release package (optimized)
mojo package shared --release -o dist/ml_odyssey_shared.mojopkg

# Build and install in one step
mojo package shared --install
```

## Build Modes

### Development Build (Default)

Fast compilation with minimal optimizations, suitable for development:

```bash
mojo package shared
```

**Characteristics**:

- Fast compilation
- Debug symbols included
- No optimizations
- Good for development and testing

### Release Build

Optimized build for production use:

```bash
mojo package shared --release -o dist/ml_odyssey_shared.mojopkg
```

**Characteristics**:

- Full optimizations enabled
- No debug symbols
- Longer compilation time
- Best performance
- Use for production and benchmarking

### Debug Build

Build with full debug information:

```bash
mojo package shared --debug
```

**Characteristics**:

- Full debug symbols
- Minimal optimizations
- Good for debugging issues
- Larger binary size

## Package Configuration

### mojo.toml

The package is configured via `mojo.toml` (or `mojoproject.toml`) in the repository root:

```toml
[package]
name = "ml-odyssey-shared"
version = "0.1.0"
description = "Shared library for ML Odyssey paper implementations"
authors = ["ML Odyssey Team"]
license = "MIT"
readme = "shared/README.md"

[dependencies]
# Mojo standard library dependencies will be listed here

[build]
# Build configuration
src = "shared"
out = "build"

[paths]
# Path configuration
tests = "tests/shared"
examples = "examples/shared"
```

### Directory Structure

The build system expects this structure:

```text
ml-odyssey/
├── mojo.toml                  # Package configuration
├── shared/                    # Source directory
│   ├── __init__.mojo          # Package root
│   ├── core/                  # Core module
│   ├── training/              # Training module
│   ├── data/                  # Data module
│   └── utils/                 # Utils module
├── tests/                     # Test directory
│   └── shared/                # Shared library tests
├── build/                     # Build artifacts (gitignored)
└── dist/                      # Distribution packages (gitignored)
```

## Build Commands

### Basic Build

```bash
# Build with default settings
mojo package shared

# Output: ./shared.mojopkg
```

### Custom Output Path

```bash
# Specify output file
mojo package shared -o build/ml_odyssey_shared.mojopkg

# Specify output directory (filename auto-generated)
mojo package shared -o build/
```

### Build and Install

```bash
# Build and install in one step
mojo package shared --install

# Equivalent to:
# mojo package shared -o /tmp/shared.mojopkg
# mojo install /tmp/shared.mojopkg
```

### Clean Build

```bash
# Remove build artifacts
rm -rf build/ dist/

# Remove installed package
mojo uninstall ml-odyssey-shared

# Full clean
rm -rf build/ dist/
mojo uninstall ml-odyssey-shared
```

## Build Options

### Optimization Levels

```bash
# No optimization (default for development)
mojo package shared

# Full optimization (release mode)
mojo package shared --release

# Optimize for size
mojo package shared --optimize-size
```

### Debug Information

```bash
# Include debug symbols
mojo package shared --debug

# Strip debug symbols (default for release)
mojo package shared --release
```

### Parallelization

```bash
# Use multiple cores for compilation
mojo package shared -j 8

# Use all available cores
mojo package shared -j auto
```

## Testing the Build

### Run Tests

```bash
# Run all tests
mojo test tests/shared/

# Run specific test file
mojo test tests/shared/test_imports.mojo

# Run tests with verbose output
mojo test tests/shared/ -v
```

### Verify Installation

```bash
# Run verification script
mojo run scripts/verify_installation.mojo

# Check installed packages
mojo list | grep ml-odyssey
```

### Import Tests

```bash
# Test imports work
mojo run -c "from shared import VERSION; print(VERSION)"
```

## Distribution

### Creating Release Package

```bash
# Build release package with version
mojo package shared --release -o dist/ml_odyssey_shared-0.1.0.mojopkg

# Generate checksums
sha256sum dist/ml_odyssey_shared-0.1.0.mojopkg > dist/checksums.txt

# Create archive for distribution
tar -czf dist/ml_odyssey_shared-0.1.0.tar.gz \
    dist/ml_odyssey_shared-0.1.0.mojopkg \
    dist/checksums.txt \
    shared/README.md \
    shared/INSTALL.md \
    shared/EXAMPLES.md
```

### Publishing (Future)

When Mojo package registry is available:

```bash
# Publish to registry
mojo publish dist/ml_odyssey_shared-0.1.0.mojopkg

# Install from registry
mojo install ml-odyssey-shared
```

## Build Targets

### Library Package

```bash
# Build shared library package
mojo package shared -o build/ml_odyssey_shared.mojopkg
```

### Static Library (Future)

When Mojo supports static libraries:

```bash
# Build static library
mojo build shared --lib -o build/libml_odyssey_shared.a
```

### Shared Object (Future)

When Mojo supports shared objects:

```bash
# Build shared object
mojo build shared --shared -o build/libml_odyssey_shared.so
```

## Build Performance

### Improving Build Speed

1. **Use multiple cores**:

   ```bash
   mojo package shared -j auto
   ```

2. **Incremental builds**: Mojo caches compiled modules

3. **Development mode**: Skip optimizations during development

   ```bash
   mojo package shared  # Fast, no optimizations
   ```

### Build Times (Approximate)

With a modern CPU (8 cores):

- **Development build**: 10-30 seconds
- **Release build**: 1-3 minutes
- **Incremental build**: 1-5 seconds (only changed files)

## Build Artifacts

### Output Files

After a successful build:

```text
build/
├── ml_odyssey_shared.mojopkg     # Main package file
├── .build_cache/                 # Compilation cache
└── intermediate/                 # Intermediate build files
```

### Package File Structure

The `.mojopkg` file contains:

- Compiled Mojo modules
- Package metadata
- Dependency information
- Optional debug symbols

## Build Environment

### Environment Variables

```bash
# Set Mojo path
export MOJO_PATH="/path/to/mojo"

# Set package search path
export MOJO_PACKAGE_PATH="/path/to/packages"

# Set optimization level
export MOJO_OPT_LEVEL="2"  # 0-3

# Enable verbose output
export MOJO_VERBOSE="1"
```

### Build Requirements

- **Mojo 24.5+**: Required for building
- **Disk Space**: ~100MB for build artifacts
- **Memory**: ~2GB recommended for parallel builds
- **CPU**: Multi-core recommended for faster builds

## Continuous Integration

### CI Build Script

```bash
#!/bin/bash
set -e

# Install dependencies
pixi install

# Build package
pixi run mojo package shared --release -o dist/ml_odyssey_shared.mojopkg

# Run tests
pixi run mojo test tests/shared/

# Verify installation
pixi run mojo package shared --install
pixi run mojo run scripts/verify_installation.mojo

# Create distribution
sha256sum dist/ml_odyssey_shared.mojopkg > dist/checksums.txt
```

### GitHub Actions Example

```yaml
name: Build and Test

on: [push, pull_request]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Install Mojo
        run: |
          # Install Mojo (see Modular docs)

      - name: Build Package
        run: mojo package shared --release -o dist/shared.mojopkg

      - name: Run Tests
        run: mojo test tests/shared/

      - name: Verify Installation
        run: |
          mojo package shared --install
          mojo run scripts/verify_installation.mojo
```

## Troubleshooting

### Build Failures

**Problem**: Compilation errors

**Solution**:

1. Check Mojo version: `mojo --version`
2. Clean build artifacts: `rm -rf build/`
3. Update dependencies: `pixi install`
4. Check for syntax errors in `.mojo` files

**Problem**: Out of memory during build

**Solution**:

1. Reduce parallelization: `mojo package shared -j 2`
2. Close other applications
3. Build in release mode only when needed

**Problem**: Missing dependencies

**Solution**:

1. Check `mojo.toml` dependencies
2. Verify Mojo standard library is available
3. Update Mojo to latest version

### Installation Issues

**Problem**: Package not found after installation

**Solution**:

1. Check installation path: `mojo list`
2. Verify MOJO_PATH: `echo $MOJO_PATH`
3. Reinstall: `mojo package shared --install`

## Best Practices

### Development Workflow

1. **Fast iteration**: Use development builds during development

   ```bash
   mojo package shared --install
   # Make changes
   mojo package shared --install  # Fast rebuild
   ```

2. **Test frequently**: Run tests after each significant change

   ```bash
   mojo test tests/shared/
   ```

3. **Release validation**: Build release mode before commits

   ```bash
   mojo package shared --release
   mojo test tests/shared/
   ```

### Version Management

1. **Update version** in `mojo.toml` for releases
2. **Tag releases** in git: `git tag v0.1.0`
3. **Create release package** with version in filename

## Additional Resources

- [Mojo Packages Guide](https://docs.modular.com/mojo/manual/packages/)
- [Mojo Build System](https://builds.modular.com/)
- [Installation Guide](INSTALL.md)
- [Contributing Guide](../CONTRIBUTING.md)
