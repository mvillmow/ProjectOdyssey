# Installation Guide - ML Odyssey Shared Library

This guide covers installing and setting up the ML Odyssey shared library for use in paper implementations.

## Prerequisites

### Required

- **Mojo 24.5 or later** - Download from [Modular](https://www.modular.com/mojo)
- **Git** - For cloning the repository

### Optional

- **Python 3.10+** - For development tooling and scripts
- **Pixi** - For managing development environment (recommended)

## Installation Methods

### Option 1: Development Install (Recommended for Contributors)

This method is best for contributors who want to modify the shared library code.

```bash
# Clone the repository
git clone https://github.com/mvillmow/ml-odyssey.git
cd ml-odyssey

# Install in development mode (creates symlinks)
mojo package shared --install

# Verify installation
mojo run scripts/verify_installation.mojo
```

**Benefits**:

- Changes to shared library immediately available
- No need to reinstall after modifications
- Easy to contribute changes back

**Use When**:

- Contributing to the shared library
- Developing new paper implementations
- Testing experimental features

### Option 2: Package Install

Install from a pre-built package (for users who won't modify the shared library).

```bash
# Download the package (from release or build it)
mojo package shared -o ml_odyssey_shared.mojopkg

# Install the package
mojo install ml_odyssey_shared.mojopkg

# Verify installation
mojo run scripts/verify_installation.mojo
```

**Benefits**:

- Fast installation
- No repository needed
- Stable, tested version

**Use When**:

- Using the library without modifications
- Deploying paper implementations
- Production use

### Option 3: Path-Based Install

Add the shared library to your project via path dependency.

**File**: `your_project/mojo.toml` or `mojoproject.toml`

```toml
[package]
name = "your-paper-implementation"
version = "0.1.0"

[dependencies]
ml-odyssey-shared = { path = "../ml-odyssey/shared" }
```

Then use in your code:

```mojo
from shared import Linear, SGD, Tensor
```

**Benefits**:

- Easy setup for related projects
- Automatic updates when shared library changes
- Good for monorepo setups

**Use When**:

- Working in a monorepo with ml-odyssey
- Local development across multiple projects
- Quick prototyping

## Build from Source

### Building the Package

```bash
# Build development package (fast, no optimizations)
mojo package shared -o build/ml_odyssey_shared.mojopkg

# Build release package (optimized for performance)
mojo package shared --release -o dist/ml_odyssey_shared-0.1.0.mojopkg

# Build with debug symbols
mojo package shared --debug -o build/ml_odyssey_shared_debug.mojopkg
```

### Build Options

- `--release` - Enable optimizations, disable debug info (use for production)
- `--debug` - Include debug symbols (use for development/debugging)
- `-o OUTPUT` - Specify output path for package file
- `--install` - Build and install in one step

## Verification

### Quick Verification

Run the verification script to check if installation succeeded:

```bash
mojo run scripts/verify_installation.mojo
```

Expected output:

```text
✓ Core imports successful
✓ Core functionality working
✓ Training functionality working
✓ Utils functionality working

✅ Shared library installation verified!
```

### Manual Verification

Test imports manually:

```mojo
from shared import Linear, SGD, DataLoader, Logger

# If this imports without error, installation successful
print("Shared library installed successfully!")
```

### Detailed Verification

Run the full test suite:

```bash
# Run import validation tests
mojo test tests/shared/test_imports.mojo

# Run integration tests
mojo test tests/shared/integration/

# Run all tests
mojo test tests/shared/
```

## Usage

### Basic Usage

After installation, import components in your Mojo code:

```mojo
# Import commonly used items from root
from shared import Linear, ReLU, Sequential, SGD, Tensor

# Build model
var model = Sequential([
    Linear(784, 256),
    ReLU(),
    Linear(256, 10),
])

# Create optimizer
var optimizer = SGD(learning_rate=0.01, momentum=0.9)
```

### Advanced Usage

Import from specific modules for less common items:

```mojo
# Import from specific submodules
from shared.core.layers import MaxPool2D, Dropout, BatchNorm2D
from shared.training.schedulers import CosineAnnealingLR
from shared.training.metrics import Precision, Recall
from shared.data.transforms import RandomCrop, RandomHorizontalFlip
```

See [EXAMPLES.md](EXAMPLES.md) for complete usage examples.

## Environment Setup

### Using Pixi (Recommended)

If you're using Pixi for environment management:

```bash
# Install dependencies
pixi install

# Run commands in Pixi environment
pixi run mojo --version
pixi run mojo package shared --install
```

### Manual Setup

Without Pixi:

```bash
# Ensure Mojo is in your PATH
export PATH="/path/to/mojo/bin:$PATH"

# Verify Mojo installation
mojo --version

# Install shared library
mojo package shared --install
```

## Troubleshooting

### Import Errors

**Problem**: `Error: Cannot find module 'shared'`

**Solution**:

1. Verify installation: `mojo list` should show `ml-odyssey-shared`
2. Reinstall: `mojo package shared --install`
3. Check MOJO_PATH: `echo $MOJO_PATH`

**Problem**: `Error: Module 'shared' found but component 'Linear' not found`

**Solution**:

- The implementation is not yet complete for that component
- Check Issue #49 for implementation status
- Use placeholder or stub for development

### Build Errors

**Problem**: Build fails with compilation errors

**Solution**:

1. Check Mojo version: `mojo --version` (requires 24.5+)
2. Clean build artifacts: `rm -rf build/`
3. Update dependencies in `pixi.toml`
4. Report issue if problem persists

### Version Conflicts

**Problem**: Multiple versions of shared library installed

**Solution**:

```bash
# List installed packages
mojo list

# Uninstall old version
mojo uninstall ml-odyssey-shared

# Reinstall correct version
mojo package shared --install
```

## Uninstallation

### Remove Package

```bash
# Uninstall the package
mojo uninstall ml-odyssey-shared

# Verify removal
mojo list | grep ml-odyssey
```

### Clean Build Artifacts

```bash
# Remove build artifacts
rm -rf build/ dist/

# Remove cached files (if any)
rm -rf .mojo_cache/
```

## Platform Support

### Supported Platforms

- Linux (x86_64, ARM64)
- macOS (Intel, Apple Silicon)
- Windows (WSL2 required)

### Platform-Specific Notes

**Linux**:

- Works out of the box on most distributions
- Ensure glibc 2.31+ for optimal performance

**macOS**:

- Works on Intel and Apple Silicon
- May require Xcode Command Line Tools

**Windows**:

- Use WSL2 (Windows Subsystem for Linux)
- Native Windows support planned

## Development Setup

### For Contributors

Set up a complete development environment:

```bash
# Clone repository
git clone https://github.com/mvillmow/ml-odyssey.git
cd ml-odyssey

# Install development dependencies
pixi install

# Install pre-commit hooks
pre-commit install

# Install shared library in development mode
mojo package shared --install

# Run tests to verify setup
mojo test tests/shared/

# Format code
mojo format shared/
```

### IDE Setup

**VS Code**:

1. Install Mojo extension
2. Set Mojo path in settings
3. Configure formatter: `mojo format`

**Other Editors**:

- Configure LSP for Mojo (if available)
- Set formatter to `mojo format`
- Configure syntax highlighting

## Next Steps

After installation:

1. **Read Examples**: See [EXAMPLES.md](EXAMPLES.md) for usage patterns
2. **Check API Docs**: Review auto-generated API documentation
3. **Run Tests**: Verify everything works with test suite
4. **Start Building**: Begin implementing your paper

## Support

### Getting Help

- **Documentation**: Check [README.md](README.md) for overview
- **Examples**: See [EXAMPLES.md](EXAMPLES.md) for code samples
- **Issues**: Report bugs on [GitHub Issues](https://github.com/mvillmow/ml-odyssey/issues)
- **Discussions**: Ask questions in GitHub Discussions

### Reporting Issues

When reporting installation issues:

1. Include Mojo version: `mojo --version`
2. Include OS and architecture: `uname -a`
3. Provide full error message
4. Describe steps to reproduce

## Additional Resources

- [Mojo Documentation](https://docs.modular.com/mojo/)
- [Mojo Packages Guide](https://docs.modular.com/mojo/manual/packages/)
- [Project README](../README.md)
- [Contributing Guide](../CONTRIBUTING.md)

## License

See the main repository [LICENSE](../LICENSE) file.
