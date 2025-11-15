# ML Odyssey Installation Guide

## Overview

ML Odyssey provides distributable Mojo packages for machine learning research. This guide covers installation of the
training module and other shared library components.

## Prerequisites

- Mojo compiler installed ([Modular Installation Guide](https://docs.modular.com/mojo/manual/get-started/))
- Mojo available in PATH
- Compatible operating system (Linux, macOS)

## Quick Start

### Installing Pre-built Packages

If you have access to pre-built `.mojopkg` files:

```bash
# Install training module
mojo install dist/training-0.1.0.mojopkg

# Verify installation
mojo run -c "from training import Callback, LRScheduler; print('Training module ready!')"
```

### Building from Source

To build packages from source:

```bash
# Build training package
./scripts/build_training_package.sh

# Test installation
./scripts/install_verify_training.sh
```

## Package Modules

### Training Module

**Package**: `training-0.1.0.mojopkg`

**Description**: Training utilities including optimizers, schedulers, callbacks, and training loops.

**Installation**:

```bash
mojo install dist/training-0.1.0.mojopkg
```

**Verification**:

```bash
./scripts/install_verify_training.sh
```

**Usage Example**:

```mojo
from training import (
    Callback,
    CallbackSignal,
    CONTINUE,
    STOP,
    TrainingState,
    LRScheduler,
    StepLR,
    CosineAnnealingLR,
    WarmupLR,
    EarlyStopping,
    ModelCheckpoint,
    LoggingCallback,
    is_valid_loss,
    clip_gradients,
)

# Create learning rate scheduler
var scheduler = StepLR(initial_lr=0.1, step_size=10, gamma=0.1)

# Create early stopping callback
var early_stop = EarlyStopping(patience=5, min_delta=0.001)
```

## Building Packages

### Training Module

```bash
# Navigate to repository root
cd /path/to/ml-odyssey

# Create distribution directory
mkdir -p dist/

# Build package
mojo package shared/training -o dist/training-0.1.0.mojopkg

# Verify build
ls -lh dist/training-0.1.0.mojopkg
```

### Automated Build

Use the provided build scripts:

```bash
# Make script executable
chmod +x scripts/build_training_package.sh

# Build package
./scripts/build_training_package.sh
```

## Verification

### Automated Testing

Each module has a verification script:

```bash
# Training module
chmod +x scripts/install_verify_training.sh
./scripts/install_verify_training.sh
```

### Manual Testing

Test installation manually:

```bash
# Create test environment
TEMP_DIR=$(mktemp -d)
cd "$TEMP_DIR"

# Install package
mojo install /path/to/dist/training-0.1.0.mojopkg

# Test imports
mojo run -c "from training import Callback; print('Success!')"
mojo run -c "from training import LRScheduler; print('Success!')"
mojo run -c "from training import StepLR; print('Success!')"

# Cleanup
cd -
rm -rf "$TEMP_DIR"
```

## Troubleshooting

### Package Build Fails

**Issue**: `mojo package` command fails

**Solutions**:

1. Verify Mojo compiler is installed:

```bash
mojo --version
```

1. Check source files compile individually:

```bash
mojo build shared/training/__init__.mojo
mojo build shared/training/base.mojo
```

1. Review error messages for syntax errors or import issues

### Installation Fails

**Issue**: `mojo install` command fails

**Solutions**:

1. Verify package file exists:

```bash
ls -lh dist/training-0.1.0.mojopkg
```

1. Check file permissions:

```bash
chmod 644 dist/training-0.1.0.mojopkg
```

1. Try installing to custom location:

```bash
mojo install --prefix /custom/path dist/training-0.1.0.mojopkg
```

### Import Errors After Installation

**Issue**: Imports fail after successful installation

**Solutions**:

1. Verify package is in Mojo's module search path:

```bash
mojo run -c "import sys; print(sys.path)"
```

1. Try explicit path:

```bash
mojo run -M /path/to/installed/packages -c "from training import Callback"
```

1. Reinstall package:

```bash
mojo uninstall training
mojo install dist/training-0.1.0.mojopkg
```

### Verification Script Fails

**Issue**: `install_verify_training.sh` fails

**Solutions**:

1. Check package was built:

```bash
ls -lh dist/training-0.1.0.mojopkg
```

1. Run script with verbose output:

```bash
bash -x scripts/install_verify_training.sh
```

1. Test specific imports manually:

```bash
mojo run -c "from training import Callback"
```

## Package Exports (16 total)

The training package includes:

### Core Components (3)

- `TrainingState` - Training state management
- `Callback` - Base callback interface
- `CallbackSignal` - Callback signal types

### Callback Signals (2)

- `CONTINUE` - Continue training signal
- `STOP` - Stop training signal

### Learning Rate Schedulers (4)

- `LRScheduler` - Base scheduler interface
- `StepLR` - Step learning rate scheduler
- `CosineAnnealingLR` - Cosine annealing scheduler
- `WarmupLR` - Warmup learning rate scheduler

### Training Callbacks (3)

- `EarlyStopping` - Early stopping callback
- `ModelCheckpoint` - Model checkpointing callback
- `LoggingCallback` - Training logging callback

### Utilities (2)

- `is_valid_loss` - Loss validation utility
- `clip_gradients` - Gradient clipping utility

### Installation Verification

After installation, verify all exports work:

```bash
./scripts/install_verify_training.sh
```

Or test manually:

```mojo
from training import Optimizer, LRScheduler, EarlyStopping
print("Training module installed successfully!")
```

## Package Contents

### Training Module (training-0.1.0.mojopkg)

**Total Exports**: 16 (see section above for details)

**Dependencies**: Mojo standard library

**Version**: 0.1.0

**License**: BSD-3-Clause

## Uninstallation

To remove installed packages:

```bash
# Uninstall training module
mojo uninstall training

# Verify removal
mojo run -c "from training import Callback" 2>&1 | grep -q "ModuleNotFoundError" && echo "Uninstalled successfully"
```

## Development Installation

For development, use the source directly without installing:

```bash
# Set MOJO_PATH to include source directory
export MOJO_PATH=/path/to/ml-odyssey/shared:$MOJO_PATH

# Run code using source directly
mojo run your_script.mojo
```

## Version Information

| Package  | Version | Release Date | Status        |
|----------|---------|--------------|---------------|
| training | 0.1.0   | 2025-11-14   | In Development|

## Support

For issues or questions:

- Report bugs via GitHub Issues
- Check documentation in `shared/training/README.md`
- Review build documentation in `BUILD_PACKAGE.md`

## Next Steps

After installation:

1. Review module documentation in `shared/training/README.md`
1. Explore examples in `examples/` directory
1. Check test files in `tests/shared/training/` for usage patterns
1. Read the comprehensive guides in `docs/`

## Building Other Modules

As additional modules are packaged, they will follow the same pattern:

```bash
# Data module (future)
mojo package shared/data -o dist/data-0.1.0.mojopkg
./scripts/install_verify_data.sh

# Utils module (future)
mojo package shared/utils -o dist/utils-0.1.0.mojopkg
./scripts/install_verify_utils.sh
```

Check individual module README files for specific installation instructions.
