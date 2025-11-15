# Training Module Package Build Instructions

## Overview

This document provides instructions for building the distributable training module package.

## Prerequisites

- Mojo compiler installed and available in PATH
- Repository checked out (any location works)
- Training module source code in `shared/training/`

## Build Steps

### 1. Create Distribution Directory

```bash
# Navigate to repository root
cd /path/to/ml-odyssey
mkdir -p dist/
```

### 2. Build the Package

```bash
mojo package shared/training -o dist/training-0.1.0.mojopkg
```

This command will:

- Compile all Mojo source files in `shared/training/`
- Package them into a binary `.mojopkg` file
- Output to `dist/training-0.1.0.mojopkg`

### 3. Verify Package Creation

```bash
ls -lh dist/training-0.1.0.mojopkg
```

You should see a file with the package binary.

### 4. Test Installation

Make the verification script executable and run it:

```bash
chmod +x scripts/install_verify_training.sh
./scripts/install_verify_training.sh
```

This will:

- Create a temporary test environment
- Install the package
- Test all exports work correctly
- Clean up the test environment

## Automated Build

Alternatively, use the build script:

```bash
chmod +x scripts/build_training_package.sh
./scripts/build_training_package.sh
```

## Expected Output

After successful build:

```text
dist/
└── training-0.1.0.mojopkg     # Binary package (size varies)
```

## Verification Tests

The verification script tests:

1. Callback system imports (Callback, CallbackSignal, CONTINUE, STOP)
2. TrainingState import
3. LRScheduler interface import
4. Scheduler implementations (StepLR, CosineAnnealingLR, WarmupLR)
5. Callback implementations (EarlyStopping, ModelCheckpoint, LoggingCallback)
6. Utility functions (is_valid_loss, clip_gradients)

## Troubleshooting

### Mojo not found

```bash
which mojo
mojo --version
# Should be v0.25.7 or later
```

### Build fails

```bash
# Check source files compile
mojo shared/training/__init__.mojo
```

### Permission denied on dist/

```bash
mkdir -p dist
chmod 755 dist
```

### Package Build Fails

If `mojo package` fails:

1. Verify all Mojo source files compile individually:

```bash
mojo build shared/training/__init__.mojo
mojo build shared/training/base.mojo
mojo build shared/training/schedulers.mojo
mojo build shared/training/callbacks.mojo
```

1. Check for syntax errors in source files
1. Ensure all imports in `__init__.mojo` are correct

### Installation Test Fails

If the verification script fails:

1. Check the package file exists: `ls -lh dist/training-0.1.0.mojopkg`
1. Try manual installation: `mojo install dist/training-0.1.0.mojopkg`
1. Test imports manually: `mojo run -c "from training import Callback"`

## Package Contents

The package includes:

- `__init__.mojo` - Main package initialization and exports
- `base.mojo` - Base interfaces (Callback, TrainingState, LRScheduler)
- `schedulers.mojo` - Learning rate scheduler implementations
- `callbacks.mojo` - Callback implementations
- `stubs.mojo` - Placeholder implementations
- Subdirectory packages: optimizers/, schedulers/, callbacks/, metrics/, loops/

## Version Information

- Package name: `training`
- Version: `0.1.0`
- Full package name: `training-0.1.0.mojopkg`

## Next Steps

After successful build and verification:

1. Commit the build scripts (not the .mojopkg file)
1. Update `.gitignore` to exclude `dist/*.mojopkg`
1. Create PR linking to Issue #35
1. Document installation instructions in main README
