# ML Odyssey

A Mojo-based AI research platform for reproducing classic deep learning papers.

[![Mojo](https://img.shields.io/badge/Mojo-0.26+-orange.svg)](https://www.modular.com/mojo)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Status**: Active Research Project | **Current Focus**: Foundation Models

## Overview

ML Odyssey is a from-scratch implementation of classic deep learning papers in Mojo, designed to demonstrate modern
systems programming for AI. The project provides pure Mojo implementations of neural network architectures (LeNet-5,
AlexNet, VGG-16, ResNet-18, etc.), manual gradient computation for educational clarity, and a production-quality
shared library with tensor operations, optimizers, and training loops.

This is a **research and education platform**, not a production ML framework. Think "PyTorch from scratch" rather
than "use this for production."

## Quick Start

### Prerequisites

- **Mojo 0.26+** ([Download from Modular](https://www.modular.com/mojo))
- **Pixi** (optional but recommended for dependency management)
- **Just** ([Install from just.systems](https://just.systems/))
- **Git** for cloning the repository

### Installation

```bash
# Clone the repository
git clone https://github.com/mvillmow/ml-odyssey.git
cd ml-odyssey

# Install dependencies (if using Pixi)
pixi install

# Build the shared library
just build
```

### Train Your First Model: LeNet-5 on EMNIST

LeNet-5 is the **only fully functional model** with complete forward/backward passes and achieves **~81% accuracy** on EMNIST.

```bash
# 1. Download the dataset
python scripts/download_emnist.py --split balanced

# 2. Train LeNet-5 (simple command, 10 epochs)
just train

# Or with full control:
pixi run mojo run -I . examples/lenet-emnist/train.mojo \
  --epochs 10 \
  --batch-size 32 \
  --lr 0.001

# 3. Run inference on test set
just infer lenet lenet5_weights
```

**Expected Results:**

- Training: ~81% accuracy on EMNIST Balanced (47 classes)
- Training time: ~30 minutes on modern CPU
- Model size: ~61K parameters (~244KB)

## Project Status

### What Works ✅

- **LeNet-5 + EMNIST**: Fully functional training pipeline with 81% accuracy
- **Shared Library**:
  - Core ops: conv2d, maxpool2d, linear, relu, sigmoid, tanh (forward + backward)
  - Training: SGD optimizer, cross-entropy loss, data loaders, metrics
  - Utils: Argument parsing, logging, weight serialization
- **Build System**: Comprehensive justfile with 40+ recipes
- **Documentation**: BUILD.md (489 lines), INSTALL.md (615 lines)

### In Development 🚧

- **7 models with forward-only passes**: AlexNet, VGG-16, ResNet-18, DenseNet-121, GoogLeNet, MobileNetV1
  - Complete forward passes implemented
  - Missing: ~2000-3000 lines of backward pass gradients per model
  - See [Issue #2576](https://github.com/mvillmow/ml-odyssey/issues/2576) for details
- **Optimizer Enhancements**: SGD with momentum, Adam, learning rate schedulers
- **Autograd System**: Replacing manual gradient computation

### Known Issues ⚠️

- **Broken Examples**: Getting-started examples use non-existent Sequential/Layer API
  - See [Issue #2575](https://github.com/mvillmow/ml-odyssey/issues/2575)
- **Dataset Coverage**: Only EMNIST download script exists, CIFAR-10 script needed
  - See [Issue #2577](https://github.com/mvillmow/ml-odyssey/issues/2577)

## Model Zoo

| Model | Status | Dataset | Forward | Backward |
|-------|--------|---------|---------|----------|
| **LeNet-5** | ✅ **READY** | EMNIST | ✅ Complete | ✅ Complete |
| AlexNet | Forward only | CIFAR-10 | ✅ Complete | ❌ Missing |
| VGG-16 | Forward only | CIFAR-10 | ✅ Complete | ❌ Missing |
| ResNet-18 | Forward only | CIFAR-10 | ✅ Complete | ❌ Missing |
| DenseNet-121 | Forward only | CIFAR-10 | ✅ Complete | ❌ Missing |
| GoogLeNet | Forward only | CIFAR-10 | ✅ Complete | ❌ Missing |
| MobileNetV1 | Forward only | CIFAR-10 | ✅ Complete | ❌ Missing |

**Note**: Each incomplete model needs ~2000-3000 lines of gradient computation code. See individual model directories
for `GAP_ANALYSIS.md` with detailed backward pass requirements.

## Build & Development

### Quick Reference

```bash
# Show all available commands
just --list

# Build commands
just build              # Build shared library (debug mode)
just build-release      # Build with optimizations
just ci-validate        # Full CI validation (build + test)

# Testing
just test               # Run all tests (Mojo + Python)
just test-mojo          # Mojo tests only

# Linting & Formatting
just lint               # Run all linters
just format             # Format all files

# Training
just train              # Train LeNet-5 with defaults
just train lenet fp16 20  # Train with FP16, 20 epochs
```

### Detailed Documentation

For comprehensive build and installation instructions, see:

- **[shared/BUILD.md](shared/BUILD.md)** - Complete build system guide (489 lines)
- **[shared/INSTALL.md](shared/INSTALL.md)** - Installation methods and troubleshooting (615 lines)

### Development Workflow

```bash
# 1. Set up environment
pixi install
pre-commit install

# 2. Make changes to shared library
# Edit files in shared/core/, shared/training/, etc.

# 3. Build and test
just build
just test-mojo

# 4. Run pre-commit checks
just pre-commit

# 5. Create PR
git checkout -b feature/my-feature
git add .
git commit -m "feat: add new feature"
gh pr create --title "Add feature" --body "Closes #<issue-number>"
```

## Documentation

### User Documentation

- **[Quick Start](#quick-start)** - Get running in 5 minutes
- **[shared/INSTALL.md](shared/INSTALL.md)** - Installation guide (615 lines)
- **[shared/BUILD.md](shared/BUILD.md)** - Build system reference (489 lines)

### Developer Documentation

- **[CONTRIBUTING.md](CONTRIBUTING.md)** - How to contribute
- **[.claude/shared/mojo-guidelines.md](.claude/shared/mojo-guidelines.md)** - Mojo syntax patterns
- **[.claude/shared/mojo-anti-patterns.md](.claude/shared/mojo-anti-patterns.md)** - 64+ failure patterns to avoid
- **[agents/hierarchy.md](agents/hierarchy.md)** - AI agent development system

## Key Features

- **Zero Python dependencies** for core ML operations (pure Mojo)
- **Type-safe tensor operations** with compile-time shape checking
- **SIMD optimization** for performance-critical operations
- **Functional API design** (ExTensor, functional ops like conv2d, linear, relu)
- **Manual backpropagation** for educational transparency
- **Comprehensive build system** via Just (just build, just test, just train)

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Priority Areas:**

- Complete backward passes for AlexNet, VGG-16, ResNet-18, etc. ([Issue #2576](https://github.com/mvillmow/ml-odyssey/issues/2576))
- Fix broken getting-started examples ([Issue #2575](https://github.com/mvillmow/ml-odyssey/issues/2575))
- Add CIFAR-10 dataset download script ([Issue #2577](https://github.com/mvillmow/ml-odyssey/issues/2577))
- Implement autograd system
- Add data augmentation module

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

Built with [Mojo](https://www.modular.com/mojo) from Modular. Inspired by classic papers from LeCun, Krizhevsky,
Simonyan, He, and others.
