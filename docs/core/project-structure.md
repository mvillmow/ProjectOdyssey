# Project Structure

Quick navigation guide to the ML Odyssey repository structure. For comprehensive details, see
[repository-structure.md](../getting-started/repository-structure.md).

## Overview

ML Odyssey organizes code and documentation by **purpose**, making it easy to find what you need. The repository
separates ML implementations, reusable components, documentation, tools, and configuration into logical directories.

## Key Directories

**papers/** - ML research paper implementations

- Each paper gets its own directory with model, training, and test files
- Use `tools/paper-scaffold/` to create new papers

**shared/** - Reusable ML components

- Core layers, activations, loss functions
- Training utilities (optimizers, schedulers)
- Data loading and utilities
- Import these into papers/ for common functionality

**tests/** - Test suite

- Mirrored structure matching source code
- Unit tests for shared components
- Paper-specific tests
- Run all: `mojo test tests/`

**docs/** - User documentation

- `getting-started/` - Setup and onboarding
- `core/` - Core concepts
- `advanced/` - Advanced topics
- `api/` - API reference

**agents/** - AI agent system

- Agent hierarchy and documentation
- Agent configuration templates
- Coordination patterns and delegation rules

**tools/** - Development utilities

- `paper-scaffold/` - Generate paper structure
- `benchmarking/` - Performance measurement
- `codegen/` - Code generation utilities
- Testing and validation tools

**benchmarks/** - Performance tracking

- `scripts/` - Benchmark execution
- `baselines/` - Baseline results
- `results/` - Timestamped benchmark runs

**configs/** - Experiment configuration

- `defaults/` - Base settings
- `papers/` - Paper-specific configs
- `experiments/` - Experiment variations

## Quick Navigation

**"I want to..."**

| Goal | Location | Command |
| --- | --- | --- || Start a new paper | papers/ | `python tools/paper-scaffold/scaffold.py --paper {name}` |
| Add a reusable component | shared/ | Edit `shared/core/`, `shared/training/`, etc. |
| Find how something works | docs/core/ | Browse core concepts |
| Run tests | tests/ | `mojo test tests/` |
| Measure performance | benchmarks/ | `mojo benchmarks/scripts/run_benchmarks.mojo` |
| Configure an experiment | configs/ | Copy template, edit `configs/experiments/{paper}/` |
| Understand architecture | notes/review/ | Read architectural decision records |

## Related Resources

- **[Repository Structure Guide](../getting-started/repository-structure.md)** - Comprehensive directory reference and
- **[Repository Structure Guide](../getting-started/repository-structure.md)** - Comprehensive directory reference and

  workflows

- **STRUCTURE.md** (in repo root) - Complete directory tree with descriptions
- **[Installation Guide](../getting-started/installation.md)** - Environment setup
- **CONTRIBUTING.md** (in repo root) - Development workflow and standards
- **Agent System** (`agents/` directory) - AI agent hierarchy and delegation
- **[Quick Start: New Paper](../integration/quick-start-new-paper.md)** - Creating new papers
