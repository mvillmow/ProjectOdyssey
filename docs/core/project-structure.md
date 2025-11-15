# Project Structure

Understanding ML Odyssey's repository organization and architecture.

## Overview

ML Odyssey follows a hierarchical structure designed for clarity, reusability, and scalability. The repository
is organized around three main concepts: **shared components**, **paper implementations**, and **infrastructure**.

## Top-Level Directory Structure

```text
ml-odyssey/
├── shared/              # Shared library (reusable across papers)
├── papers/              # Individual paper implementations
├── benchmarks/          # Performance benchmarks
├── tests/               # Test suites
├── docs/                # Documentation
├── scripts/             # Automation scripts
├── .claude/             # Agent system configuration
├── agents/              # Team documentation
├── notes/               # Planning and architectural docs
└── configs/             # Configuration files
```

## Shared Library (`shared/`)

The shared library contains reusable components for all paper implementations:

```text
shared/
├── core/                # Core building blocks
│   ├── layers.mojo      # Neural network layers
│   ├── ops.mojo         # Basic operations
│   ├── types.mojo       # Common data types
│   └── utils.mojo       # Core utilities
├── training/            # Training infrastructure
│   ├── base.mojo        # Trainer base class
│   ├── callbacks.mojo   # Training callbacks
│   ├── schedulers.mojo  # Learning rate schedulers
│   └── stubs.mojo       # Placeholder implementations
├── data/                # Data processing
│   ├── datasets.mojo    # Dataset abstractions
│   ├── loaders.mojo     # Data loaders
│   ├── transforms.mojo  # Data transformations
│   └── samplers.mojo    # Sampling strategies
└── utils/               # Utility modules
    ├── config.mojo      # Configuration management
    ├── logging.mojo     # Logging utilities
    ├── io.mojo          # File I/O operations
    ├── profiling.mojo   # Performance profiling
    ├── random.mojo      # Random number generation
    └── visualization.mojo # Visualization tools
```

### Design Principles

1. **Modularity**: Each module has a clear, single responsibility
2. **Reusability**: Code is designed to be reused across papers
3. **Type Safety**: Leverages Mojo's type system for safety
4. **Performance**: Optimized with SIMD where applicable
5. **Testing**: Comprehensive test coverage

## Paper Implementations (`papers/`)

Each paper gets its own directory with a consistent structure:

```text
papers/
├── lenet5/              # LeNet-5 (Gradient-Based Learning Applied to Document Recognition)
│   ├── model.mojo       # Model implementation
│   ├── train.mojo       # Training script
│   ├── evaluate.mojo    # Evaluation script
│   ├── README.md        # Paper overview and implementation notes
│   └── tests/           # Paper-specific tests
├── alexnet/             # AlexNet (future)
└── vgg/                 # VGG (future)
```

### Paper Structure Guidelines

Each paper implementation follows this template:

- **model.mojo**: Network architecture matching the paper
- **train.mojo**: Training pipeline with hyperparameters from paper
- **evaluate.mojo**: Evaluation metrics and benchmarks
- **README.md**: Paper details, implementation notes, results
- **tests/**: Unit and integration tests

## Test Structure (`tests/`)

Tests mirror the source structure:

```text
tests/
├── shared/              # Tests for shared library
│   ├── core/            # Core component tests
│   ├── training/        # Training system tests
│   ├── data/            # Data processing tests
│   └── utils/           # Utility tests
├── papers/              # Tests for paper implementations
│   └── lenet5/          # LeNet-5 tests
├── foundation/          # Infrastructure tests
│   └── docs/            # Documentation tests
└── conftest.py          # Pytest configuration
```

### Testing Philosophy

- **Comprehensive Coverage**: All public APIs are tested
- **Test-Driven Development**: Tests written before implementation
- **Isolated Tests**: Each test is independent
- **Fast Execution**: Unit tests run in < 1 second each
- **Integration Tests**: End-to-end workflows validated

## Benchmarks (`benchmarks/`)

Performance benchmarking infrastructure:

```text
benchmarks/
├── scripts/             # Benchmark scripts
│   ├── run_benchmarks.mojo      # Execute benchmarks
│   └── compare_results.mojo     # Compare against baseline
├── baselines/           # Baseline results
└── results/             # Benchmark outputs
```

## Documentation (`docs/`)

User-facing documentation:

```text
docs/
├── index.md             # Documentation hub
├── getting-started/     # Tutorials for new users
│   ├── quickstart.md
│   ├── installation.md
│   └── first_model.md
├── core/                # Core concepts
│   ├── project-structure.md
│   ├── shared-library.md
│   └── mojo-patterns.md
├── advanced/            # Advanced topics
│   └── performance.md
└── dev/                 # Developer guides
    └── architecture.md
```

## Infrastructure

### Agent System (`.claude/`)

Hierarchical agent system for development:

```text
.claude/
├── agents/              # Agent configurations
│   ├── orchestrators/   # High-level coordination
│   ├── specialists/     # Domain experts
│   └── engineers/       # Implementation agents
└── prompts/             # Reusable prompts
```

### Planning Documentation (`notes/`)

Project planning and architectural decisions:

```text
notes/
├── plan/                # 4-level hierarchical plans
│   ├── 01-foundation/
│   ├── 02-shared-library/
│   ├── 03-tooling/
│   ├── 04-first-paper/
│   ├── 05-ci-cd/
│   └── 06-agentic-workflows/
├── issues/              # Issue-specific documentation
│   ├── 39/README.md     # Issue #39 implementation notes
│   └── 59/README.md     # Issue #59 implementation notes
└── review/              # Architectural decisions
    ├── adr/             # Architecture Decision Records
    └── designs/         # Design documents
```

### Configuration (`configs/`)

Configuration files for different environments:

```text
configs/
├── dev.toml             # Development configuration
├── test.toml            # Testing configuration
└── prod.toml            # Production configuration
```

## Module Dependencies

### Dependency Graph

```text
papers/lenet5
    └── shared/
        ├── core/
        ├── training/
        │   └── shared/core/
        ├── data/
        │   └── shared/core/
        └── utils/
```

### Import Conventions

```mojo
# Import from shared library
from shared.core import Layer, Sequential
from shared.training import Trainer, SGD
from shared.data import TensorDataset, BatchLoader

# Paper-specific imports
from papers.lenet5 import LeNet5Model
```

## File Naming Conventions

### Mojo Files

- **Modules**: lowercase with underscores (`neural_network.mojo`)
- **Structs**: PascalCase in module (`struct NeuralNetwork`)
- **Functions**: snake_case (`fn train_model()`)
- **Constants**: UPPER_SNAKE_CASE (`alias MAX_EPOCHS = 100`)

### Documentation

- **Guides**: lowercase with hyphens (`getting-started.md`)
- **READMEs**: UPPERCASE (`README.md`)
- **ADRs**: numbered (`ADR-001-language-selection.md`)

### Tests

- **Test files**: `test_*.py` or `test_*.mojo`
- **Test functions**: `test_feature_behavior()`
- **Fixtures**: `conftest.py` or `conftest.mojo`

## Build Artifacts

Temporary build files are gitignored:

```text
.gitignore entries:
*.o                      # Object files
*.so                     # Shared libraries
*.pyc                    # Python bytecode
__pycache__/             # Python cache directories
.pixi/                   # Pixi environment
.pytest_cache/           # Pytest cache
.vscode/                 # Editor settings (user-specific)
```

## Development Workflow

### Working Directory Layout

When developing:

```text
ml-odyssey/              # Main repository
└── worktrees/           # Git worktrees (temporary)
    ├── issue-39-impl-data/
    └── issue-59-impl-docs/
```

Git worktrees allow parallel development on multiple features.

## Key Locations Quick Reference

| What | Where |
|------|-------|
| Implement shared components | `shared/` |
| Implement a paper | `papers/<paper-name>/` |
| Write tests | `tests/` matching source structure |
| Add documentation | `docs/` following tier structure |
| Create benchmarks | `benchmarks/scripts/` |
| Automation scripts | `scripts/` |
| Configuration | `configs/` or `pixi.toml` |
| Planning docs | `notes/plan/` |
| Issue notes | `notes/issues/<number>/` |
| ADRs | `notes/review/adr/` |

## Next Steps

- **[Shared Library Guide](shared-library.md)** - Deep dive into shared components
- **[Paper Implementation Guide](paper-implementation.md)** - Implementing a paper
- **[Testing Strategy](testing-strategy.md)** - Testing approach
- **[Development Guide](../dev/architecture.md)** - Contributing to the project

## Related Documentation

- [Repository README](https://github.com/mvillmow/ml-odyssey/blob/main/README.md) - Project overview
- Contributing Guide (`CONTRIBUTING.md`) - Contribution guidelines
- [Agent System](agent-system.md) - Development workflow with agents
