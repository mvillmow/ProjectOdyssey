# Issue #82: [Plan] Directory Structure - Design and Documentation

## Objective

Establish and document the complete directory structure for the Mojo AI Research Repository, separating individual paper implementations from reusable code components.

## Phase: Plan (Current Phase)

This document contains the comprehensive planning for the directory structure design, including specifications, architecture documentation, and API contracts.

## Deliverables

### Primary Deliverables
1. Complete directory structure specification
2. README templates for each major directory
3. Template structure for new paper implementations
4. Directory purpose documentation
5. Architectural decisions for structure organization

## Directory Structure Specification

### Top-Level Directory Organization

```text
ml-odyssey/
├── papers/               # Individual research paper implementations
├── shared/               # Reusable components and libraries
├── benchmarks/           # Performance benchmarking tools and results
├── docs/                 # User-facing documentation
├── agents/               # Claude agent configurations and guides
├── tools/                # Development and build tools
├── configs/              # Configuration files and templates
├── scripts/              # Automation and utility scripts
├── tests/                # Test suites and testing infrastructure
├── examples/             # Usage examples and tutorials
├── notes/                # Planning and internal documentation
└── .github/              # GitHub-specific configurations
```

## Detailed Directory Specifications

### 1. papers/ - Research Paper Implementations

**Purpose**: Contains isolated, self-contained implementations of individual research papers.

**Structure**:
```text
papers/
├── README.md              # Papers directory overview and index
├── _template/             # Template for new paper implementations
│   ├── README.md         # Template documentation
│   ├── model.mojo        # Model implementation template
│   ├── training.mojo     # Training script template
│   ├── evaluation.mojo   # Evaluation script template
│   ├── paper_info.yaml   # Paper metadata template
│   └── requirements.txt  # Dependencies template
├── lenet5/               # Example: LeNet-5 implementation
│   ├── README.md         # Paper-specific documentation
│   ├── model.mojo        # LeNet-5 model implementation
│   ├── training.mojo     # Training script
│   ├── evaluation.mojo   # Evaluation script
│   ├── paper_info.yaml   # Paper metadata
│   └── results/          # Training results and checkpoints
└── [future_papers]/      # Future paper implementations follow template
```

**README Template** (`papers/README.md`):
```markdown
# Research Papers Implementation

This directory contains implementations of classic and modern AI research papers using Mojo.

## Implemented Papers

| Paper | Year | Directory | Status | Performance |
|-------|------|-----------|--------|-------------|
| LeNet-5 | 1998 | [lenet5/](lenet5/) | Complete | 98.5% accuracy |

## Adding a New Paper

1. Copy the `_template/` directory
2. Rename to paper identifier (e.g., `alexnet/`)
3. Update paper_info.yaml with paper metadata
4. Implement model in model.mojo
5. Add training and evaluation scripts
6. Document results in README.md

## Guidelines

- Each paper is self-contained and independent
- Use shared/ components where possible
- Follow Mojo best practices for performance
- Include comprehensive documentation
```

### 2. shared/ - Reusable Components Library

**Purpose**: Core library of reusable components used across multiple paper implementations.

**Structure**:
```text
shared/
├── README.md           # Shared library overview
├── BUILD.md           # Build instructions
├── INSTALL.md         # Installation guide
├── core/              # Core ML operations
│   ├── README.md
│   ├── tensor.mojo    # Tensor operations
│   ├── autodiff.mojo  # Automatic differentiation
│   └── kernels.mojo   # SIMD kernels
├── layers/            # Neural network layers
│   ├── README.md
│   ├── conv2d.mojo    # Convolutional layers
│   ├── dense.mojo     # Fully connected layers
│   ├── pooling.mojo   # Pooling layers
│   └── activation.mojo # Activation functions
├── optimizers/        # Optimization algorithms
│   ├── README.md
│   ├── sgd.mojo       # Stochastic Gradient Descent
│   ├── adam.mojo      # Adam optimizer
│   └── base.mojo      # Optimizer base class
├── training/          # Training utilities
│   ├── README.md
│   ├── trainer.mojo   # Training loop
│   ├── callbacks.mojo # Training callbacks
│   └── metrics.mojo   # Performance metrics
├── data/              # Data loading and processing
│   ├── README.md
│   ├── dataset.mojo   # Dataset base class
│   ├── dataloader.mojo # Data loading utilities
│   └── transforms.mojo # Data transformations
└── utils/             # General utilities
    ├── README.md
    ├── io.mojo        # I/O operations
    ├── logging.mojo   # Logging utilities
    └── visualization.mojo # Visualization helpers
```

**README Template** (`shared/README.md`):
```markdown
# Shared Library

Core reusable components for ML research implementations in Mojo.

## Components

- **core/**: Fundamental tensor operations and autodiff
- **layers/**: Neural network layer implementations
- **optimizers/**: Optimization algorithms
- **training/**: Training loops and utilities
- **data/**: Data loading and processing
- **utils/**: General utilities

## Installation

See [INSTALL.md](INSTALL.md) for installation instructions.

## Usage

```mojo
from shared.layers import Conv2D, Dense
from shared.optimizers import Adam
from shared.training import Trainer
```

## API Documentation

Comprehensive API documentation available in each component's README.
```

### 3. benchmarks/ - Performance Benchmarking

**Purpose**: Performance benchmarking tools and results for comparing implementations.

**Structure**:
```text
benchmarks/
├── README.md             # Benchmarking overview
├── BUILD.md             # Build instructions for benchmarks
├── INSTALL.md           # Installation guide
├── core/                # Core benchmarking infrastructure
│   ├── README.md
│   ├── timer.mojo       # Timing utilities
│   ├── profiler.mojo    # Profiling tools
│   └── reporter.mojo    # Results reporting
├── suites/              # Benchmark suites
│   ├── README.md
│   ├── inference/       # Inference benchmarks
│   ├── training/        # Training benchmarks
│   └── memory/          # Memory usage benchmarks
├── results/             # Benchmark results
│   ├── README.md
│   └── [timestamp]/     # Timestamped results
└── scripts/             # Benchmark automation
    ├── run_all.py       # Run all benchmarks
    └── compare.py       # Compare results
```

### 4. docs/ - User Documentation

**Purpose**: User-facing documentation, tutorials, and guides.

**Structure**:
```text
docs/
├── README.md           # Documentation index
├── getting-started/    # Getting started guides
│   ├── README.md
│   ├── installation.md
│   └── first-model.md
├── tutorials/          # Step-by-step tutorials
│   ├── README.md
│   └── implementing-papers.md
├── api/                # API reference
│   ├── README.md
│   └── [auto-generated]/
└── contributing/       # Contribution guides
    ├── README.md
    └── code-style.md
```

### 5. agents/ - Claude Agent Configurations

**Purpose**: Agent configurations, hierarchy documentation, and automation guides.

**Structure**:
```text
agents/
├── README.md           # Agent system overview
├── hierarchy.md        # Visual hierarchy diagram
├── delegation-rules.md # Coordination patterns
├── guides/             # Implementation guides
│   ├── github-review-comments.md
│   └── verification-checklist.md
├── templates/          # Agent configuration templates
│   └── agent-template.md
└── skills/             # Agent skill definitions
    ├── tier-1/         # Basic skills
    ├── tier-2/         # Advanced skills
    └── tier-3/         # Expert skills
```

### 6. tools/ - Development Tools

**Purpose**: Development, build, and deployment tools.

**Structure**:
```text
tools/
├── README.md           # Tools overview
├── build/              # Build tools
│   ├── README.md
│   ├── package.py      # Packaging scripts
│   └── release.py      # Release automation
├── testing/            # Testing tools
│   ├── README.md
│   ├── runner.py       # Test runner
│   └── coverage.py     # Coverage tools
└── development/        # Development utilities
    ├── README.md
    ├── formatter.py    # Code formatting
    └── linter.py       # Code linting
```

### 7. configs/ - Configuration Files

**Purpose**: Centralized configuration files and templates.

**Structure**:
```text
configs/
├── README.md           # Configuration overview
├── mojo/               # Mojo-specific configs
│   ├── mojoproject.toml.template
│   └── compile_flags.txt
├── python/             # Python configs
│   ├── pyproject.toml
│   └── requirements.txt
├── ci/                 # CI/CD configurations
│   ├── pre-commit.yaml
│   └── workflows/
└── environments/       # Environment configs
    ├── dev.yaml
    ├── test.yaml
    └── prod.yaml
```

## Architecture Decisions

### Separation of Concerns

1. **papers/**: Each paper implementation is isolated and self-contained
   - Allows independent development and testing
   - Prevents cross-contamination of implementations
   - Enables easy comparison of different approaches

2. **shared/**: Reusable components are centralized
   - Reduces code duplication
   - Ensures consistent implementations
   - Facilitates performance optimizations

3. **benchmarks/**: Dedicated benchmarking infrastructure
   - Provides consistent performance measurement
   - Enables fair comparisons between implementations
   - Tracks performance over time

### Naming Conventions

- **Directories**: lowercase with underscores (e.g., `shared_library/`)
- **Mojo files**: lowercase with underscores (e.g., `conv2d.mojo`)
- **Python files**: lowercase with underscores (e.g., `create_issues.py`)
- **Documentation**: Title case for READMEs, lowercase for guides

### Import Patterns

```mojo
# From papers importing shared components
from shared.layers import Conv2D
from shared.optimizers import Adam

# From benchmarks importing implementations
from papers.lenet5.model import LeNet5
from shared.training import Trainer
```

## API Contracts

### Paper Implementation Contract

Each paper implementation must provide:

1. `model.mojo`: Model implementation with standard interface
2. `training.mojo`: Training script with CLI interface
3. `evaluation.mojo`: Evaluation script for testing
4. `paper_info.yaml`: Metadata about the paper
5. `README.md`: Documentation including results

### Shared Component Contract

Each shared component must:

1. Implement a consistent API
2. Include comprehensive documentation
3. Provide unit tests
4. Support both CPU and GPU execution (where applicable)
5. Use Mojo's type system for safety

## Success Criteria

- [x] Complete directory structure defined with clear purposes
- [x] README templates created for each major directory
- [x] Template structure for new paper implementations defined
- [x] Architectural decisions documented
- [x] API contracts specified
- [x] Import patterns established
- [x] Naming conventions defined

## Implementation Notes

### Phase Dependencies

This planning phase output will be used by:

1. **Test Phase**: Create tests for directory validation
2. **Implementation Phase**: Create actual directory structure and templates
3. **Package Phase**: Bundle templates and documentation
4. **Cleanup Phase**: Refine based on implementation feedback

### Migration Considerations

For existing code:
- Papers currently in repository need to be moved to papers/
- Shared components need to be extracted and centralized
- Tests need to be reorganized to match new structure

## References

- [Agent Architecture Review](/home/user/ml-odyssey/notes/review/agent-architecture-review.md)
- [Worktree Strategy](/home/user/ml-odyssey/notes/review/worktree-strategy.md)
- [CLAUDE.md](/home/user/ml-odyssey/CLAUDE.md) - Project conventions and guidelines

## Next Steps

1. **Test Phase**: Write validation tests for directory structure
2. **Implementation Phase**: Create directories and README templates
3. **Package Phase**: Create distributable template packages
4. **Cleanup Phase**: Refine based on implementation experience

---

**Last Updated**: 2025-11-15
**Phase Status**: Plan - COMPLETE
