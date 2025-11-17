# System Architecture

ML Odyssey is a Mojo-based AI research platform that combines a hierarchical agent system,
modular shared library architecture, and structured paper implementation framework. This document
provides an overview of the system design and links to detailed specifications.

## System Overview

ML Odyssey consists of five integrated architectural components:

1. **Repository Architecture** - Organized into logical sections (foundation, shared library, papers,
   tooling, CI/CD, agentic workflows)
2. **Shared Library** - Reusable components for tensors, layers, activations, training utilities, and data processing
3. **Paper Implementations** - Research paper reproductions using shared library components
4. **Agent System** - 6-level hierarchical structure for AI-assisted development with 43+ reusable skills
5. **Build & Deployment** - Mojo package management, CI/CD pipelines, and distribution infrastructure

## Repository Architecture

The repository uses a purpose-driven top-level structure:

```text
```text

ml-odyssey/
├── .claude/              # Operational agent configs and skills
│   ├── agents/          # Working sub-agent configurations (~23 agents)
│   └── skills/          # Reusable capability definitions (43 skills)
├── agents/              # Team documentation and templates
├── shared/              # Shared library components
│   ├── core/            # Layers, activations, loss functions
│   ├── training/        # Optimizers, learning schedulers
│   ├── data/            # Dataset loading and preprocessing
│   └── utils/           # General utilities
├── papers/              # Research paper implementations
│   ├── _template/       # Template for new papers
│   └── lenet5/          # Example: LeNet-5 implementation
├── tools/               # Development utilities
│   ├── paper-scaffold/  # Paper scaffolding automation
│   ├── test-utils/      # Testing framework
│   └── benchmarking/    # Performance measurement
├── tests/               # Test suite (foundation, shared, agents, tools)
├── docs/                # User-facing documentation
├── notes/               # Planning, issues, and reviews
│   ├── plan/            # 4-level hierarchical task plans
│   ├── issues/          # Issue-specific working notes
│   └── review/          # Comprehensive architectural decisions
├── scripts/             # Automation and build scripts
└── configs/             # Centralized configuration management

```text

See `STRUCTURE.md` in the repository root for complete directory documentation.

## Shared Library Architecture

The shared library provides reusable ML components following modular design principles:

**Core Module** (`shared/core/`):

- **Layers** - Fully connected, convolutional, pooling, batch normalization
- **Activations** - ReLU, Sigmoid, Tanh, Softmax with forward/backward passes
- **Loss Functions** - Cross-entropy, MSE, MAE implementations

**Training Module** (`shared/training/`):

- Optimizers: SGD, Adam, RMSprop with momentum support
- Learning schedulers: Step decay, cosine annealing, warm-up
- Gradient computation and backpropagation utilities

**Data Module** (`shared/data/`):

- Dataset loaders for standard benchmarks (MNIST, CIFAR-10, ImageNet)
- Preprocessing pipelines: normalization, augmentation, batching
- Data validation and type safety

**Utils Module** (`shared/utils/`):

- Tensor utilities: transpose, reshape, slice operations
- Numerical utilities: precision conversion, quantization
- Profiling and debugging tools

All components use Mojo's performance features (struct for value types, SIMD for vectorization, compile-time optimization).

## Paper Implementation Architecture

Paper implementations follow a consistent pattern with shared library reuse:

```text
```text

papers/{paper_name}/
├── model.mojo          # Model definition using shared/core components
├── train.mojo          # Training script using shared/training utilities
├── evaluate.mojo       # Evaluation and metrics
├── README.md           # Paper documentation and results
└── tests/              # Paper-specific test suite

```text

Each paper:

- Implements core model architecture from the original paper
- Uses shared library components (no reimplementation)
- Includes benchmarking against original paper results
- Provides training scripts with standard configurations
- Maintains reproducibility documentation

## Agent System Architecture

The agent system implements a 6-level hierarchy enabling AI-assisted development through task decomposition and delegation:

**Level 0: Chief Architect**

- Strategic decisions and system-wide architecture
- Paper selection and repository roadmap

**Level 1: Section Orchestrators** (6 sections)

- Foundation, Shared Library, Tooling, Papers, CI/CD, Agentic Workflows
- Coordinate work within major sections

**Level 2: Design/Module Orchestrators**

- Architecture design and code review coordination
- Cross-module integration planning

**Level 3: Specialists** (5-8 per module)

- Implementation, testing, documentation, performance, security
- Component-level expert work

**Level 4: Engineers**

- Implementation tasks and code authoring
- Focused, well-defined technical work

**Level 5: Junior Engineers**

- Simple, well-defined tasks with detailed instructions
- Support for standard operations

**Skills System** (43 reusable capabilities):

- GitHub operations (PR creation, CI status checking, review automation)
- Code quality (linting, formatting, security scanning)
- Mojo-specific tools (formatting, testing, optimization)
- Workflow automation (phase management, planning, packaging)

See `notes/review/agent-system-overview.md` for complete hierarchy documentation.

## Build and Deployment

**Build System**:

- **Mojo Packages** (`.mojopkg`) - Compiled modules for shared library and tools
- **Build Artifacts** - Generated from `mojo.toml` and `pixi.toml`
- **Local Development** - `pixi shell` provides development environment

**Deployment**:

- **CI/CD Pipelines** - `.github/workflows/` with pre-commit validation, testing, and packaging
- **Distribution** - Package archives (`.tar.gz`, `.zip`) for tools and documentation
- **Installation** - Local development via pixi, package installation via Mojo package manager

**Testing**:

- Unit tests with pytest for Python utilities
- Mojo test framework for Mojo code
- Integration tests for agent system
- Performance benchmarks in `benchmarks/` with CI tracking

## Development Workflow

The system uses a 5-phase structured workflow for component development:

1. **Plan** - Design and specification (produces requirements for other phases)
2. **Test** - TDD-driven test suite (parallel after Plan)
3. **Implementation** - Code authoring and functionality (parallel after Plan)
4. **Package** - Building distributable artifacts (parallel after Plan)
5. **Cleanup** - Refactoring and finalization (after parallel phases)

Each phase has dedicated agent roles, clear entry/exit criteria, and automated CI validation.

## Detailed Specifications

For comprehensive architectural decisions and design rationale, see:

- **Agent Architecture** - `notes/review/agent-architecture-review.md`
- **Skills Design** - `notes/review/skills-design.md`
- **Orchestration Patterns** - `notes/review/orchestration-patterns.md`
- **Git Workflow** - `notes/review/worktree-strategy.md`
- **System Overview** - `notes/review/agent-system-overview.md`

See `notes/review/README.md` for the complete specifications index.

## Design Principles

- **Mojo First** - ML/AI implementation in Mojo with pragmatic Python exceptions
- **Modularity** - Independent components with well-defined interfaces
- **Type Safety** - Compile-time type checking and memory safety
- **Performance** - SIMD vectorization and gradient-based optimization
- **Reproducibility** - Detailed documentation and standard configurations
- **Automation** - Agent-driven development with reusable skills
