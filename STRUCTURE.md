# ML Odyssey Repository Structure

## Overview

ML Odyssey is organized into distinct top-level directories, each serving a specific purpose in the ML research
platform. This guide helps you navigate the repository and understand where to find or add content.

## Quick Navigation

**What are you looking for?**

- **ML implementations** → `papers/`
- **Reusable components** → `shared/`
- **Documentation** → `docs/`
- **Performance benchmarks** → `benchmarks/`
- **Development tools** → `tools/`
- **Configuration files** → `configs/`
- **Test files** → `tests/`
- **Examples** → `examples/`
- **Build scripts** → `scripts/`
- **Agent system** → `agents/`, `.claude/agents/`
- **CI/CD workflows** → `.github/workflows/`

## Top-Level Directory Structure

```text
ml-odyssey/
├── .claude/             # Claude Code configurations
│   ├── agents/          # Operational agent configuration files
│   └── skills/          # Agent skill definitions
├── .github/             # GitHub workflows and configurations
│   └── workflows/       # CI/CD pipeline definitions
├── agents/              # Agent system documentation and templates
├── benchmarks/          # Performance benchmarking infrastructure
├── configs/             # Centralized configuration management
├── docs/                # User-facing documentation
├── examples/            # Usage examples and demonstrations
├── logs/                # Execution logs and state files
├── notes/               # Planning, issues, and architectural reviews
│   ├── plan/            # 4-level hierarchical plans
│   ├── issues/          # Issue-specific documentation
│   └── review/          # Comprehensive specifications
├── papers/              # Research paper implementations
│   └── _template/       # Template for new papers
├── scripts/             # Automation and build scripts
├── shared/              # Shared library components
│   ├── core/            # Core operations (layers, activations, loss)
│   ├── training/        # Training utilities
│   ├── data/            # Data loading and preprocessing
│   └── utils/           # General utilities
├── tests/               # Test suite
│   ├── foundation/      # Foundation tests
│   ├── shared/          # Shared library tests
│   ├── agents/          # Agent system tests
│   └── tools/           # Tool tests
└── tools/               # Development utilities
    ├── paper-scaffold/  # Paper scaffolding tool
    ├── test-utils/      # Testing utilities
    ├── benchmarking/    # Benchmarking framework
    └── codegen/         # Code generation tools
```

## Core Directories

### papers/ - Research Implementations

**Purpose**: Implementations of ML research papers

**Structure**:
```text
papers/
├── _template/          # Template for new papers
├── lenet5/             # LeNet-5 implementation
│   ├── model.mojo      # Model definition
│   ├── train.mojo      # Training script
│   ├── tests/          # Paper-specific tests
│   └── README.md       # Paper documentation
└── {paper_name}/       # Future paper implementations
```

**When to Use**:
- Implementing a new research paper
- Looking for ML model implementations
- Finding paper-specific training code

**See Also**: `tools/paper-scaffold/` for creating new papers

### shared/ - Reusable Components

**Purpose**: Core ML components used across paper implementations

**Structure**:
```text
shared/
├── core/               # Core ML operations
│   ├── layers/         # Neural network layers
│   ├── activations/    # Activation functions
│   └── loss/           # Loss functions
├── training/           # Training utilities
│   ├── optimizers/     # Optimization algorithms
│   └── schedulers/     # Learning rate schedulers
├── data/               # Data handling
│   └── loaders/        # Data loading utilities
└── utils/              # General utilities
    └── config_loader/  # Configuration management
```

**When to Use**:
- Building ML models (use layers, activations)
- Training models (use optimizers, schedulers)
- Loading data (use data loaders)
- Managing configs (use config_loader)

**See Also**: `docs/api/` for API reference

## Supporting Directories

### benchmarks/ - Performance Measurement

**Purpose**: Benchmark ML implementations for performance tracking

**Structure**:
```text
benchmarks/
├── scripts/            # Benchmark execution scripts
├── baselines/          # Baseline results for comparison
└── results/            # Timestamped benchmark results
```

**When to Use**:
- Measuring performance of implementations
- Detecting performance regressions
- Comparing different approaches
- Tracking historical performance

**See Also**: `tools/benchmarking/` for benchmarking framework

### docs/ - User Documentation

**Purpose**: Comprehensive documentation for users and contributors

**Structure**:
```text
docs/
├── getting-started/    # New user onboarding
├── core/               # Core concepts and fundamentals
├── advanced/           # Advanced topics
└── dev/                # Developer documentation
```

**When to Use**:
- Onboarding new team members
- Learning how to use the platform
- Understanding core concepts
- Finding API documentation

**See Also**: `notes/review/` for architectural decisions

### agents/ - AI Agent System

**Purpose**: Documentation for the Claude agent hierarchy

**Structure**:
```text
agents/
├── hierarchy.md        # Visual hierarchy diagram
├── delegation-rules.md # Coordination patterns
├── templates/          # Agent configuration templates
├── guides/             # Practical guides
└── docs/               # Integration documentation
```

**Operational Configs**: `.claude/agents/` (actual agent files)

**When to Use**:
- Understanding agent hierarchy
- Creating new agents
- Debugging agent workflows
- Learning delegation patterns

**See Also**: `.claude/agents/` for operational configs

### tools/ - Development Utilities

**Purpose**: Tools for developer productivity during implementation

**Structure**:
```text
tools/
├── paper-scaffold/     # Generate paper structure
├── test-utils/         # Testing utilities
├── benchmarking/       # Benchmarking framework
└── codegen/            # Code generation
```

**When to Use**:
- Starting new paper implementations
- Generating test fixtures
- Creating boilerplate code
- Running benchmarks

**See Also**: `scripts/` for automation scripts

### configs/ - Configuration Management

**Purpose**: Centralized configuration for experiments

**Structure**:
```text
configs/
├── defaults/           # Default configurations
├── papers/             # Paper-specific configs
├── experiments/        # Experiment variations
└── templates/          # Configuration templates
```

**When to Use**:
- Creating new experiments
- Configuring training parameters
- Managing environment settings
- Overriding default values

**See Also**: `shared/utils/config_loader/` for loading configs

## Specialized Directories

### tests/ - Test Suite

**Purpose**: Comprehensive test coverage for all components

**Structure**:
```text
tests/
├── foundation/         # Foundation tests (structure, configs)
├── shared/             # Shared library tests
├── agents/             # Agent system tests
└── tools/              # Development tool tests
```

**Organization**: Tests mirror the structure they test

**When to Use**:
- Running tests (`mojo test tests/`)
- Adding new tests
- Checking test coverage

**See Also**: `tools/test-utils/` for test utilities

### examples/ - Usage Demonstrations

**Purpose**: Working examples and tutorials

**Structure**:
```text
examples/
├── mnist/              # MNIST dataset examples
├── custom_layer/       # Custom layer tutorial
└── training/           # Training examples
```

**When to Use**:
- Learning how to use components
- Finding working code examples
- Understanding best practices

**See Also**: `docs/` for comprehensive documentation

### scripts/ - Automation Scripts

**Purpose**: Repository management and automation

**Structure**:
```text
scripts/
├── setup.py            # Environment setup
├── create_issues.py    # GitHub issue creation
├── validate_*.py       # Validation scripts
└── agents/             # Agent management scripts
```

**When to Use**:
- Setting up development environment
- Automating repository tasks
- Running validation checks
- Managing GitHub issues

**See Also**: `tools/` for development utilities

### notes/ - Planning and Documentation

**Purpose**: Planning hierarchies, issue docs, and architectural reviews

**Structure**:
```text
notes/
├── plan/               # 4-level hierarchical plans
├── issues/             # Issue-specific documentation
└── review/             # Architectural decisions and specs
```

**When to Use**:
- Understanding project planning
- Reading issue-specific notes
- Reviewing architectural decisions
- Finding comprehensive specs

**See Also**: `docs/` for user-facing documentation

## Decision Tree: Where to Add Content

### Is it...

**An ML paper implementation?**
→ `papers/{paper_name}/`

**A reusable ML component?**
→ `shared/{core|training|data|utils}/`

**User-facing documentation?**
→ `docs/{getting-started|core|advanced|dev}/`

**A performance benchmark?**
→ `benchmarks/scripts/`

**A development utility?**
→ `tools/{category}/`

**A configuration file?**
→ `configs/{defaults|papers|experiments}/`

**A test?**
→ `tests/{foundation|shared|agents|tools}/`

**An example or tutorial?**
→ `examples/`

**An automation script?**
→ `scripts/`

**Agent documentation?**
→ `agents/` (docs) or `.claude/agents/` (configs)

**Issue-specific notes?**
→ `notes/issues/{issue_number}/`

**Architectural decisions?**
→ `notes/review/`

## Common Workflows

### Workflow 1: Implementing a New Paper

1. **Generate structure**: `python tools/paper-scaffold/scaffold.py --paper {name}`
2. **Create configs**: Copy `configs/templates/paper.yaml` to `configs/papers/{name}/`
3. **Implement model**: Write `papers/{name}/model.mojo`
4. **Add tests**: Create `papers/{name}/tests/`
5. **Write docs**: Update `papers/{name}/README.md`
6. **Add benchmarks**: Create `benchmarks/scripts/{name}_benchmark.mojo`

### Workflow 2: Adding a Reusable Component

1. **Implement**: Add to `shared/{core|training|data|utils}/`
2. **Test**: Add tests to `tests/shared/`
3. **Document**: Update API docs in `docs/api/`
4. **Benchmark**: Add to `benchmarks/scripts/`

### Workflow 3: Adding Documentation

1. **Determine category**: Getting-started, core, advanced, or dev
2. **Create file**: `docs/{category}/{topic}.md`
3. **Update index**: Add link to `docs/index.md`
4. **Validate**: Run `pre-commit run markdownlint-cli2`

### Workflow 4: Creating a New Tool

1. **Implement**: Add to `tools/{category}/`
2. **Test**: Add tests to `tests/tools/`
3. **Document**: Create `tools/{category}/README.md`
4. **Integrate**: Update `tools/README.md`

## Directory Dependencies

### Dependency Graph

```text
papers/
  ↓ uses
shared/ ← configs/
  ↓ uses
tests/

tools/
  ↓ generates
papers/ + configs/

benchmarks/
  ↓ measures
papers/ + shared/

docs/
  ↓ documents
(all directories)

agents/
  ↓ automates
(all directories)
```

### Key Relationships

**papers/** depends on:
- `shared/` - Reusable components
- `configs/` - Configuration files
- `tools/` - Scaffolding and utilities

**shared/** used by:
- `papers/` - Paper implementations
- `examples/` - Usage examples
- `tests/` - Test suites

**configs/** used by:
- `papers/` - Training scripts
- `benchmarks/` - Benchmark configs
- `tools/` - Template generation

**tools/** generates:
- `papers/` - Paper structure
- `configs/` - Configuration files
- `tests/` - Test templates

## File Organization Principles

### 1. Single Responsibility

Each directory has one clear purpose. Don't mix:
- User documentation (`docs/`) with planning notes (`notes/`)
- Development tools (`tools/`) with automation scripts (`scripts/`)
- Agent docs (`agents/`) with agent configs (`.claude/agents/`)

### 2. Mirrored Structure

Tests mirror the structure they test:
- `shared/core/layers/` ↔ `tests/shared/core/layers/`
- `tools/paper-scaffold/` ↔ `tests/tools/paper-scaffold/`

### 3. Template Pattern

Reusable templates in dedicated locations:
- `papers/_template/` - Paper implementation template
- `configs/templates/` - Configuration templates
- `agents/templates/` - Agent configuration templates

### 4. Clear Separation

Operational files separate from documentation:
- `.claude/agents/` - Operational agent configs
- `agents/` - Agent documentation
- `.github/workflows/` - CI/CD workflows
- `docs/dev/ci-cd.md` - CI/CD documentation

## Validation

### Structure Validation

Check directory structure is correct:
```bash
python scripts/validate_structure.py
```

### Documentation Validation

Validate all READMEs are complete:
```bash
python scripts/check_readmes.py
```

### Link Validation

Check all documentation links:
```bash
python scripts/validate_links.py
```

## Best Practices

### When Creating New Content

1. **Check existing structure** - Don't create duplicate directories
2. **Follow conventions** - Use established patterns
3. **Add READMEs** - Every directory needs a README
4. **Update indexes** - Link new content from appropriate indexes
5. **Run validation** - Ensure structure compliance

### When Organizing Files

1. **Choose the right directory** - Use decision tree
2. **Follow naming conventions** - Consistent file naming
3. **Add comprehensive READMEs** - Explain purpose and usage
4. **Link related content** - Cross-reference where appropriate
5. **Keep it simple** - Don't over-nest directories

### When Documenting

1. **User docs** → `docs/`
2. **Issue-specific notes** → `notes/issues/`
3. **Architectural decisions** → `notes/review/`
4. **Code comments** → In source files
5. **API reference** → `docs/api/`

## References

- **Issue #80**: Package - Create Supporting Directories
- **CLAUDE.md**: Project conventions and guidelines
- **CONTRIBUTING.md**: Contribution guidelines
- **docs/getting-started/repository-structure.md**: Team onboarding guide
- **docs/core/supporting-directories.md**: Integration guide

## Support

Questions about repository organization?

1. Check this guide first
2. Review `docs/getting-started/repository-structure.md`
3. Consult `notes/review/` for architectural decisions
4. Ask in team channels

---

**Last Updated**: 2025-11-16
**Maintained By**: Foundation Orchestrator
