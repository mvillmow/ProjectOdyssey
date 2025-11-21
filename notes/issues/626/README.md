# Issue #626: [Plan] Add Python Dependencies - Design and Documentation

## Objective

Define comprehensive specifications for Python dependency management in pyproject.toml, including runtime dependencies, development tools, and optional feature groups. This planning phase establishes the architecture for organizing dependencies to support development, testing, documentation, and optional features while maintaining clean separation of concerns.

## Deliverables

- **Dependency Architecture Specification**: Complete design for organizing Python dependencies into appropriate groups (runtime, development, testing, documentation, optional features)
- **Version Constraint Strategy**: Guidelines for specifying version constraints and pinning strategies
- **Dependency Group Design**: Structure for [project.dependencies] and [project.optional-dependencies] sections
- **Integration Documentation**: How Python dependencies integrate with the Mojo-based project and pixi environment

## Success Criteria

- [ ] All necessary Python dependencies are identified and categorized
- [ ] Dependencies are organized into appropriate groups (runtime, dev, test, docs)
- [ ] Version constraint strategy is documented with rationale
- [ ] Optional dependency groups are properly designed
- [ ] Integration with existing pyproject.toml structure is planned
- [ ] Compatibility with pixi environment management is verified
- [ ] Design documentation is complete and actionable for implementation phase

## Design Decisions

### 1. Dependency Organization Strategy

**Decision**: Separate dependencies into three primary categories:

1. **Runtime Dependencies** ([project.dependencies])
   - Currently includes: pytest suite (pytest, pytest-cov, pytest-timeout, pytest-xdist)
   - Rationale: These are automation/testing tools used by the Python scripts
   - Note: Primary ML/AI implementation will use Mojo, not Python libraries

1. **Development Dependencies** ([project.optional-dependencies.dev])
   - Already configured: pre-commit, safety, bandit, mkdocs, mkdocs-material, pytest-benchmark, ruff, mypy
   - Rationale: Tools needed for development workflow but not runtime execution

1. **Feature-Specific Groups** (future expansion)
   - Potential groups: docs, lint, security, benchmarking
   - Allows selective installation based on use case

**Rationale**: This separation provides:

- Minimal production dependencies
- Flexible development environment setup
- Clear distinction between required and optional tools
- Compatibility with CI/CD workflows (install only what's needed)

### Alternatives Considered

- Single dependency list: Rejected - would bloat production installations
- More granular groups (lint, test, security separate): Deferred - current scale doesn't justify additional complexity (YAGNI principle)

### 2. Version Constraint Strategy

**Decision**: Use minimum version constraints with compatibility markers

- **Runtime Dependencies**: Use `>=` with tested minimum version (e.g., `pytest>=7.0.0`)
- **Development Tools**: Use `>=` for tools where latest features are beneficial (e.g., `ruff>=0.1.0`)
- **Avoid Upper Bounds**: Do not pin maximum versions unless known incompatibilities exist

### Rationale

- Allows benefiting from bug fixes and security patches
- Prevents dependency conflicts in larger projects
- Follows Python packaging best practices
- Upper bounds create maintenance burden and dependency hell

### Alternatives Considered

- Exact pinning (==): Rejected - creates brittle dependency chains
- Pessimistic versioning (~=): Considered overkill for current project maturity

### 3. Python Dependency Scope in Mojo Project

**Decision**: Limit Python dependencies to automation and tooling only

### Current State Analysis

- Project is Mojo-first for ML/AI implementation (per ADR-001)
- Python used for: automation scripts, GitHub integration, testing infrastructure
- Existing dependencies align with this: pytest for testing, ruff/mypy for linting, mkdocs for docs

### Implications

- No NumPy, PyTorch, TensorFlow, or ML libraries in dependencies
- Python dependencies support the development workflow, not the ML implementation
- Keep dependency list lean to avoid conflicts with Mojo ecosystem

### Rationale

- Aligns with ADR-001 language selection strategy
- Reduces maintenance burden
- Avoids version conflicts between Python ML libs and Mojo
- Keeps focus on Mojo as primary implementation language

### 4. Integration with Pixi Environment

**Decision**: pyproject.toml manages Python-specific dependencies; pixi.toml manages environment

### Coordination Strategy

- pixi.toml: Defines the complete development environment (Mojo, Python, system tools)
- pyproject.toml: Defines Python package dependencies and tool configurations
- Installation workflow: `pixi install` → sets up environment including Python packages

**Current State**: pyproject.toml already exists with dependencies; pixi.toml manages environment

### Rationale

- Follows standard Python project structure
- Pixi respects pyproject.toml dependency specifications
- Allows standard Python tooling to work (pip install -e .[dev])
- Maintains compatibility with both pixi and traditional Python workflows

## Architecture Specification

### Dependency Groups Structure

```toml
[project.dependencies]
# Automation and testing infrastructure
# Rationale: Python scripts need pytest for test automation
pytest>=7.0.0
pytest-cov>=4.0.0
pytest-timeout>=2.1.0
pytest-xdist>=3.0.0

[project.optional-dependencies]
dev = [
    # Pre-commit hooks and security
    "pre-commit>=3.0.0",
    "safety>=2.0.0",
    "bandit>=1.7.0",

    # Documentation
    "mkdocs>=1.5.0",
    "mkdocs-material>=9.0.0",

    # Testing and benchmarking
    "pytest-benchmark>=4.0.0",

    # Linting and type checking
    "ruff>=0.1.0",
    "mypy>=1.0.0",
]
```text

### Version Constraint Guidelines

1. **Minimum Versions**: Based on first version with required features
   - pytest 7.0.0: Introduces improved fixture handling and modern test discovery
   - ruff 0.1.0: First stable release with comprehensive linting rules
   - mypy 1.0.0: Stable API with modern type checking features

1. **No Maximum Versions**: Trust semantic versioning for compatibility
   - Exception: Document known incompatibilities if discovered

1. **Python Version Requirement**: Already set to `>=3.11`
   - Aligns with modern Python features
   - Compatible with pixi environment

### Dependency Addition Process

When adding new dependencies:

1. **Justify the Need**: Document why the dependency is required (avoid YAGNI violations)
1. **Choose Appropriate Group**: runtime vs dev vs optional feature group
1. **Specify Minimum Version**: Based on required features, not latest version
1. **Document in Comments**: Explain non-obvious choices in pyproject.toml
1. **Test Integration**: Verify compatibility with existing dependencies and pixi environment

## References

### Source Plan

- [notes/plan/01-foundation/02-configuration-files/02-pyproject-toml/02-add-python-deps/plan.md](notes/plan/01-foundation/02-configuration-files/02-pyproject-toml/02-add-python-deps/plan.md)

### Parent Plan

- [notes/plan/01-foundation/02-configuration-files/02-pyproject-toml/plan.md](notes/plan/01-foundation/02-configuration-files/02-pyproject-toml/plan.md)

### Related Issues

- Issue #627: [Test] Add Python Dependencies
- Issue #628: [Implementation] Add Python Dependencies
- Issue #629: [Package] Add Python Dependencies
- Issue #630: [Cleanup] Add Python Dependencies

### Comprehensive Documentation

- [ADR-001: Language Selection Strategy](../../review/adr/ADR-001-language-selection-tooling.md) - Mojo-first approach for ML/AI, Python for automation
- [Development Principles (CLAUDE.md)](CLAUDE.md) - KISS, YAGNI, DRY, SOLID principles

### Current State

- **pyproject.toml**: Already exists with basic dependencies configured
- **Dependencies Present**: pytest suite, pre-commit, safety, bandit, mkdocs, ruff, mypy
- **Next Phase**: Testing (verify dependency resolution and compatibility)

## Implementation Notes

*This section will be populated during subsequent phases (Test, Implementation, Packaging, Cleanup) with findings, issues discovered, and decisions made during execution.*

### Phase Progress

- [x] Planning (#626) - Complete
- [ ] Testing (#627) - Pending
- [ ] Implementation (#628) - Pending
- [ ] Packaging (#629) - Pending
- [ ] Cleanup (#630) - Pending
