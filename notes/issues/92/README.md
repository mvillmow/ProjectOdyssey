# Issue #92: [Plan] Add Dependencies - Design and Documentation

## Objective

Design and document the dependencies section for `magic.toml`, defining all necessary packages for Mojo/MAX development and ML research with appropriate version constraints and organization.

## Deliverables

- Comprehensive specification for the dependencies section structure
- Complete list of required dependencies with version constraints
- Dependency categorization and organization strategy
- Documentation of version constraint rationale
- Migration strategy from pixi.toml dependencies

## Success Criteria

- [ ] All necessary dependencies for ML research identified
- [ ] Version constraints are appropriate and not overly restrictive
- [ ] Dependencies organized clearly by category
- [ ] Each dependency choice is documented with rationale
- [ ] Compatible with MAX/Mojo ecosystem requirements
- [ ] Migration path from pixi.toml defined

## References

- [Issue #87 Documentation](../87/README.md) - Base magic.toml structure
- [Add Dependencies Plan](../../plan/01-foundation/02-configuration-files/01-magic-toml/02-add-dependencies/plan.md) - Component requirements
- [CLAUDE.md](../../../CLAUDE.md) - Project conventions and guidelines
- [ADR-001](../../review/adr/ADR-001-language-selection-tooling.md) - Language selection strategy

## Planning Documentation

### 1. Dependencies Section Overview

**Purpose**: Define all external packages required for the ML Odyssey project, including runtime dependencies, development tools, and ML research libraries.

**Integration Point**: This section will be added to the base `magic.toml` structure defined in Issue #87.

### 2. Dependencies Structure Design

```toml
# Dependencies section for magic.toml
# Organized by category for clarity and maintenance

[dependencies]
# Core Language and Runtime Dependencies
max = ">=24.5.0,<25.0.0"           # MAX SDK for Mojo development
python = ">=3.11,<3.13"             # Python interpreter (3.11+ for modern features, <3.13 for stability)

# Machine Learning Frameworks
# Note: We use numpy for basic operations, will implement neural networks in Mojo
numpy = ">=1.24.0,<2.0.0"           # Numerical computing (major version pinned for API stability)
matplotlib = ">=3.7.0,<4.0.0"       # Plotting and visualization

# Development Tools
pre-commit = ">=3.5.0,<4.0.0"      # Git hooks for code quality
pytest = ">=7.4.0,<9.0.0"          # Testing framework for Python code
mypy = ">=1.5.0,<2.0.0"            # Type checking for Python code
ruff = ">=0.1.0,<1.0.0"            # Fast Python linter and formatter

# Documentation and Analysis
jupyterlab = ">=4.0.0,<5.0.0"      # Interactive notebooks for experiments
ipykernel = ">=6.25.0,<7.0.0"      # Jupyter kernel for Python

# Utilities
pyyaml = ">=6.0.0,<7.0.0"          # YAML parsing for configuration
toml = ">=0.10.0,<1.0.0"           # TOML parsing for magic.toml validation
click = ">=8.1.0,<9.0.0"           # CLI creation for tooling scripts
rich = ">=13.0.0,<14.0.0"          # Rich terminal output for better UX

# Optional: Research-specific packages (commented out initially)
# pillow = ">=10.0.0,<11.0.0"      # Image processing for vision models
# scikit-learn = ">=1.3.0,<2.0.0"  # ML algorithms for comparison
# pandas = ">=2.1.0,<3.0.0"        # Data manipulation (if needed)
```

### 3. Dependency Categories

#### 3.1 Core Dependencies

| Package | Purpose | Version Strategy | Justification |
|---------|---------|-----------------|---------------|
| `max` | MAX SDK for Mojo | `>=24.5.0,<25.0.0` | Pin major version for API stability |
| `python` | Python runtime | `>=3.11,<3.13` | 3.11+ for modern features, avoid bleeding edge |

#### 3.2 Machine Learning Libraries

| Package | Purpose | Version Strategy | Justification |
|---------|---------|-----------------|---------------|
| `numpy` | Numerical computing | `>=1.24.0,<2.0.0` | Essential for data manipulation, pin major |
| `matplotlib` | Visualization | `>=3.7.0,<4.0.0` | For plotting results and debugging |

**Note**: We intentionally exclude PyTorch/TensorFlow as we'll implement neural networks in Mojo for performance and learning purposes.

#### 3.3 Development Tools

| Package | Purpose | Version Strategy | Justification |
|---------|---------|-----------------|---------------|
| `pre-commit` | Code quality hooks | `>=3.5.0,<4.0.0` | Already in use, maintain compatibility |
| `pytest` | Testing framework | `>=7.4.0,<9.0.0` | Standard Python testing, wide range |
| `mypy` | Type checking | `>=1.5.0,<2.0.0` | Enforce type safety in Python code |
| `ruff` | Linting/formatting | `>=0.1.0,<1.0.0` | Fast, modern Python tooling |

#### 3.4 Documentation & Analysis

| Package | Purpose | Version Strategy | Justification |
|---------|---------|-----------------|---------------|
| `jupyterlab` | Interactive notebooks | `>=4.0.0,<5.0.0` | For experiments and tutorials |
| `ipykernel` | Jupyter Python kernel | `>=6.25.0,<7.0.0` | Required for notebooks |

#### 3.5 Utility Libraries

| Package | Purpose | Version Strategy | Justification |
|---------|---------|-----------------|---------------|
| `pyyaml` | YAML parsing | `>=6.0.0,<7.0.0` | For configuration files |
| `toml` | TOML parsing | `>=0.10.0,<1.0.0` | For validating magic.toml |
| `click` | CLI framework | `>=8.1.0,<9.0.0` | For creating command-line tools |
| `rich` | Terminal formatting | `>=13.0.0,<14.0.0` | Better CLI output experience |

### 4. Version Constraint Philosophy

#### 4.1 Constraint Types

1. **Caret (^)**: Allow compatible updates (not used in Magic, use >= and < instead)
2. **Tilde (~)**: Allow patch updates only (not used in Magic)
3. **Range (>=X,<Y)**: Explicit range for clarity (preferred approach)
4. **Exact (==)**: Pin to specific version (avoid unless necessary)

#### 4.2 Strategy by Package Type

- **Core Dependencies**: Conservative, pin major version
- **Development Tools**: Flexible, allow minor updates
- **ML Libraries**: Moderate, pin major for API stability
- **Utilities**: Flexible, wide ranges acceptable

#### 4.3 Update Strategy

- Review and test dependency updates quarterly
- Update development tools more frequently
- Core dependencies only when features needed
- Security updates as soon as available

### 5. Migration from Pixi

**Current pixi.toml dependencies:**
```toml
mojo = ">=0.25.7.0.dev2025110405,<0.26"
pre-commit = ">=3.5.0"
```

**Migration mapping:**
- `mojo` → Replaced by `max` (includes Mojo)
- `pre-commit` → Maintained with upper bound added

**Migration steps:**
1. Create magic.toml with all dependencies
2. Test both environments work
3. Gradually transition workflows
4. Deprecate pixi.toml when stable

### 6. Dependency Management Best Practices

#### 6.1 Selection Criteria

- **Necessity**: Is this dependency truly required?
- **Maintenance**: Is the package actively maintained?
- **Security**: Any known vulnerabilities?
- **License**: Compatible with MIT license?
- **Size**: Impact on environment size?
- **Alternatives**: Mojo implementation possible?

#### 6.2 Documentation Requirements

Each dependency should have:
- Clear purpose comment
- Version constraint rationale
- Usage location reference
- Alternative consideration note

#### 6.3 Review Process

- Dependencies reviewed in PR
- Security scan before merge
- Performance impact assessed
- Documentation updated

### 7. Future Considerations

#### 7.1 Potential Future Dependencies

**Computer Vision** (for CNN papers):
- `pillow` - Image processing
- `opencv-python` - Advanced vision operations

**Data Processing** (if needed):
- `pandas` - Dataframe operations
- `scikit-learn` - Reference implementations

**Deep Learning** (for comparison only):
- NOT including PyTorch/TensorFlow by design
- Implement everything in Mojo for learning

#### 7.2 Mojo-specific Packages

As Mojo ecosystem grows, prioritize:
- Native Mojo packages over Python
- MAX-optimized libraries
- Community Mojo implementations

### 8. Validation Strategy

#### 8.1 Syntax Validation

```python
# Validate TOML syntax
import toml

with open('magic.toml', 'r') as f:
    config = toml.load(f)
    assert 'dependencies' in config
    assert isinstance(config['dependencies'], dict)
```

#### 8.2 Version Constraint Validation

- Check all constraints are valid
- Ensure no conflicts between packages
- Verify packages exist in channels

#### 8.3 Installation Testing

```bash
# Test clean installation
magic env create -f magic.toml
magic run python -c "import numpy, matplotlib"
magic run mojo --version
```

### 9. Implementation Checklist

- [ ] Add dependencies section to magic.toml
- [ ] Group dependencies by category
- [ ] Add explanatory comments for each package
- [ ] Document version constraint rationale
- [ ] Ensure compatibility with MAX/Mojo
- [ ] Test installation in clean environment
- [ ] Update README with dependency info
- [ ] Create dependency update guide

### 10. Configuration Integration

The dependencies section will integrate with the base configuration from Issue #87:

```toml
# Complete magic.toml structure with dependencies

[project]
name = "ml-odyssey"
version = "0.1.0"
description = "A Mojo-based AI research platform for reproducing classic research papers"
# ... other project fields from Issue #87 ...

[dependencies]
# Dependencies as specified above
max = ">=24.5.0,<25.0.0"
python = ">=3.11,<3.13"
# ... additional dependencies ...

# Channels section (Issue #89) will follow
# [channels]
# default = ["conda-forge", "https://conda.modular.com/max"]
```

## Implementation Notes

### Key Decisions

1. **No PyTorch/TensorFlow**: Intentionally excluded to encourage Mojo implementations
2. **Conservative Core Versions**: MAX and Python pinned to major versions
3. **Development Tool Flexibility**: Wider ranges for non-critical tools
4. **Minimal Initial Set**: Start with essentials, add as needed

### Trade-offs

- **Simplicity vs Completeness**: Starting minimal, will expand based on needs
- **Stability vs Latest Features**: Favoring stability for research reproducibility
- **Python vs Mojo**: Using Python tools where Mojo alternatives don't exist yet

### Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Version conflicts | Build failures | Test all combinations, use lock file |
| Missing dependencies | Runtime errors | Comprehensive testing in clean env |
| Overly restrictive versions | Update difficulties | Regular review cycles |
| Package deprecation | Maintenance burden | Monitor package health metrics |

## Next Steps

1. **Issue #93: [Test] Add Dependencies** - Write tests for dependency validation
2. **Issue #94: [Implementation] Add Dependencies** - Add dependencies to magic.toml
3. **Issue #95: [Package] Add Dependencies** - Test installation and packaging
4. **Issue #96: [Cleanup] Add Dependencies** - Optimize and document

---

**Last Updated**: 2025-11-15
**Status**: Planning Complete ✅
**Agent**: Chief Architect
