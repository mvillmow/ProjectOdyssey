# Issue #103: [Plan] Magic TOML - Design and Documentation

## Objective

Create and configure the magic.toml file for the Magic package manager. This file defines project metadata, dependencies, channels, and environment configuration for the Mojo/MAX development environment. The planning phase will define detailed specifications, design the architecture and approach, document API contracts and interfaces, and create comprehensive design documentation.

## Deliverables

- Detailed specifications for magic.toml structure
- Magic package manager configuration design
- Project metadata schema
- Dependency specification format
- Channel configuration design
- Documentation of configuration best practices

## Success Criteria

- [ ] magic.toml file structure is fully specified
- [ ] Project metadata schema is complete and accurate
- [ ] All necessary dependencies are identified and documented
- [ ] Channel configuration approach is defined
- [ ] Configuration can be used to set up development environment
- [ ] Design documentation is comprehensive and actionable

## References

- **Source Plan**: `/notes/plan/01-foundation/02-configuration-files/01-magic-toml/plan.md`
- **Related Issues**:
  - #104 [Test] Magic TOML - Test Suite
  - #105 [Implementation] Magic TOML - Implementation
  - #106 [Packaging] Magic TOML - Integration and Packaging
  - #107 [Cleanup] Magic TOML - Cleanup and Finalization
- **Magic Documentation**: [Magic Package Manager](https://docs.modular.com/magic/)

## Implementation Notes

*To be filled during implementation*

## Design Decisions

### Magic.toml Structure

The magic.toml file follows the TOML format and is structured into several key sections:

#### 1. Project Metadata Section

```toml
[project]
name = "ml-odyssey"
version = "0.1.0"
description = "Mojo-based AI research platform for reproducing classic research papers"
authors = ["Your Name <your.email@example.com>"]
license = "MIT"
readme = "README.md"
homepage = "https://github.com/yourusername/ml-odyssey"
```

**Purpose**: Defines the project identity, versioning, and metadata for package management and distribution.

**Design Rationale**:
- Follows standard package manager conventions (similar to Python's pyproject.toml)
- Version uses semantic versioning (MAJOR.MINOR.PATCH)
- Metadata enables proper attribution and discoverability

#### 2. Dependencies Section

```toml
[dependencies]
max = ">=24.5.0"
mojo = ">=24.5.0"
```

**Purpose**: Specifies runtime dependencies required for the project.

**Design Rationale**:
- Uses version constraints to ensure compatibility
- MAX SDK provides the Mojo runtime and standard library
- Version pinning strategy: Use minimum version constraints (>=) for flexibility while ensuring minimum required features

#### 3. Development Dependencies Section

```toml
[dev-dependencies]
pytest = ">=7.0.0"
pre-commit = ">=3.0.0"
```

**Purpose**: Specifies dependencies only needed for development and testing.

**Design Rationale**:
- Separates dev tools from runtime requirements
- Keeps production environment lean
- Enables reproducible development setups

#### 4. Channel Configuration Section

```toml
[tool.magic.channels]
conda-forge = "https://conda.anaconda.org/conda-forge"
modular = "https://conda.modular.com/max"
```

**Purpose**: Defines package sources and their priority.

**Design Rationale**:
- Multiple channels enable access to both community and vendor packages
- Channel ordering determines resolution priority
- Modular channel provides official MAX/Mojo packages
- conda-forge provides Python ecosystem packages

### Dependencies Strategy

#### Core Dependencies

1. **MAX SDK** (>=24.5.0)
   - Provides Mojo compiler and runtime
   - Includes MLIR infrastructure
   - Required for all Mojo code execution

2. **Mojo Language** (>=24.5.0)
   - Core language support
   - Standard library
   - SIMD and tensor operations

#### Development Dependencies

1. **pytest** - Python testing framework
   - Used for automation script testing
   - Integration testing support
   - Familiar to Python developers

2. **pre-commit** - Git hook manager
   - Automated code quality checks
   - Consistent formatting enforcement
   - Prevents committing problematic code

#### Dependency Resolution Strategy

- **Version Constraints**: Use `>=` for minimum version requirements
- **Lock Files**: Magic generates lock files for reproducible builds
- **Update Policy**: Regular security updates, conservative feature updates
- **Compatibility**: Test across supported MAX/Mojo versions

### Channel Configuration

#### Channel Priority

1. **Modular Channel** (Primary)
   - Official MAX/Mojo packages
   - Guaranteed compatibility
   - Latest Mojo features
   - URL: `https://conda.modular.com/max`

2. **conda-forge** (Secondary)
   - Community Python packages
   - Development tools
   - Testing frameworks
   - URL: `https://conda.anaconda.org/conda-forge`

#### Channel Selection Rationale

- **Modular first**: Ensures official Mojo packages take precedence
- **conda-forge as fallback**: Provides access to Python ecosystem
- **No defaults channel**: Avoid conflicts with Anaconda defaults
- **Explicit ordering**: Deterministic package resolution

#### Channel Configuration Best Practices

1. **Minimize channels**: Only include necessary sources
2. **Official sources first**: Prefer vendor channels for core dependencies
3. **Document custom channels**: Explain any non-standard sources
4. **Version awareness**: Some packages may differ across channels

### Configuration Best Practices

#### 1. Minimal but Complete

- Include only necessary dependencies
- Avoid over-specifying version constraints
- Let Magic handle transitive dependencies
- Document non-obvious choices with comments

#### 2. Reproducibility

- Use lock files for production deployments
- Document environment setup in README
- Test configuration on clean environments
- Version control magic.toml and lock files

#### 3. Maintainability

- Group related dependencies logically
- Use comments to explain unusual choices
- Keep dependencies up-to-date
- Regular security audits

#### 4. Cross-Platform Compatibility

- Test on multiple platforms (Linux, macOS, Windows)
- Avoid platform-specific dependencies when possible
- Document platform-specific requirements
- Use Magic's platform markers if needed

### File Template

```toml
# ML Odyssey - Magic Package Manager Configuration
# This file defines the Mojo/MAX development environment

[project]
name = "ml-odyssey"
version = "0.1.0"
description = "Mojo-based AI research platform for reproducing classic research papers"
authors = ["Your Name <your.email@example.com>"]
license = "MIT"
readme = "README.md"
homepage = "https://github.com/yourusername/ml-odyssey"

# Runtime dependencies
[dependencies]
max = ">=24.5.0"  # MAX SDK provides Mojo runtime and standard library
mojo = ">=24.5.0"  # Mojo language support

# Development-only dependencies
[dev-dependencies]
pytest = ">=7.0.0"  # Python testing framework for automation scripts
pre-commit = ">=3.0.0"  # Git hook manager for code quality

# Package sources
[tool.magic.channels]
modular = "https://conda.modular.com/max"  # Official MAX/Mojo packages (primary)
conda-forge = "https://conda.anaconda.org/conda-forge"  # Python ecosystem packages (secondary)
```

### Validation Strategy

The configuration will be validated through:

1. **Syntax validation**: Magic CLI validates TOML syntax
2. **Dependency resolution**: Test that all dependencies can be resolved
3. **Environment creation**: Create clean environment from config
4. **Build testing**: Verify project builds with specified dependencies
5. **Cross-platform testing**: Test on Linux, macOS, Windows

### Future Considerations

1. **Additional dependencies** as project grows
2. **Build scripts** for complex compilation steps
3. **Environment variables** for configuration
4. **Custom tasks** using Magic's task runner
5. **Publishing configuration** when ready to distribute

## Next Steps

After this planning phase is complete:

1. **Issue #104**: Create test suite to validate magic.toml configuration
2. **Issue #105**: Implement the actual magic.toml file
3. **Issue #106**: Integrate with CI/CD and documentation
4. **Issue #107**: Cleanup and finalization
