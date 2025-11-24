# Issue #661: [Plan] Configuration Files - Design and Documentation

## Objective

Set up all necessary configuration files for the Mojo/MAX development environment, including Magic package manager configuration, Python project configuration, and Git configuration for proper handling of ML artifacts and large files.

## Deliverables

- `magic.toml` - Magic package manager configuration for Mojo/MAX development environment
- `pyproject.toml` - Python project configuration for development tools and dependencies
- `.gitignore` - Comprehensive ignore patterns for generated files and ML artifacts
- `.gitattributes` - Git file handling configuration for different file types
- Git LFS configuration - Setup for handling large model files and datasets

## Success Criteria

- [ ] `magic.toml` is valid and properly configured with project metadata, dependencies, and channels
- [ ] `pyproject.toml` is valid with all necessary development tools configured
- [ ] Git properly ignores generated files and handles large files with LFS
- [ ] All configuration files follow best practices and are well-documented
- [ ] Development environment can be reproducibly set up from configuration files

## Design Decisions

### 1. Magic Package Manager Configuration

**Decision**: Use Magic as the primary package manager for Mojo/MAX dependencies

### Rationale

- Magic is the recommended package manager for Mojo/MAX projects
- Provides consistent environment management across different platforms
- Simplifies dependency resolution for Mojo-specific packages

### Approach

- Define project metadata (name, version, description)
- Specify all Mojo/MAX development dependencies
- Configure appropriate package channels for Mojo ecosystem
- Document any unusual dependency choices with inline comments

### Alternatives Considered

- Using system-level package managers (rejected: lacks Mojo ecosystem integration)
- Managing dependencies manually (rejected: not reproducible)

### 2. Python Project Configuration

**Decision**: Use `pyproject.toml` as the central configuration file for Python tooling

### Rationale

- PEP 518/621 standard for Python project configuration
- Consolidates tool configurations in a single file
- Better IDE and tooling support

### Approach

- Configure project metadata following Python packaging standards
- Specify Python dependencies for automation and testing scripts
- Configure development tools (black, ruff, pytest, mypy) with sensible defaults
- Keep tool configurations minimal and document non-standard choices

### Tools to Configure

- **black**: Code formatting for Python scripts
- **ruff**: Fast Python linting
- **pytest**: Testing framework
- **mypy**: Static type checking

### Alternatives Considered

- Separate config files for each tool (rejected: more complex to maintain)
- No Python tooling configuration (rejected: inconsistent code quality)

### 3. Git Configuration Strategy

**Decision**: Use comprehensive `.gitignore`, `.gitattributes`, and Git LFS for ML project file management

### Rationale

- ML projects generate many artifacts (models, checkpoints, logs) that shouldn't be version-controlled
- Large model files require special handling to avoid repository bloat
- Different file types need appropriate Git handling (binary vs text, line endings)

### Git Ignore Strategy

- Exclude all generated files (build artifacts, compiled code, logs)
- Ignore Python cache and environment files
- Exclude ML-specific artifacts (checkpoints, trained models, datasets)
- Keep configuration minimal but comprehensive

### Git LFS Strategy

- Track large binary files (>50MB) automatically
- Configure LFS for model files (`.pt`, `.pth`, `.onnx`, `.safetensors`)
- Track dataset files if they must be version-controlled
- Document LFS usage patterns for team

### Git Attributes Strategy

- Configure line ending normalization for cross-platform compatibility
- Mark binary files to prevent diff attempts
- Configure merge strategies for specific file types

### Alternatives Considered

- DVC for data versioning (deferred: added complexity, may add later)
- No LFS configuration (rejected: will cause repository bloat)
- Minimal `.gitignore` (rejected: will clutter repository)

### 4. Documentation Strategy

**Decision**: Use inline comments in configuration files to explain non-obvious choices

### Rationale

- Configuration files are self-documenting
- Comments provide context for future maintainers
- Reduces need for separate documentation

### Approach

- Add comments explaining unusual dependency choices
- Document tool configuration reasoning
- Link to relevant documentation for complex configurations

## Architecture

### Configuration File Hierarchy

```text
ml-odyssey/
├── magic.toml              # Mojo/MAX environment configuration
├── pyproject.toml          # Python project and tooling configuration
├── .gitignore              # Git ignore patterns
├── .gitattributes          # Git file handling configuration
└── .git/
    └── lfs/                # Git LFS storage (auto-created)
```text

### Dependency Management Strategy

### Two-Layer Approach

1. **Mojo/MAX Dependencies** (magic.toml):
   - Mojo compiler and runtime
   - MAX framework and libraries
   - System-level development tools

1. **Python Dependencies** (pyproject.toml):
   - Automation scripts dependencies
   - Testing frameworks
   - Development tooling

### File Handling Strategy

### Three-Tier Classification

1. **Version-Controlled**: Source code, configuration, documentation
1. **Ignored**: Generated files, caches, logs, temporary files
1. **LFS-Tracked**: Large binary files (models, datasets) when necessary

## References

- **Source Plan**: [notes/plan/01-foundation/02-configuration-files/plan.md](notes/plan/01-foundation/02-configuration-files/plan.md)
- **Parent Plan**: [notes/plan/01-foundation/plan.md](notes/plan/01-foundation/plan.md)

### Child Plans

- [Magic TOML Configuration](notes/plan/01-foundation/02-configuration-files/01-magic-toml/plan.md)
- [Pyproject TOML Configuration](notes/plan/01-foundation/02-configuration-files/02-pyproject-toml/plan.md)
- [Git Configuration](notes/plan/01-foundation/02-configuration-files/03-git-config/plan.md)

### Related Issues

- Issue #662: [Test] Configuration Files - Test Strategy and Implementation
- Issue #663: [Implementation] Configuration Files - File Creation and Setup
- Issue #664: [Package] Configuration Files - Integration and Validation
- Issue #665: [Cleanup] Configuration Files - Finalization and Documentation

### Related Documentation

- [Agent Hierarchy](agents/hierarchy.md)
- [5-Phase Development Workflow](notes/review/README.md)
- [Repository Architecture](CLAUDE.md#repository-architecture)

## Implementation Notes

This section will be populated during the implementation phase with:

- Actual configuration values chosen
- Issues encountered during setup
- Deviations from planned approach
- Lessons learned
