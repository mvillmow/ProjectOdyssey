# Issue #636: [Plan] Pyproject TOML - Design and Documentation

## Objective

Create and configure the pyproject.toml file for Python project configuration. This file defines Python package metadata, dependencies, and tool configurations for linting, formatting, and testing.

## Deliverables

- pyproject.toml file at repository root
- Project metadata configured ([project] section)
- Python dependencies specified (runtime and development)
- Tool configurations for black, ruff, pytest, mypy, and coverage
- Build system configuration ([build-system] section)

## Success Criteria

- [ ] pyproject.toml file exists and is valid TOML
- [ ] Project metadata is complete (name, version, description, authors, license)
- [ ] Python version requirement specified (>=3.11 for ML Odyssey)
- [ ] Dependencies are organized into runtime and optional groups
- [ ] Development tools (formatters, linters, type checkers, test runners) are configured
- [ ] File follows Python packaging standards (PEP 621, PEP 518)
- [ ] Tool configurations align with team coding standards

## Design Decisions

### 1. Build System Configuration

**Decision**: Use setuptools as the build backend

**Rationale**:
- Industry standard for Python packaging
- Excellent compatibility with pip and other Python tools
- Well-documented and widely understood
- Supports both src-layout and flat-layout projects

**Alternatives Considered**:
- poetry: More opinionated, additional dependency
- flit: Simpler but less flexible for complex projects
- hatch: Newer, still maturing ecosystem

**Implementation**: Configure [build-system] section with setuptools>=65.0

### 2. Dependency Organization

**Decision**: Separate runtime dependencies from development dependencies using optional-dependencies

**Rationale**:
- Clear separation of concerns
- Users installing ML Odyssey don't need dev tools
- CI/CD can install specific dependency groups as needed
- Follows PEP 621 best practices

**Structure**:
- [project.dependencies]: Core runtime dependencies (currently pytest for TDD)
- [project.optional-dependencies.dev]: Development tools (pre-commit, ruff, mypy, mkdocs, etc.)

### 3. Python Version Requirement

**Decision**: Require Python >=3.11

**Rationale**:
- Modern Python features (improved type hints, better error messages)
- Performance improvements in 3.11+
- Mojo interop benefits from modern Python
- Reasonable baseline for new AI/ML project

### 4. Tool Configurations

#### Black (Code Formatting)

**Status**: Not currently configured in existing pyproject.toml

**Recommendation**: Add if black is adopted, otherwise use ruff for formatting

#### Ruff (Linting)

**Decision**: Configure with 120-character line length, target Python 3.11

**Rationale**:
- Fast, modern linter that replaces multiple tools (flake8, isort, etc.)
- 120-character line length balances readability and modern displays
- Target version ensures linter uses appropriate Python features

#### Pytest (Testing)

**Decision**: Comprehensive test configuration with coverage reporting

**Configuration**:
- Test discovery: tests/ directory, test_*.py files
- Coverage: Track src/ directory, report missing lines
- Output formats: terminal, XML (for CI), HTML (for review)
- Verbose output for better debugging

**Rationale**:
- TDD is a core development principle (see CLAUDE.md)
- Coverage tracking ensures test quality
- Multiple report formats support both local and CI workflows

#### MyPy (Type Checking)

**Decision**: Strict type checking configuration

**Configuration**:
- Disallow untyped function definitions
- Warn on unused configs and return types
- Target Python 3.11

**Rationale**:
- Type safety is important for ML/AI code correctness
- Catches errors early in development
- Aligns with Mojo's type-safe philosophy

### 5. Package Discovery

**Decision**: Use setuptools package discovery with explicit include/exclude

**Configuration**:
- Include: src* directories
- Exclude: tests, notes, agents, papers, docs, logs, scripts

**Rationale**:
- Prevents accidentally packaging development artifacts
- Clear boundary between source code and supporting files
- Supports potential src-layout migration

## Architectural Considerations

### File Structure Philosophy

The pyproject.toml follows a progressive disclosure pattern:

1. **Build System** (lines 1-3): Foundation for packaging
2. **Project Metadata** (lines 5-21): What is ML Odyssey?
3. **Dependencies** (lines 16-38): What does it need?
4. **Tool Configurations** (lines 40+): How do we maintain quality?

### Integration Points

- **Pre-commit**: Ruff and mypy configurations will be referenced by .pre-commit-config.yaml
- **CI/CD**: pytest and coverage configurations drive GitHub Actions workflows
- **Documentation**: mkdocs dependencies support documentation generation
- **Security**: safety and bandit in dev dependencies for vulnerability scanning

### Future Extensibility

The current configuration supports:

- Adding new tool sections (e.g., [tool.black] if adopted)
- Extending dependency groups (e.g., [project.optional-dependencies.docs])
- Adding entry points when CLI tools are created
- Configuring package data and resources

## References

### Source Plans

- **Component Plan**: [notes/plan/01-foundation/02-configuration-files/02-pyproject-toml/plan.md](/home/mvillmow/ml-odyssey-manual/notes/plan/01-foundation/02-configuration-files/02-pyproject-toml/plan.md)
- **Child Plan 1**: [01-create-base-config/plan.md](/home/mvillmow/ml-odyssey-manual/notes/plan/01-foundation/02-configuration-files/02-pyproject-toml/01-create-base-config/plan.md)
- **Child Plan 2**: [02-add-python-deps/plan.md](/home/mvillmow/ml-odyssey-manual/notes/plan/01-foundation/02-configuration-files/02-pyproject-toml/02-add-python-deps/plan.md)
- **Child Plan 3**: [03-configure-tools/plan.md](/home/mvillmow/ml-odyssey-manual/notes/plan/01-foundation/02-configuration-files/02-pyproject-toml/03-configure-tools/plan.md)

### Related Issues

- Issue #637: [Test] Pyproject TOML - Validation tests
- Issue #638: [Implementation] Pyproject TOML - File creation and configuration
- Issue #639: [Package] Pyproject TOML - Integration with build system
- Issue #640: [Cleanup] Pyproject TOML - Refinement and documentation

### Comprehensive Documentation

- [CLAUDE.md](/home/mvillmow/ml-odyssey-manual/CLAUDE.md) - Project conventions and development principles
- [PEP 621](https://peps.python.org/pep-0621/) - Storing project metadata in pyproject.toml
- [PEP 518](https://peps.python.org/pep-0518/) - Specifying minimum build system requirements

### Current State

- **Status**: pyproject.toml already exists with comprehensive configuration
- **Location**: `/home/mvillmow/ml-odyssey-manual/pyproject.toml`
- **Completeness**: All major sections configured (build-system, project, dependencies, tools)

## Implementation Notes

### Current Analysis

The existing pyproject.toml (71 lines) is already well-structured and comprehensive:

**Strengths**:
- Complete project metadata (name, version, description, authors, license)
- Clear dependency separation (runtime vs. dev)
- Comprehensive pytest configuration with coverage tracking
- Modern tool configurations (ruff, mypy)
- Sensible defaults aligned with Python best practices

**Potential Enhancements** (for implementation phase):
- Consider adding [tool.ruff.lint] for more granular linting rules
- Evaluate if black should be added (currently using ruff for formatting)
- Add [project.urls] for repository, documentation, and issue tracker links
- Consider adding [project.scripts] for future CLI entry points

**Alignment with Plans**:
- Satisfies all three child plan objectives:
  - Base config with metadata ✓
  - Python dependencies organized ✓
  - Tool configurations present ✓

### Testing Considerations

The test phase (#637) should validate:
- TOML syntax validity
- PEP 621 compliance
- All referenced tools are in dependencies
- Configuration values are sensible (e.g., Python version compatibility)
- Package discovery excludes development artifacts

### Implementation Approach

Since pyproject.toml already exists, the implementation phase (#638) should:
1. Review current configuration against plan requirements
2. Identify any gaps or needed enhancements
3. Make minimal, focused changes to address gaps
4. Validate changes don't break existing functionality

### Packaging Integration

The packaging phase (#639) should verify:
- Build system can successfully create distributions
- Dependencies install correctly
- Tool configurations are honored
- Package metadata is correct in distributions
