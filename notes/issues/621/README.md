# Issue #621: [Plan] Create Base Config

## Objective

Create the basic pyproject.toml file with project metadata including name, version, description, authors, and basic
Python requirements. This establishes the foundation for Python project configuration following PEP 621 standards.

## Deliverables

- pyproject.toml file at repository root
- [project] section with metadata (name, version, description, authors)
- Basic build system configuration ([build-system] section)
- Minimum Python version requirement

## Success Criteria

- [ ] pyproject.toml file exists at repository root
- [ ] Project metadata is complete and accurate
- [ ] Build system is configured with appropriate backend
- [ ] File is valid TOML and follows PEP 621 standards
- [ ] Minimum Python version is specified

## Design Decisions

### 1. Build System Configuration

**Decision**: Use setuptools as the build backend

**Rationale**:

- Industry standard for Python packaging
- Well-documented and widely supported
- Compatible with modern PEP standards (PEP 517, PEP 518)
- Supports both pure Python and extension modules (future Mojo integration)

**Alternatives Considered**:

- Poetry: More opinionated, adds dependency management complexity
- Flit: Simpler but less flexible for complex projects
- Hatchling: Modern but less mature ecosystem

**Implementation**:

```toml
[build-system]
requires = ["setuptools>=65.0", "wheel"]
build-backend = "setuptools.build_meta"
```

### 2. Project Metadata Structure

**Decision**: Follow PEP 621 for all project metadata

**Rationale**:

- Standard format for Python projects
- Ensures compatibility with packaging tools
- Machine-readable format for automated processing
- Forward-compatible with ecosystem evolution

**Required Fields**:

- name: "ml-odyssey" (project identifier)
- version: "0.1.0" (semantic versioning)
- description: Clear one-line summary
- authors: List with name and email
- requires-python: Minimum Python version

### 3. Minimum Python Version

**Decision**: Require Python 3.11 or higher

**Rationale**:

- Mojo requires modern Python for interoperability
- Access to latest type hinting features
- Performance improvements in 3.11+
- Modern syntax and standard library features

**Trade-offs**:

- Excludes users on older Python versions
- Acceptable given project's focus on cutting-edge ML/AI

### 4. License Selection

**Decision**: Use BSD license

**Rationale**:

- Permissive license suitable for research code
- Compatible with academic and commercial use
- Allows code reuse with minimal restrictions
- Standard choice for ML research projects

### 5. Configuration Scope

**Decision**: Start with minimal valid configuration

**Rationale**:

- YAGNI principle - don't add until needed
- Dependencies will be added in subsequent issues (#622-625)
- Tool configurations handled separately (issue #627-630)
- Easier to review and validate incrementally

**What's Excluded from Base Config**:

- Python dependencies (handled in issue #626)
- Development dependencies (handled in issue #626)
- Tool configurations (pytest, ruff, mypy - handled in issue #627-630)

## Architecture

### File Structure

```text
pyproject.toml (root level)
├── [build-system]      # Build backend configuration
└── [project]           # Project metadata
    ├── name
    ├── version
    ├── description
    ├── readme
    ├── requires-python
    ├── license
    └── authors
```

### Standards Compliance

- **PEP 621**: Storing project metadata in pyproject.toml
- **PEP 517**: Build system interface
- **PEP 518**: Build system requirements
- **Semantic Versioning**: Version numbering (0.1.0)

### Integration Points

- Setuptools: Build and packaging
- Package managers: pip, uv, pixi
- CI/CD: Automated builds and tests
- IDEs: Project recognition and configuration

## Implementation Steps

1. Create pyproject.toml file at repository root
2. Add [build-system] section with setuptools configuration
3. Add [project] section with:
   - name: "ml-odyssey"
   - version: "0.1.0"
   - description: Project summary
   - authors: Developer information
   - readme: "README.md"
   - requires-python: ">=3.11"
   - license: BSD
4. Validate TOML syntax
5. Verify PEP 621 compliance

## Validation Requirements

### TOML Syntax Validation

```bash
# Validate TOML syntax
python3 -c "import tomllib; tomllib.load(open('pyproject.toml', 'rb'))"
```

### PEP 621 Compliance

- All required fields present
- Field types match specification
- Version follows semantic versioning
- Python version requirement is valid

### Build System Validation

```bash
# Verify build system works
pip install build
python -m build --sdist --wheel
```

## Current State Assessment

**Status**: pyproject.toml already exists with comprehensive configuration

**Findings**:

- Base configuration is complete and correct
- Includes build system configuration (setuptools)
- Project metadata follows PEP 621
- Minimum Python version: 3.11 (appropriate)
- License: BSD (appropriate for research)
- Already includes additional sections:
  - dependencies (should be in issue #626)
  - optional-dependencies (should be in issue #626)
  - tool configurations (should be in issues #627-630)

**Implications**:

- The "base config" scope has been exceeded
- Current file includes content from multiple planned issues
- May need to decide: keep comprehensive config or split it

**Recommendation**: Document that the base config is complete but includes additional content that was planned for
subsequent issues. This doesn't violate the plan but represents early implementation of dependent tasks.

## References

### Source Plan

- [notes/plan/01-foundation/02-configuration-files/02-pyproject-toml/01-create-base-config/plan.md](/home/mvillmow/ml-odyssey-manual/notes/plan/01-foundation/02-configuration-files/02-pyproject-toml/01-create-base-config/plan.md)

### Parent Plan

- [notes/plan/01-foundation/02-configuration-files/02-pyproject-toml/plan.md](/home/mvillmow/ml-odyssey-manual/notes/plan/01-foundation/02-configuration-files/02-pyproject-toml/plan.md)

### Related Issues

- #622: [Test] Create Base Config - Test suite for base configuration
- #623: [Impl] Create Base Config - Implementation of base configuration
- #624: [Package] Create Base Config - Integration and packaging
- #625: [Cleanup] Create Base Config - Refactoring and finalization
- #626: Add Python Dependencies (planned)
- #627-630: Configure Tools (planned)

### Standards References

- [PEP 621 - Storing project metadata in pyproject.toml](https://peps.python.org/pep-0621/)
- [PEP 517 - Build system interface](https://peps.python.org/pep-0517/)
- [PEP 518 - Build system requirements](https://peps.python.org/pep-0518/)
- [Semantic Versioning](https://semver.org/)

### Documentation

- [Python Packaging User Guide](https://packaging.python.org/)
- [Setuptools Documentation](https://setuptools.pypa.io/)
- [TOML Specification](https://toml.io/)

## Implementation Notes

### Planning Phase

**Status**: Complete

**Decisions Made**:

1. Use setuptools as build backend (standard choice)
2. Require Python 3.11+ (compatible with Mojo)
3. Use BSD license (appropriate for research)
4. Start with minimal valid configuration (YAGNI)
5. Follow PEP 621 for all metadata

**Current State**: pyproject.toml exists with comprehensive configuration that exceeds the minimal base config scope.
The file includes dependencies and tool configurations that were planned for subsequent issues, representing early
implementation of the full pyproject.toml configuration.

**Next Steps**: Testing phase (#622) should validate the existing configuration rather than creating new content.
