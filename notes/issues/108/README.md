# Issue #108: [Plan] Create Base Config - Design and Documentation

## Objective

Create the basic pyproject.toml file with project metadata including name, version, description, authors, and basic Python requirements. This establishes the foundation for Python project configuration.

## Deliverables

- Detailed specifications for basic pyproject.toml structure
- Project metadata schema (name, version, description, authors)
- Python version requirements
- Build system configuration
- Documentation of design decisions

## Success Criteria

- [ ] pyproject.toml file exists at repository root
- [ ] Project metadata is complete and accurate
- [ ] Build system is configured
- [ ] File is valid TOML and follows PEP standards
- [ ] Design decisions are documented

## References

- Source Plan: `/notes/plan/01-foundation/02-configuration-files/02-pyproject-toml/01-create-base-config/plan.md`
- Parent Component: #123 (Pyproject TOML - Design and Documentation)
- Related Issues: #109 (Test), #110 (Impl), #111 (Package), #112 (Cleanup)
- PEP 621: [Storing project metadata in pyproject.toml](https://peps.python.org/pep-0621/)
- PEP 518: [Specifying Minimum Build System Requirements](https://peps.python.org/pep-0518/)

## Implementation Notes

[To be filled during implementation]

## Design Decisions

### Project Metadata Structure

The pyproject.toml will follow PEP 621 standards with the following metadata fields:

**Required Fields:**

- `name` = "ml-odyssey" - Project name following PyPI naming conventions
- `version` = "0.1.0" - Semantic versioning starting at 0.1.0 for initial development
- `description` = "Mojo-based AI research platform for reproducing classic ML papers"
- `authors` = List of author objects with name and email

**Optional Fields (for initial version):**

- `readme` = "README.md" - Link to project README
- `requires-python` = ">=3.7" - Minimum Python version requirement
- `license` = {text = "MIT"} or similar - To be determined
- `keywords` = ["mojo", "machine-learning", "ai", "research"] - Project keywords

**Structure:**

```toml
[project]
name = "ml-odyssey"
version = "0.1.0"
description = "Mojo-based AI research platform for reproducing classic ML papers"
authors = [
    {name = "Author Name", email = "author@example.com"}
]
readme = "README.md"
requires-python = ">=3.7"
keywords = ["mojo", "machine-learning", "ai", "research"]
```

### Build System Selection

**Decision: Use `setuptools` with `setuptools-scm` for version management**

**Rationale:**

- **setuptools**: Most widely adopted and stable Python build backend
- **Mature ecosystem**: Extensive documentation and community support
- **Compatibility**: Works seamlessly with pip and standard Python tooling
- **Future-proof**: Can be migrated to hatchling or other backends later if needed

**Configuration:**

```toml
[build-system]
requires = ["setuptools>=61.0", "setuptools-scm"]
build-backend = "setuptools.build_meta"
```

**Alternatives Considered:**

- **hatchling**: Modern, simpler configuration, but less mature
- **poetry**: Opinionated dependency management, but adds complexity
- **flit**: Minimal approach, but may lack features as project grows

**Decision Factors:**

1. **Stability**: setuptools is battle-tested in production environments
2. **Flexibility**: Easy to extend with custom build steps if needed
3. **Migration Path**: Can switch backends later without major refactoring
4. **Team Familiarity**: Most Python developers know setuptools

### Python Version Strategy

**Decision: Minimum Python 3.7, Target Python 3.11+**

**Rationale:**

- **Python 3.7**: End of life June 2023, but provides wide compatibility
- **Python 3.11**: Current stable version with performance improvements
- **Future-looking**: Avoid bleeding-edge to ensure stability

**Compatibility Approach:**

1. **Minimum viable**: Set `requires-python = ">=3.7"` initially
2. **Testing**: Test on Python 3.7, 3.9, 3.11 in CI/CD
3. **Migration**: Plan to bump to 3.9 minimum after initial release
4. **Type hints**: Use Python 3.7-compatible type hints (no PEP 604 union syntax)

**Version Constraint Syntax:**

```toml
requires-python = ">=3.7"  # Minimum version
```

**Future Considerations:**

- Python 3.13+ introduces performance improvements for ML workloads
- May want to add `<4.0` upper bound for Python 4 compatibility
- Monitor Python release schedule for EOL dates

### File Structure and Organization

**Location:** Repository root (`/pyproject.toml`)

**Section Order (PEP 621 recommended):**

1. `[build-system]` - Build requirements (PEP 518)
2. `[project]` - Core metadata (PEP 621)
3. `[project.optional-dependencies]` - Development dependencies (future)
4. `[tool.*]` - Tool-specific configurations (future)

**Validation:**

- Must be valid TOML syntax
- Must pass `pip install -e .` validation
- Should pass `pyproject-validate` or similar linting

### Initial Configuration Template

```toml
# Build system configuration
[build-system]
requires = ["setuptools>=61.0", "setuptools-scm"]
build-backend = "setuptools.build_meta"

# Project metadata
[project]
name = "ml-odyssey"
version = "0.1.0"
description = "Mojo-based AI research platform for reproducing classic ML papers"
authors = [
    {name = "Project Author", email = "author@example.com"}
]
readme = "README.md"
requires-python = ">=3.7"
keywords = ["mojo", "machine-learning", "ai", "research", "neural-networks"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
]
```

### Extensibility Considerations

**Future Additions (Not in Initial Version):**

1. **Dependencies**: Will be added in #113 (Add Python Deps)
2. **Tool Configurations**: Will be added in #114 (Configure Tools)
3. **Scripts/Entry Points**: To be determined based on CLI needs
4. **URLs**: Project homepage, repository, documentation links

**Design Principle:**
Start with minimal valid configuration and expand incrementally. Each addition should:

- Have clear purpose and justification
- Follow PEP standards
- Be documented in subsequent issues

### Validation Strategy

**Pre-commit Checks:**

1. TOML syntax validation (`check-toml` hook)
2. File format validation (trailing whitespace, EOF newline)

**Manual Validation:**

```bash
# Verify TOML syntax
python3 -c "import tomllib; tomllib.load(open('pyproject.toml', 'rb'))"

# Verify pip can read it
pip install -e . --dry-run

# Validate against PEP 621 (optional tool)
pyproject-validate pyproject.toml
```

**CI/CD Validation:**

- Include in pre-commit CI workflow
- Test installation on multiple Python versions

## Next Steps

After this planning issue (#108) is completed:

1. **#109 [Test]**: Write validation tests for pyproject.toml structure
2. **#110 [Impl]**: Create the actual pyproject.toml file with specified configuration
3. **#111 [Package]**: Integrate with CI/CD and pre-commit hooks
4. **#112 [Cleanup]**: Review and finalize configuration, remove any temporary settings

## Questions for Review

1. Should we include license information in the initial version?
2. Do we need project URLs (homepage, repository) in the base config?
3. Should we use setuptools-scm for version management or manual versioning?
4. Are there any additional classifiers we should include?

## Status

- [ ] Design decisions documented
- [ ] Template structure defined
- [ ] Validation strategy planned
- [ ] Related issues updated with plan reference
