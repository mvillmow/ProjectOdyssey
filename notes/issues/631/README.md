# Issue #631: [Plan] Configure Tools - Design and Documentation

## Objective

Define comprehensive specifications for configuring development tools in pyproject.toml, including code formatters
(black), linters (ruff), type checkers (mypy), and test runners (pytest). This planning phase establishes the
architectural approach and design decisions for ensuring consistent code quality across the ML Odyssey project.

## Deliverables

- **[tool.black] configuration** - Code formatting settings for consistent Python style
- **[tool.ruff] configuration** - Linting rules and import sorting configuration
- **[tool.pytest] configuration** - Test discovery and execution settings
- **[tool.mypy] configuration** - Type checking rules and strictness levels

## Success Criteria

- [ ] All development tools are configured with sensible defaults
- [ ] Configurations are consistent with team coding standards (see CLAUDE.md)
- [ ] Tool settings are documented with inline comments explaining rationale
- [ ] Configurations enable good development practices without being overly strict
- [ ] Design decisions are documented with alternatives considered
- [ ] Configuration approach is validated against Python packaging standards (PEP 517/518)

## Design Decisions

### 1. Tool Selection Rationale

### Black (Code Formatter)

- **Decision**: Use Black for Python code formatting
- **Rationale**: Industry standard, opinionated formatter that eliminates style debates
- **Configuration Strategy**: Minimal configuration, rely on Black's defaults
- **Key Settings**:
  - Line length: 120 characters (matches ruff, accommodates modern displays)
  - Target Python version: 3.11+ (aligns with project requirements)

### Ruff (Linter)

- **Decision**: Use Ruff as primary linter (replaces flake8, isort, and several other tools)
- **Rationale**:
  - Written in Rust, significantly faster than traditional Python linters
  - Consolidates multiple tools (flake8, isort, pydocstyle, etc.) into one
  - Already included in project dependencies
- **Configuration Strategy**: Start with recommended rules, expand as needed
- **Key Settings**:
  - Line length: 120 characters (consistent with Black)
  - Target version: py311 (matches project Python requirement)
  - Select rules: Start conservative, enable more as project matures

### Pytest (Test Runner)

- **Decision**: Use pytest as test framework
- **Rationale**:
  - Most popular Python testing framework
  - Already configured and in use (see existing pyproject.toml)
- **Configuration Strategy**: Enhance existing configuration for better coverage
- **Key Settings**:
  - Test discovery paths: tests/
  - Coverage reporting: term-missing, xml, html
  - Verbose output for better debugging

### Mypy (Type Checker)

- **Decision**: Use mypy for static type checking
- **Rationale**:
  - Industry standard for Python type checking
  - Already included in dev dependencies
  - Helps catch errors before runtime
- **Configuration Strategy**: Start with moderate strictness, increase over time
- **Key Settings**:
  - Python version: 3.11
  - Strictness: Moderate (warn on return types, unused configs, untyped defs)
  - Incremental checking enabled for performance

### 2. Configuration Philosophy

### Progressive Strictness

- **Principle**: Start with reasonable defaults, tighten as project matures
- **Rationale**: Overly strict initial settings can hinder rapid development
- **Implementation**: Document which settings can be made stricter later
- **Examples**:
  - Ruff: Start with core rules, add specific checks as patterns emerge
  - Mypy: Enable `disallow_untyped_defs` but defer stricter options

### Consistency Across Tools

- **Line Length**: 120 characters across all tools (Black, Ruff, Mypy)
- **Python Version**: 3.11 minimum across all tools
- **Rationale**: Prevents conflicts between tools, reduces cognitive load

### Documentation-First

- **Principle**: Every non-default setting must have an inline comment explaining why
- **Rationale**: Future maintainers need to understand configuration decisions
- **Implementation**: Use TOML comments to document each configuration section

### 3. Integration with Existing Infrastructure

### Pre-commit Hooks

- Tools must integrate with existing pre-commit setup
- Ruff already configured in `.pre-commit-config.yaml`
- Consider adding Black and Mypy to pre-commit hooks

### CI/CD Pipeline

- Tool configurations must work in CI environment
- Test coverage reports must be compatible with CI tooling
- Consider GitHub Actions integration

### Developer Experience

- Configurations should provide clear, actionable error messages
- Fast execution to avoid slowing down development workflow
- Editor integration (VS Code, PyCharm) should work out of box

### 4. Alternatives Considered

### Black vs. Autopep8/YAPF

- **Rejected**: Autopep8 and YAPF require more configuration decisions
- **Chosen**: Black's opinionated approach reduces bikeshedding

### Ruff vs. Flake8 + isort + pydocstyle

- **Rejected**: Multiple tools increase complexity and slow execution
- **Chosen**: Ruff consolidates functionality with better performance

### Mypy vs. Pyright

- **Evaluated**: Pyright is faster and Microsoft-backed
- **Chosen**: Mypy for broader ecosystem support and familiarity
- **Future**: Could reconsider if Pyright adoption increases

### Pytest vs. unittest

- **Rejected**: unittest is more verbose, less flexible
- **Chosen**: pytest for superior fixtures, parametrization, and plugins

### 5. Configuration Sections Breakdown

**[tool.black]**:

```toml
[tool.black]
line-length = 120              # Matches ruff, modern display standard
target-version = ["py311"]     # Minimum Python version for project
```text

**[tool.ruff]**:

```toml
[tool.ruff]
line-length = 120              # Consistent with Black
target-version = "py311"       # Minimum Python version

[tool.ruff.lint]
select = ["E", "F", "I"]       # Errors, Pyflakes, Import sorting
ignore = []                    # Start permissive, add as needed

[tool.ruff.lint.isort]
known-first-party = ["src"]    # Local package for import sorting
```text

**[tool.pytest.ini_options]** (already configured):

- Current configuration is comprehensive
- Consider adding `--strict-markers` for better test organization
- Consider adding `--tb=short` for more readable tracebacks

**[tool.mypy]** (already configured):

- Current configuration is reasonable
- Consider adding `strict = true` in future iterations
- Consider adding per-module overrides for gradual adoption

### 6. Documentation Requirements

### Inline Comments

- Every configuration section must have a comment explaining its purpose
- Non-default values must explain the rationale
- Reference relevant PEPs or documentation where applicable

### External Documentation

- Link to tool documentation in comments where complex
- Reference CLAUDE.md for coding standards alignment
- Document how to run tools locally and in CI

### 7. Migration and Rollout Strategy

**Phase 1 (Current)**: Planning and Design

- Document configuration approach
- Review existing configurations
- Identify gaps and improvements

**Phase 2**: Implementation

- Add/update tool configurations in pyproject.toml
- Test configurations locally
- Verify compatibility with pre-commit and CI

**Phase 3**: Validation

- Run tools against existing codebase
- Fix any issues uncovered
- Document any exceptions or waivers

**Phase 4**: Integration

- Update CI/CD pipelines to use new configurations
- Update developer documentation
- Train team on new tooling

## References

### Source Plan

- [notes/plan/01-foundation/02-configuration-files/02-pyproject-toml/03-configure-tools/plan.md](notes/plan/01-foundation/02-configuration-files/02-pyproject-toml/03-configure-tools/plan.md)

### Parent Plan

- [notes/plan/01-foundation/02-configuration-files/02-pyproject-toml/plan.md](notes/plan/01-foundation/02-configuration-files/02-pyproject-toml/plan.md)

### Related Issues

- #632 - [Test] Configure Tools - Test Implementation
- #633 - [Implementation] Configure Tools - Core Implementation
- #634 - [Package] Configure Tools - Integration and Packaging
- #635 - [Cleanup] Configure Tools - Refactor and Finalize

### Project Documentation

- [CLAUDE.md](CLAUDE.md) - Project coding standards and principles
- [Python Coding Standards](CLAUDE.md#python-coding-standards)
- [Pre-commit Hooks](CLAUDE.md#pre-commit-hooks)

### Tool Documentation

- [Black Documentation](https://black.readthedocs.io/)
- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [Pytest Documentation](https://docs.pytest.org/)
- [Mypy Documentation](https://mypy.readthedocs.io/)
- [PEP 517 - Build System](https://peps.python.org/pep-0517/)
- [PEP 518 - pyproject.toml](https://peps.python.org/pep-0518/)

### Current State

- [pyproject.toml](pyproject.toml) - Existing configuration file
- [.pre-commit-config.yaml](.pre-commit-config.yaml) - Pre-commit hook configuration

## Implementation Notes

This section will be populated during the implementation phase with:

- Discoveries made during configuration
- Issues encountered and resolutions
- Performance observations
- Team feedback on tooling
- Any deviations from planned approach and rationale

---

**Status**: Planning Complete
**Phase**: Planning → Test/Implementation/Packaging (parallel phases can begin)
**Next Steps**: Implementation team can proceed with #633 based on this specification
