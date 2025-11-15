# Issue #118: [Plan] Configure Tools - Design and Documentation

## Objective

Configure development tools in pyproject.toml including code formatters (black), linters (ruff), type checkers (mypy),
and test runners (pytest). These configurations ensure consistent code quality across the project.

## Deliverables

- Black formatter configuration
- Ruff linter configuration
- MyPy type checker configuration
- Pytest testing framework configuration
- Tool integration strategy
- Documentation of configuration choices

## Success Criteria

- [ ] All development tools are configured
- [ ] Configurations are consistent with team standards
- [ ] Tool settings are documented with comments
- [ ] Configurations enable good development practices

## References

- Source Plan: `/notes/plan/01-foundation/02-configuration-files/02-pyproject-toml/03-configure-tools/plan.md`
- Parent Component: #123 (Pyproject TOML)
- Related Issues: #119 (Test), #120 (Impl), #121 (Package), #122 (Cleanup)
- Current pyproject.toml: `/pyproject.toml`

## Implementation Notes

### Current State Analysis

The existing `pyproject.toml` already contains partial tool configurations:

- **Pytest**: Fully configured with test paths, coverage, and reporting options
- **Coverage**: Configured with source paths and reporting precision
- **Ruff**: Basic configuration exists (line-length: 120, target-version: py311)
- **MyPy**: Basic configuration exists (python_version: 3.11, basic type checking flags)
- **Black**: NOT YET CONFIGURED

### Configuration Strategy

Following the principle of "sensible defaults" from the plan notes:

1. **Start with minimal, practical configurations**
2. **Document all non-standard choices**
3. **Allow for tightening as project matures**
4. **Ensure consistency across tools** (e.g., all use 120 char line length)

## Design Decisions

### Black Configuration

**Purpose**: Consistent code formatting across the project

**Recommended Configuration**:

```toml
[tool.black]
line-length = 120
target-version = ['py311']
include = '\.pyi?$'
extend-exclude = '''
/(
  # directories
  \.git
  | \.mypy_cache
  | \.tox
  | \.venv
  | _build
  | buck-out
  | build
  | dist
)/
'''
```

**Rationale**:

- **line-length = 120**: Matches Ruff configuration for consistency
- **target-version = py311**: Aligns with project's minimum Python version
- **extend-exclude**: Excludes common build/cache directories
- **include pattern**: Only Python files (.py, .pyi)

### Ruff Configuration

**Purpose**: Fast Python linter and code quality checker

**Current State**: Basic configuration exists

**Recommended Enhancements**:

```toml
[tool.ruff]
line-length = 120
target-version = "py311"
select = [
    "E",   # pycodestyle errors
    "W",   # pycodestyle warnings
    "F",   # pyflakes
    "I",   # isort
    "N",   # pep8-naming
    "UP",  # pyupgrade
    "B",   # flake8-bugbear
    "C4",  # flake8-comprehensions
    "SIM", # flake8-simplify
]
ignore = [
    "E501",  # line too long (handled by black)
]

[tool.ruff.per-file-ignores]
"tests/*" = ["S101"]  # Allow assert in tests

[tool.ruff.isort]
known-first-party = ["src"]
```

**Rationale**:

- **Selected rules**: Start with essential quality checks, not overly strict
- **E501 ignored**: Black handles line length, avoid duplicate warnings
- **Test exemptions**: Allow assert statements in test files
- **isort integration**: Automated import sorting

### MyPy Configuration

**Purpose**: Static type checking for Python code

**Current State**: Basic configuration exists

**Recommended Enhancements**:

```toml
[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
warn_redundant_casts = true
warn_unused_ignores = true
warn_no_return = true
check_untyped_defs = true
strict_optional = true

# Per-module options
[[tool.mypy.overrides]]
module = "tests.*"
disallow_untyped_defs = false
```

**Rationale**:

- **Core strictness**: Maintain existing `disallow_untyped_defs` requirement
- **Additional warnings**: Catch common type-related issues early
- **Test exemptions**: Allow untyped test functions for flexibility
- **Gradual adoption**: Can tighten with `strict = true` later if needed

### Pytest Configuration

**Purpose**: Test framework and test execution

**Current State**: FULLY CONFIGURED - already includes:

- Test discovery paths and patterns
- Coverage reporting (term, xml, html)
- Verbose output
- Coverage source and omit patterns

**Recommended Status**: KEEP AS-IS

**Rationale**:

- Current configuration is comprehensive
- Follows best practices (coverage reporting, verbose output)
- No changes needed for this phase

**Current Configuration**:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
python_classes = "Test*"
python_functions = "test_*"
addopts = [
    "--verbose",
    "--cov=src",
    "--cov-report=term-missing",
    "--cov-report=xml",
    "--cov-report=html",
]

[tool.coverage.run]
source = ["src"]
omit = ["tests/*", "**/__pycache__/*"]

[tool.coverage.report]
precision = 2
show_missing = true
skip_covered = false
```

### Tool Integration Strategy

**Pre-commit Hook Integration**:

All tools should be integrated into pre-commit hooks:

1. **Black**: Auto-format on commit
2. **Ruff**: Lint check on commit
3. **MyPy**: Type check on commit (optional, can be slow)
4. **Pytest**: Run tests in CI only (too slow for pre-commit)

**CI/CD Integration**:

All tools should run in CI pipeline:

1. **Black**: Check formatting (fail if not formatted)
2. **Ruff**: Check linting (fail on errors)
3. **MyPy**: Check types (fail on type errors)
4. **Pytest**: Run full test suite with coverage

**Developer Workflow**:

```bash
# Format code
black src/ tests/

# Check linting
ruff check src/ tests/

# Check types
mypy src/

# Run tests
pytest
```

## Configuration Principles

Following project guidelines from CLAUDE.md:

1. **KISS** (Keep It Simple Stupid): Start with minimal configs, add complexity only when needed
2. **YAGNI** (You Ain't Gonna Need It): Don't add tool rules until they solve real problems
3. **Consistency**: All tools use 120 character line length
4. **Python 3.11+**: All tools target project's minimum Python version
5. **Gradual Tightening**: Start permissive, tighten as project matures

## Next Steps

After this planning phase is approved:

1. **Issue #119** (Test): Create test cases to validate tool configurations
2. **Issue #120** (Implementation): Add/update tool configurations in pyproject.toml
3. **Issue #121** (Packaging): Integrate tools into pre-commit and CI/CD
4. **Issue #122** (Cleanup): Review and optimize configurations

## Dependencies

**Blocks**:

- #119 (Test Configure Tools)
- #120 (Implement Configure Tools)
- #121 (Package Configure Tools)
- #122 (Cleanup Configure Tools)

**Depends On**:

- Existing pyproject.toml structure
- Development dependencies already defined in `[project.optional-dependencies]`

## Notes

- Black is NOT currently configured - this is the main addition needed
- Ruff and MyPy have minimal configs - these should be enhanced
- Pytest configuration is comprehensive and should be preserved
- All configurations should maintain 120 character line length for consistency
- Tool settings should be documented with inline comments in pyproject.toml
