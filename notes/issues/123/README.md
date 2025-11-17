# Issue #123: [Plan] Pyproject TOML - Design and Documentation

## Objective

Plan Python project configuration strategy for ml-odyssey repository.

## Planning Complete

**Why Complete:**
The repository uses a dual-file strategy for Python dependencies per ADR-001:

1. **`/pyproject.toml`** (71 lines) - Test framework and development tooling:
   - Build system: setuptools>=65.0
   - Core dependencies: pytest suite (pytest, pytest-cov, pytest-timeout, pytest-xdist)
   - Dev dependencies: pre-commit, safety, bandit, mkdocs, ruff, mypy
   - Tool configurations: pytest, coverage, ruff, mypy settings

2. **`/tools/requirements.txt`** (23 lines) - Automation script dependencies:
   - jinja2, pyyaml, click (template engine, config parsing, CLI framework)
   - Optional: matplotlib, pandas (for benchmarking reports)

**Design Decision (ADR-001):**

- **Python for tooling only** - automation scripts with subprocess output capture, regex, GitHub API
- **Mojo for ML/AI** - all machine learning implementations
- **Clear separation** - tools/requirements.txt for scripts, pyproject.toml for dev/test

**Success Criteria:**

- ✅ Dual-file strategy documented and justified (ADR-001)
- ✅ pyproject.toml exists with comprehensive configuration
- ✅ tools/requirements.txt exists for automation dependencies
- ✅ Clear separation between test deps and tool deps

**References:**

- `/pyproject.toml:1-71` (complete Python project configuration)
- `/tools/requirements.txt:1-23` (automation script dependencies)
- `/notes/review/adr/ADR-001-language-selection-tooling.md` (design decision)
