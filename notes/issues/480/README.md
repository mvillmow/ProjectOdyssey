# Issue #480: [Impl] Coverage Reports - Core Implementation

## Objective

Implement coverage reporting functionality that generates console and HTML reports from coverage data, providing clear visualization of test coverage and identifying untested code.

## Deliverables

- Console report generator
- HTML report generator
- Coverage statistics calculator
- Historical tracking implementation
- Report configuration options
- Integration with coverage collection (Issue #475)

## Success Criteria

- [ ] Console reports display coverage summary clearly
- [ ] HTML reports provide line-by-line coverage view
- [ ] Coverage statistics are accurate
- [ ] Historical tracking stores and retrieves data
- [ ] Reports integrate with CI pipeline
- [ ] All tests from Issue #479 pass

## References

### Parent Issues

- [Issue #478: [Plan] Coverage Reports](../478/README.md) - Design and architecture
- [Issue #479: [Test] Coverage Reports](../479/README.md) - Test specifications

### Related Issues

- [Issue #481: [Package] Coverage Reports](../481/README.md) - Packaging
- [Issue #482: [Cleanup] Coverage Reports](../482/README.md) - Cleanup

### Dependencies

- [Issue #473-477: Setup Coverage](../473/README.md) - Coverage data must be collected first

### Comprehensive Documentation

- [5-Phase Workflow](/home/user/ml-odyssey/notes/review/README.md)
- [Agent Hierarchy](/home/user/ml-odyssey/agents/hierarchy.md)

## Implementation Notes

### Minimal Implementation Strategy

**Leverage Existing Tools**: Python's coverage.py already provides report generation. Our implementation focuses on:

1. **Configuration** - Not new code
2. **Integration** - Connecting to CI
3. **Customization** - Project-specific needs

**What NOT to Build**:
- ❌ New report generator (coverage.py has this)
- ❌ HTML renderer (coverage.py has this)
- ❌ Coverage calculation (coverage.py does this)

**What to Build**:
- ✅ Configuration for project needs
- ✅ CI integration scripts
- ✅ Custom report filtering (if needed)
- ✅ Historical tracking wrapper (optional)

### Using coverage.py Reports

**Console Reports** (Already works):

```bash
# Generate console report
pytest --cov=scripts --cov-report=term-missing

# Output example:
# ---------- coverage: platform linux, python 3.11 ----------
# Name                  Stmts   Miss  Cover   Missing
# ---------------------------------------------------
# scripts/__init__.py       0      0   100%
# scripts/utils.py         45      5    89%   23, 45-48
# ---------------------------------------------------
# TOTAL                    45      5    89%
```

**HTML Reports** (Already works):

```bash
# Generate HTML report
pytest --cov=scripts --cov-report=html

# Creates htmlcov/ directory with:
# - index.html (file listing)
# - *.html (per-file coverage)
# - Built-in styling and highlighting
```

**XML Reports** (For CI integration):

```bash
# Generate Cobertura XML
pytest --cov=scripts --cov-report=xml

# Creates coverage.xml for CI tools
```

### Implementation Tasks

**1. Configure Report Outputs**

Update `pyproject.toml`:

```toml
[tool.pytest.ini_options]
addopts = [
    "--cov=scripts",
    "--cov-report=term-missing:skip-covered",  # Console: hide 100% covered files
    "--cov-report=html",                        # HTML: detailed reports
    "--cov-report=xml",                         # XML: for CI/Codecov
]

[tool.coverage.report]
precision = 2              # Show XX.XX%
show_missing = true        # Show line numbers
skip_covered = false       # Show all files (or true to hide 100%)
fail_under = 80            # Fail if coverage < 80%

[tool.coverage.html]
directory = "htmlcov"      # Output directory
title = "ML Odyssey Coverage Report"
```

**2. CI Integration**

Update `.github/workflows/test.yml`:

```yaml
- name: Run Tests with Coverage
  run: |
    pytest --cov=scripts \
           --cov-report=html \
           --cov-report=xml \
           --cov-report=term

- name: Upload HTML Coverage Report
  uses: actions/upload-artifact@v3
  with:
    name: coverage-html-report
    path: htmlcov/

- name: Upload Coverage to Codecov (Optional)
  uses: codecov/codecov-action@v3
  with:
    file: ./coverage.xml
    fail_ci_if_error: false
```

**3. Historical Tracking (Optional)**

If needed, create simple tracking script:

```python
# scripts/track_coverage.py
"""Store coverage snapshots for historical tracking."""

import json
from pathlib import Path
from datetime import datetime
from coverage import Coverage

def store_coverage_snapshot():
    """Store current coverage as historical snapshot."""
    cov = Coverage()
    cov.load()

    snapshot = {
        "timestamp": datetime.now().isoformat(),
        "commit": get_git_commit(),  # Get current commit hash
        "total_coverage": cov.report(),
        "file_coverage": {
            file: cov.analysis(file)[0]  # Get coverage %
            for file in cov.get_data().measured_files()
        }
    }

    # Append to historical data
    history_file = Path(".coverage_history.json")
    history = json.loads(history_file.read_text()) if history_file.exists() else []
    history.append(snapshot)
    history_file.write_text(json.dumps(history, indent=2))

def get_git_commit():
    """Get current git commit hash."""
    import subprocess
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True
    )
    return result.stdout.strip()
```

**4. Custom Report Filtering (If Needed)**

Only implement if default coverage.py reports don't meet needs:

```python
# scripts/filter_coverage_report.py
"""Filter coverage reports to exclude generated files."""

from coverage import Coverage

def generate_filtered_report():
    """Generate coverage report excluding generated files."""
    cov = Coverage()
    cov.load()

    # Exclude generated files
    for file in cov.get_data().measured_files():
        if "_generated" in file or "_pb2.py" in file:
            cov.get_data().erase(file)

    # Generate filtered report
    cov.html_report()
```

### Files to Create/Modify

**Configuration** (Primary work):
- `pyproject.toml` - Update coverage report settings
- `.github/workflows/test.yml` - Add coverage reporting steps

**Scripts** (Optional, only if needed):
- `scripts/track_coverage.py` - Historical tracking (if not using Codecov)
- `scripts/filter_coverage_report.py` - Custom filtering (if needed)

**Documentation**:
- Update from Issue #476 with report usage examples

### Validation Checklist

Before completing implementation:

- [ ] Console report displays correctly
- [ ] HTML report opens and shows coverage
- [ ] XML report validates against Cobertura schema
- [ ] CI uploads coverage artifacts successfully
- [ ] Reports match expected coverage percentages
- [ ] Performance is acceptable (< 10 seconds for full report)

### Status

Created: 2025-11-19
Status: Pending implementation
Dependencies: Issue #478 (Plan), #479 (Test), and #473-477 (Setup) must be completed first
