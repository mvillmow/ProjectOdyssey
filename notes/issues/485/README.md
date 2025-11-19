# Issue #485: [Impl] Coverage Gates - Core Implementation

## Objective

Implement coverage quality gates that enforce minimum coverage standards in CI, preventing code that reduces coverage or falls below thresholds from being merged.

## Deliverables

- Threshold enforcement logic
- Coverage regression detection
- Exception configuration system
- Failure message generation
- CI integration scripts
- Gate configuration

## Success Criteria

- [ ] CI fails when coverage < threshold
- [ ] CI fails on coverage regression
- [ ] Exceptions exclude specified files
- [ ] Failure messages are clear
- [ ] Gates integrate with GitHub Actions
- [ ] All tests from Issue #484 pass

## References

### Parent Issues

- [Issue #483: [Plan] Coverage Gates](../483/README.md) - Design and architecture
- [Issue #484: [Test] Coverage Gates](../484/README.md) - Test specifications

### Related Issues

- [Issue #486: [Package] Coverage Gates](../486/README.md) - Packaging
- [Issue #487: [Cleanup] Coverage Gates](../487/README.md) - Cleanup

### Dependencies

- [Issue #478-482: Coverage Reports](../478/README.md) - Must have coverage data

### Comprehensive Documentation

- [5-Phase Workflow](/home/user/ml-odyssey/notes/review/README.md)
- [Agent Hierarchy](/home/user/ml-odyssey/agents/hierarchy.md)

## Implementation Notes

### Minimal Implementation Strategy

**Leverage pytest-cov built-in features** instead of building custom gates:

```bash
# pytest-cov has --cov-fail-under flag
pytest --cov=scripts --cov-fail-under=80
```

This automatically fails if coverage < 80%.

**What NOT to Build**:
- ❌ Custom threshold checker (pytest-cov has this)
- ❌ Complex gate orchestration (CI does this)
- ❌ Coverage comparison engine (coverage.py has this)

**What to Build**:
- ✅ CI workflow configuration
- ✅ Exception patterns configuration
- ✅ Custom failure messages (optional enhancement)
- ✅ Regression detection script (simple comparison)

### Implementation Tasks

**1. Configure Threshold in pyproject.toml**

```toml
[tool.coverage.report]
precision = 2
show_missing = true
fail_under = 80.0  # Fail if total coverage < 80%

# Per-metric thresholds (if supported by coverage.py version)
[tool.coverage.run]
branch = true
source = ["scripts"]
omit = [
    "tests/*",
    "**/__pycache__/*",
    "**/vendor/*",
    "**/*_pb2.py",      # Generated protobuf
    "**/__generated__/*" # Generated code
]
```

**2. CI Workflow Integration**

Update `.github/workflows/test.yml`:

```yaml
name: Tests with Coverage Gates

on: [push, pull_request]

jobs:
  test-with-coverage:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -r requirements-dev.txt

      - name: Run tests with coverage
        run: |
          pytest --cov=scripts \
                 --cov-report=xml \
                 --cov-report=html \
                 --cov-report=term \
                 --cov-fail-under=80

      - name: Check coverage regression (if main branch available)
        if: github.event_name == 'pull_request'
        run: |
          python scripts/check_coverage_regression.py \
            --current coverage.xml \
            --baseline main-coverage.xml \
            --max-decrease 2.0

      - name: Upload coverage reports
        if: always()
        uses: actions/upload-artifact@v3
        with:
          name: coverage-reports
          path: |
            htmlcov/
            coverage.xml

      - name: Comment PR with coverage
        if: github.event_name == 'pull_request'
        uses: py-cov-action/python-coverage-comment-action@v3
        with:
          GITHUB_TOKEN: ${{ github.token }}
          MINIMUM_GREEN: 90
          MINIMUM_ORANGE: 80
```

**3. Regression Detection Script**

Create `scripts/check_coverage_regression.py`:

```python
#!/usr/bin/env python3
"""
Check for coverage regression against baseline.

Usage:
    python check_coverage_regression.py \
        --current coverage.xml \
        --baseline main-coverage.xml \
        --max-decrease 2.0
"""

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def parse_coverage_xml(xml_path: Path) -> float:
    """Parse coverage percentage from Cobertura XML."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    # Cobertura format: <coverage line-rate="0.85" ...>
    line_rate = float(root.attrib.get('line-rate', 0))
    return line_rate * 100  # Convert to percentage


def main():
    parser = argparse.ArgumentParser(description='Check coverage regression')
    parser.add_argument('--current', type=Path, required=True,
                        help='Current coverage XML file')
    parser.add_argument('--baseline', type=Path, required=True,
                        help='Baseline coverage XML file')
    parser.add_argument('--max-decrease', type=float, default=2.0,
                        help='Maximum allowed coverage decrease (%)')

    args = parser.parse_args()

    if not args.current.exists():
        print(f"❌ Current coverage file not found: {args.current}")
        sys.exit(1)

    if not args.baseline.exists():
        print(f"⚠️  Baseline coverage file not found: {args.baseline}")
        print("   Skipping regression check (first run?)")
        sys.exit(0)

    current_coverage = parse_coverage_xml(args.current)
    baseline_coverage = parse_coverage_xml(args.baseline)
    delta = current_coverage - baseline_coverage

    print(f"Current coverage:  {current_coverage:.2f}%")
    print(f"Baseline coverage: {baseline_coverage:.2f}%")
    print(f"Delta:             {delta:+.2f}%")

    if delta < -args.max_decrease:
        print(f"\n❌ Coverage regression detected!")
        print(f"   Coverage decreased by {abs(delta):.2f}%")
        print(f"   Maximum allowed decrease: {args.max_decrease}%")
        print(f"\n   Please add tests to restore coverage.")
        sys.exit(1)
    elif delta < 0:
        print(f"\n⚠️  Coverage decreased by {abs(delta):.2f}%")
        print(f"   Within tolerance ({args.max_decrease}%), but consider adding tests.")
        sys.exit(0)
    else:
        print(f"\n✅ Coverage check passed!")
        if delta > 0:
            print(f"   Coverage improved by {delta:.2f}%")
        sys.exit(0)


if __name__ == '__main__':
    main()
```

**4. Store Baseline Coverage**

Create workflow to store main branch coverage:

```yaml
# .github/workflows/store-baseline-coverage.yml
name: Store Baseline Coverage

on:
  push:
    branches: [main]

jobs:
  store-coverage:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: pip install -r requirements-dev.txt

      - name: Generate coverage
        run: pytest --cov=scripts --cov-report=xml

      - name: Upload baseline coverage
        uses: actions/upload-artifact@v3
        with:
          name: main-coverage
          path: coverage.xml
          retention-days: 30
```

**5. Exception Configuration**

Already handled in `pyproject.toml` via `omit` patterns:

```toml
[tool.coverage.run]
omit = [
    "tests/*",           # Test files
    "**/__pycache__/*",  # Compiled Python
    "**/vendor/*",       # Third-party code
    "**/*_pb2.py",       # Generated protobuf
    "**/__generated__/*" # Any generated code
]
```

### Files to Create/Modify

**New Files**:
- `scripts/check_coverage_regression.py` - Regression detection
- `.github/workflows/store-baseline-coverage.yml` - Store main coverage

**Modified Files**:
- `pyproject.toml` - Add `fail_under` threshold
- `.github/workflows/test.yml` - Add coverage gates

**Configuration**:
- Update `omit` patterns in pyproject.toml for exceptions

### Validation Checklist

Test the implementation:

```bash
# Test threshold enforcement
pytest --cov=scripts --cov-fail-under=80
# Should fail if coverage < 80%

# Test regression detection
python scripts/check_coverage_regression.py \
  --current coverage.xml \
  --baseline coverage-baseline.xml \
  --max-decrease 2.0

# Test in CI (locally)
act -j test-with-coverage
```

Expected behaviors:
- [ ] Fails when coverage < 80%
- [ ] Fails when regression > 2%
- [ ] Passes when coverage meets requirements
- [ ] Clear error messages on failure
- [ ] Proper exit codes (0 = pass, 1 = fail)

### Open Questions

- [ ] Should gates be blocking immediately or advisory first?
- [ ] What's the right threshold to start? (80% recommended)
- [ ] Should we integrate with Codecov/Coveralls for badges?
- [ ] How to handle coverage for entirely new files?

### Status

Created: 2025-11-19
Status: Pending implementation
Dependencies: Issue #483 (Plan), #484 (Test), and #478-482 (Reports) must be completed first
