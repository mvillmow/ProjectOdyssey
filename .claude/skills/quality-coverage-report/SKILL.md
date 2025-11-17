---
name: quality-coverage-report
description: Generate test coverage reports showing which code paths are tested. Use to identify untested code and improve test coverage.
---

# Test Coverage Report Skill

Generate and analyze test coverage reports.

## When to Use

- After running tests
- Before creating PR
- Identifying untested code
- Improving test coverage

## Usage

### Python Coverage

```bash
# Run tests with coverage
pytest --cov=src --cov-report=html tests/

# View report
open htmlcov/index.html

# Terminal report
pytest --cov=src --cov-report=term-missing tests/
```

### Mojo Coverage (Future)

```bash
# When Mojo coverage tools available
mojo test --coverage tests/
```

## Coverage Metrics

### Line Coverage

Percentage of lines executed:

```text
src/module.mojo
  Lines: 45/50 (90%)
  Missing: 12, 18, 23, 35, 41
```

### Branch Coverage

Percentage of decision branches taken:

```text
Branches: 8/10 (80%)
Missing branches: 12->15, 18->20
```

## Coverage Goals

- **Minimum**: 80% line coverage
- **Target**: 90% line coverage
- **Critical paths**: 100% coverage
- **Edge cases**: Must be tested

## Coverage Report

```text
Coverage Report
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

File                Lines    Missing    Coverage
────────────────────────────────────────────────
src/tensor.mojo       150         5      96.7%
src/nn.mojo          200        30      85.0%
src/utils.mojo        50        10      80.0%
────────────────────────────────────────────────
TOTAL                400        45      88.8%

Critical paths: 100% ✅
Minimum coverage: 80% ✅
Target coverage: 90% ❌
```

## Improving Coverage

1. **Identify gaps**: Find uncovered lines
2. **Add tests**: Write tests for gaps
3. **Re-run**: Verify coverage improved
4. **Repeat**: Until targets met

## CI Integration

```yaml
- name: Test Coverage
  run: |
    pytest --cov=src --cov-report=xml
    codecov -f coverage.xml
```

See `phase-test-tdd` for test generation.
