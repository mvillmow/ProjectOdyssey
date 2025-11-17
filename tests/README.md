# Tests Directory

## Overview

This directory contains all test files for the ML Odyssey project, organized by component and test type.

## Structure

```text
tests/
├── foundation/        # Foundation tests
├── shared/           # Shared library tests
├── tools/            # Tooling tests
├── papers/           # Paper implementation tests
├── agents/           # Agent configuration tests
└── integration/      # End-to-end integration tests
```

## Quick Start

Run all tests:

```bash
python3 -m pytest tests/
```

Run specific test suite:

```bash
python3 -m pytest tests/foundation/
```

Run with coverage:

```bash
python3 -m pytest tests/ --cov=. --cov-report=html
```

## Usage

### Writing Tests

All tests follow TDD/FIRST principles:

- Fast: Tests should run quickly
- Independent: Tests should not depend on each other
- Repeatable: Tests should produce same results
- Self-Validating: Tests should have clear pass/fail
- Thorough: Tests should cover edge cases

### Test Organization

Tests are organized to mirror the source code structure:

- Unit tests: Test individual functions/classes
- Integration tests: Test module interactions
- System tests: Test end-to-end workflows

## Test Categories

### Foundation Tests

Tests for repository structure, configuration, and supporting directories.

### Shared Library Tests

Tests for core reusable components and utilities.

### Tools Tests

Tests for development and testing tools.

### Papers Tests

Tests for paper implementations and model training.

### Agent Tests

Tests for agent configurations and validation.

### Integration Tests

End-to-end tests validating complete workflows.

## Running Tests

### Basic Test Execution

```bash
# Run all tests
python3 -m pytest

# Run verbose
python3 -m pytest -v

# Run with warnings
python3 -m pytest -W default
```

### Coverage Reports

```bash
# Generate coverage report
python3 -m pytest --cov=. --cov-report=html

# View coverage
open htmlcov/index.html
```

### Test Selection

```bash
# Run specific file
python3 -m pytest tests/foundation/test_supporting_directories.py

# Run specific test
python3 -m pytest tests/foundation/test_supporting_directories.py::TestSupportingDirectoriesExistence::test_benchmarks_directory_exists

# Run tests matching pattern
python3 -m pytest -k "supporting"
```

## Test Standards

### Naming Conventions

- Test files: `test_*.py`
- Test classes: `Test*`
- Test methods: `test_*`

### Documentation

All test files must include:

- Module docstring describing test purpose
- Test category (unit/integration/system)
- Coverage target
- Class/method docstrings

### Assertions

Use descriptive assertion messages:

```python
assert result == expected, f"Expected {expected}, got {result}"
```

## CI/CD Integration

Tests are automatically run in CI/CD:

- On all pull requests
- Before merging to main
- During deployment

## Coverage Goals

- Unit tests: >95% coverage
- Integration tests: >80% coverage
- Overall: >90% coverage
