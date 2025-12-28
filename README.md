# ML Odyssey

Description here.

[![Mojo](https://img.shields.io/badge/Mojo-0.26+-orange.svg)](https://www.modular.com/mojo)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-122%2B-brightgreen.svg)](tests/)
[![Coverage](https://img.shields.io/badge/coverage-pending-lightgrey.svg)](#coverage-status)

## Features

Features list.

## Getting Started

Quick start guide.

## Installation

Installation steps.

## Coverage Status

⚠️ **Full code coverage metrics are blocked** by [Mojo coverage tooling availability](docs/adr/ADR-008-coverage-tool-blocker.md).

### Current Workarounds

- ✅ **Test Discovery Validation**: All `test_*.mojo` files verified in CI
- ✅ **Test Metrics Tracking**: 122+ test files, 500+ test functions tracked
- ✅ **Manual Code Review**: PR checklist requires test coverage verification
- ✅ **Python Script Coverage**: 70%+ threshold for automation scripts

### When Mojo Coverage Available

Expected workflow once Mojo provides coverage tooling:

```bash
mojo test --coverage tests/
mojo coverage report --format=lcov > coverage.lcov
```

See [ADR-008](docs/adr/ADR-008-coverage-tool-blocker.md) for complete explanation.
