# Pyproject TOML

## Overview

Create and configure the pyproject.toml file for Python project configuration. This file defines Python package metadata, dependencies, and tool configurations for linting, formatting, and testing.

## Parent Plan

[../plan.md](../plan.md)

## Child Plans

- [01-create-base-config/plan.md](01-create-base-config/plan.md)
- [02-add-python-deps/plan.md](02-add-python-deps/plan.md)
- [03-configure-tools/plan.md](03-configure-tools/plan.md)

## Inputs

- Repository root directory exists
- Understanding of Python project configuration
- Knowledge of development tools needed

## Outputs

- pyproject.toml file at repository root
- Project metadata configured
- Python dependencies specified
- Tool configurations for black, ruff, pytest, etc.

## Steps

1. Create base pyproject.toml with project metadata
2. Add Python dependencies for development and testing
3. Configure development tools like formatters and linters

## Success Criteria

- [ ] pyproject.toml file exists and is valid
- [ ] Project metadata is complete
- [ ] Python dependencies are specified
- [ ] Development tools are configured
- [ ] File follows Python packaging standards

## Notes

pyproject.toml is the standard for Python project configuration. Keep tool configurations simple and use sensible defaults. Document any non-standard choices.
