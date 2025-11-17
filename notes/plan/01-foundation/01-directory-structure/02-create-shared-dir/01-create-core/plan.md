# Create Core

## Overview

Create the shared/core/ directory for fundamental building blocks and core functionality that will be used across all paper implementations. This includes basic layers, operations, and essential components.

## Parent Plan

[../plan.md](../plan.md)

## Child Plans

None (leaf node)

## Inputs

- shared/ directory exists
- Understanding of core components needed across papers
- Knowledge of fundamental ML building blocks

## Outputs

- shared/core/ directory
- shared/core/README.md explaining purpose
- __init__.py for Python package structure

## Steps

1. Create core/ subdirectory inside shared/
2. Write README explaining what goes in core/
3. Create __init__.py to make it a Python package
4. Document expected contents and organization

## Success Criteria

- [ ] core/ directory exists in shared/
- [ ] README clearly explains purpose and contents
- [ ] Directory is set up as a proper Python package
- [ ] Documentation helps contributors know what belongs here

## Notes

Core is for fundamental, low-level components that many papers will need. Keep it focused on truly shared functionality, not paper-specific code.
