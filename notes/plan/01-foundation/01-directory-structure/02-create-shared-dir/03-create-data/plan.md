# Create Data

## Overview

Create the shared/data/ directory for data processing and dataset utilities that will be reused across paper implementations. This includes data loaders, preprocessing functions, augmentation utilities, and dataset classes.

## Parent Plan

[../plan.md](../plan.md)

## Child Plans

None (leaf node)

## Inputs

- shared/ directory exists
- Understanding of common data processing needs
- Knowledge of datasets used across papers

## Outputs

- shared/data/ directory
- shared/data/README.md explaining purpose
- __init__.py for Python package structure

## Steps

1. Create data/ subdirectory inside shared/
2. Write README explaining what data utilities belong here
3. Create __init__.py to make it a Python package
4. Document expected contents and organization

## Success Criteria

- [ ] data/ directory exists in shared/
- [ ] README clearly explains purpose and contents
- [ ] Directory is set up as a proper Python package
- [ ] Documentation guides what data code is shared

## Notes

Data directory is for reusable data processing code. Include dataset classes, data loaders, preprocessing, augmentation, and other data-related utilities that multiple papers need.
