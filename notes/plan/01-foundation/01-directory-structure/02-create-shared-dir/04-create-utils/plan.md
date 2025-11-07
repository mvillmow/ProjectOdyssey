# Create Utils

## Overview
Create the shared/utils/ directory for general utility functions and helper code that will be reused across paper implementations. This includes logging, configuration, file I/O, visualization, and other general-purpose utilities.

## Parent Plan
[../plan.md](../plan.md)

## Child Plans
None (leaf node)

## Inputs
- shared/ directory exists
- Understanding of common utility needs
- Knowledge of helper functions used across papers

## Outputs
- shared/utils/ directory
- shared/utils/README.md explaining purpose
- __init__.py for Python package structure

## Steps
1. Create utils/ subdirectory inside shared/
2. Write README explaining what utilities belong here
3. Create __init__.py to make it a Python package
4. Document expected contents and organization

## Success Criteria
- [ ] utils/ directory exists in shared/
- [ ] README clearly explains purpose and contents
- [ ] Directory is set up as a proper Python package
- [ ] Documentation guides what utility code is shared

## Notes
Utils is for general-purpose helper functions that don't fit in core, training, or data. Include logging, config management, visualization, file I/O, and other utilities used across papers.
