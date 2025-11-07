# Verify Dependencies

## Overview
Check that all required project dependencies are installed and meet version requirements. This ensures the development environment has everything needed for the project.

## Parent Plan
[../plan.md](../plan.md)

## Child Plans
None (Level 4 - implementation level)

## Inputs
- Dependency specifications
- Required version constraints
- Python environment

## Outputs
- Dependency status report
- List of missing dependencies
- Version mismatches
- Installation recommendations

## Steps
1. Load dependency requirements from project files
2. Check each dependency is installed
3. Verify versions meet requirements
4. Report missing or incompatible dependencies

## Success Criteria
- [ ] All dependencies are checked
- [ ] Version compatibility is validated
- [ ] Missing dependencies are identified
- [ ] Clear installation guidance provided

## Notes
Check both Python packages and system libraries. Support checking pixi environment. Suggest commands to install missing dependencies. Handle optional dependencies gracefully.
