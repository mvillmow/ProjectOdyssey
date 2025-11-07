# Add Python Dependencies

## Overview
Add Python dependencies to pyproject.toml for development, testing, and optional features. This includes runtime dependencies, development tools, and optional dependencies for specific features.

## Parent Plan
[../plan.md](../plan.md)

## Child Plans
None (leaf node)

## Inputs
- pyproject.toml base file exists
- Knowledge of required Python packages
- Understanding of dependency groups

## Outputs
- pyproject.toml with dependencies section
- Runtime dependencies specified
- Optional dependency groups configured
- Development dependencies in appropriate groups

## Steps
1. Add runtime dependencies to [project.dependencies]
2. Create [project.optional-dependencies] for dev tools
3. Specify test dependencies in test group
4. Add documentation dependencies if needed

## Success Criteria
- [ ] All necessary Python dependencies are listed
- [ ] Dependencies are organized into appropriate groups
- [ ] Version constraints are specified where needed
- [ ] Optional dependencies are properly grouped

## Notes
Separate runtime dependencies from development dependencies. Use optional dependency groups for things like testing, docs, and development tools.
