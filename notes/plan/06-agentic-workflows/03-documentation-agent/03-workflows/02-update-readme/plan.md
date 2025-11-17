# Update README

## Overview

Create a workflow that analyzes projects and generates or updates README files with comprehensive information. The workflow examines code structure, dependencies, and existing documentation to produce complete READMEs.

## Parent Plan

[../plan.md](../plan.md)

## Child Plans

None (implementation level)

## Inputs

- Project structure and files
- README generator prompt template
- Existing README (if any)
- Documentation standards

## Outputs

- Generated or updated README file
- All required sections populated
- Installation instructions
- Usage examples
- API overview
- Validation report

## Steps

1. Analyze project structure and dependencies
2. Extract information from code and docs
3. Generate README content using template
4. Validate completeness and accuracy

## Success Criteria

- [ ] README contains all required sections
- [ ] Installation instructions are accurate
- [ ] Usage examples work correctly
- [ ] API overview is comprehensive
- [ ] Existing content is preserved when appropriate
- [ ] Workflow handles various project types

## Notes

Analyze project structure to determine type and features. Extract dependencies from configuration files. Generate installation instructions based on package manager. Create usage examples from existing code or tests. Preserve user-written content when updating.
