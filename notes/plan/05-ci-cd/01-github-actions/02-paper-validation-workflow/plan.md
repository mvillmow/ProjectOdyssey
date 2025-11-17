# Paper Validation Workflow

## Overview

Create a GitHub Actions workflow that validates paper implementations for structural correctness, required files, and reproducibility. This ensures all paper contributions meet repository standards.

## Parent Plan

[Parent](../plan.md)

## Child Plans

- [01-detect-paper-changes](./01-detect-paper-changes/plan.md)
- [02-validate-structure](./02-validate-structure/plan.md)
- [03-run-reproduction](./03-run-reproduction/plan.md)

## Inputs

- Detect changes to paper implementations in pull requests
- Validate paper directory structure and required files
- Run reproduction tests to verify paper results
- Provide clear feedback on validation failures
- Ensure consistency across all paper implementations

## Outputs

- Completed paper validation workflow
- Detect changes to paper implementations in pull requests (completed)

## Steps

1. Detect Paper Changes
2. Validate Structure
3. Run Reproduction

## Success Criteria

- [ ] Workflow triggers when papers directory changes
- [ ] Identifies which papers were modified
- [ ] Validates all required files present
- [ ] Checks README and documentation completeness
- [ ] Runs reproduction scripts successfully
- [ ] Reports validation errors clearly

## Notes

- Only validate papers that changed in the PR
- Use paper scaffolding validation tools
- Check for README, implementation, tests, and documentation
- Verify reproduction scripts exist and run
- Keep validation fast (under 10 minutes)
