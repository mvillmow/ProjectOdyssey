# Coverage Gates

## Overview

Establish coverage quality gates that enforce minimum coverage standards in the CI pipeline. Gates prevent merging code that reduces coverage or falls below thresholds, maintaining quality standards. This ensures consistent test coverage across the project.

## Parent Plan

[../plan.md](../plan.md)

## Child Plans

None (leaf node)

## Inputs

- Coverage data and metrics
- Minimum coverage thresholds
- CI/CD pipeline configuration
- Exception rules for specific files

## Outputs

- CI checks that enforce coverage minimums
- Coverage comparison against main branch
- Failure conditions and error messages
- Coverage badge for repository
- Exception configuration for generated code

## Steps

1. Define minimum coverage thresholds
2. Configure CI to check coverage levels
3. Implement comparison against main branch
4. Set up clear failure messages
5. Add coverage badge to README

## Success Criteria

- [ ] CI fails when coverage drops below threshold
- [ ] PRs show coverage changes
- [ ] Thresholds are enforced automatically
- [ ] Exceptions work for valid cases

## Notes

Start with reasonable thresholds (e.g., 80%) and increase gradually. Fail on coverage decrease, not just absolute value. Provide clear messages explaining failures. Allow exceptions for generated or external code.
