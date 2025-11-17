# Coverage

## Overview

Implement code coverage tracking to measure test completeness. This includes setting up coverage tools, generating coverage reports showing tested and untested code, and establishing coverage gates to maintain quality standards. Coverage metrics help identify gaps in testing.

## Parent Plan

[../plan.md](../plan.md)

## Child Plans

- [01-setup-coverage/plan.md](01-setup-coverage/plan.md)
- [02-coverage-reports/plan.md](02-coverage-reports/plan.md)
- [03-coverage-gates/plan.md](03-coverage-gates/plan.md)

## Inputs

- Test suite and test framework
- Coverage tool configuration
- Quality thresholds for coverage

## Outputs

- Configured coverage tracking
- Coverage reports showing test completeness
- Coverage gates enforcing quality standards

## Steps

1. Set up coverage tools and integration with test framework
2. Generate coverage reports for visibility
3. Establish coverage gates to prevent regressions

## Success Criteria

- [ ] Coverage tracking captures all test execution
- [ ] Reports clearly show covered and uncovered code
- [ ] Gates prevent merging code that reduces coverage
- [ ] All child plans are completed successfully

## Notes

Use coverage as a guide, not an absolute metric. Focus on meaningful coverage over high percentages. Ensure reports are easy to understand and actionable.
