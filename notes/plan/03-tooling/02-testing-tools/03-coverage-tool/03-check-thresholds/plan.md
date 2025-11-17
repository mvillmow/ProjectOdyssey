# Check Thresholds

## Overview

Validate coverage percentages against configured minimum thresholds. This ensures the repository maintains adequate test coverage and prevents merging code with insufficient tests.

## Parent Plan

[../plan.md](../plan.md)

## Child Plans

None (Level 4 - implementation level)

## Inputs

- Coverage data and percentages
- Threshold configuration
- Per-file and overall coverage targets

## Outputs

- Threshold validation results
- Files failing threshold requirements
- CI/CD exit code (pass/fail)
- Recommendations for improvement

## Steps

1. Load threshold configuration
2. Compare actual coverage to thresholds
3. Identify files and areas below threshold
4. Generate validation report with pass/fail

## Success Criteria

- [ ] Thresholds are configurable per project
- [ ] Validation accurately checks coverage
- [ ] Clear report shows threshold violations
- [ ] Exit code supports CI/CD integration

## Notes

Support both overall and per-file thresholds. Allow threshold configuration in project file. Provide grace period for new files. Make threshold failures block CI/CD builds to enforce quality.
