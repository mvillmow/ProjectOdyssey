# Detect Regressions

## Overview
Analyze benchmark comparisons to detect performance regressions based on configured thresholds. This automated detection helps prevent performance degradation from being merged.

## Parent Plan
[../plan.md](../plan.md)

## Child Plans
None (Level 4 - implementation level)

## Inputs
- Comparison results
- Regression thresholds
- Significance criteria

## Outputs
- Detected regressions list
- Severity assessment
- Regression report
- Exit code for CI/CD

## Steps
1. Apply regression thresholds to comparisons
2. Identify metrics that exceed thresholds
3. Assess regression severity
4. Generate regression report

## Success Criteria
- [ ] Regressions are detected accurately
- [ ] Thresholds are configurable
- [ ] Severity is assessed correctly
- [ ] Reports guide remediation

## Notes
Configure thresholds per metric (e.g., 10% slowdown is regression). Distinguish between warning and critical regressions. Ignore noise and focus on significant changes. Provide context to help understand regressions.
