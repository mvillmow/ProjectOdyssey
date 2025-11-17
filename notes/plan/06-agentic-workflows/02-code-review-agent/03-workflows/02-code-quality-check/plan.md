# Code Quality Check

## Overview

Create a workflow that runs automated code quality checks using static analysis tools, linters, and metrics calculators. The workflow provides objective quality measurements.

## Parent Plan

[../plan.md](../plan.md)

## Child Plans

None (implementation level)

## Inputs

- Codebase or specific files to check
- Configured analysis tools
- Quality thresholds and standards
- Baseline metrics for comparison

## Outputs

- Quality metrics report
- Tool outputs (linter, static analysis)
- Pass/fail status against thresholds
- Trend analysis if baseline exists
- Recommendations for improvements

## Steps

1. Run static analysis tools on code
2. Execute linters and type checkers
3. Calculate quality metrics
4. Compare against thresholds and baselines

## Success Criteria

- [ ] All configured tools execute successfully
- [ ] Metrics are accurately calculated
- [ ] Results are compared to thresholds
- [ ] Report is comprehensive and clear
- [ ] Trends are identified when possible
- [ ] Workflow handles tool failures gracefully

## Notes

Run tools in parallel for efficiency. Aggregate results into unified report. Include metrics like cyclomatic complexity, maintainability index, test coverage. Compare to project baselines. Flag regressions clearly.
