# Report Results

## Overview
Generate comprehensive test reports showing successes, failures, and statistics. Reports provide clear visibility into test results with detailed information about failures to help developers debug issues quickly.

## Parent Plan
[../plan.md](../plan.md)

## Child Plans
None (Level 4 - implementation level)

## Inputs
- Test execution results
- Test output and error messages
- Execution timing information

## Outputs
- Formatted test report
- Summary statistics
- Detailed failure information
- Exit code for CI/CD integration

## Steps
1. Aggregate test results from execution
2. Format summary with counts and percentages
3. Display detailed failure information
4. Generate report in multiple formats (console, file)

## Success Criteria
- [ ] Reports clearly show pass/fail status
- [ ] Failure details help identify issues
- [ ] Summary statistics are accurate
- [ ] Output works in CI/CD environments

## Notes
Use colors for console output to highlight failures. Show most important information first. Include helpful context like file paths and line numbers. Support JSON output for tooling integration.
