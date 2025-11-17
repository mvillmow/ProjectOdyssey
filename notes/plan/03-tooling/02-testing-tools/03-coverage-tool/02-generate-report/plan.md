# Generate Report

## Overview

Generate coverage reports in multiple formats that clearly show which code is tested and which is not. Reports help developers identify gaps in test coverage and prioritize testing efforts.

## Parent Plan

[../plan.md](../plan.md)

## Child Plans

None (Level 4 - implementation level)

## Inputs

- Coverage data files
- Source code files
- Report format preferences

## Outputs

- HTML coverage report with highlighted code
- Text summary report
- Per-file coverage statistics
- Overall coverage percentage

## Steps

1. Load and parse coverage data
2. Calculate coverage percentages per file
3. Generate HTML report with source highlighting
4. Create text summary for console output

## Success Criteria

- [ ] Reports accurately reflect coverage data
- [ ] HTML report is easy to navigate
- [ ] Uncovered lines are clearly marked
- [ ] Summary shows key statistics

## Notes

Use color coding in reports - green for covered, red for uncovered. Sort files by coverage percentage to highlight problem areas. Include line numbers and code context in HTML reports.
