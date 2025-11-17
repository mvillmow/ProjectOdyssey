# Coverage Reports

## Overview

Generate and format coverage reports to visualize test coverage and identify untested code. This includes console reports for quick feedback, HTML reports for detailed analysis, and summary statistics. Clear reports make coverage data actionable.

## Parent Plan

[../plan.md](../plan.md)

## Child Plans

None (leaf node)

## Inputs

- Coverage data from test runs
- Report format preferences
- Visualization requirements
- Output destinations (console, file, web)

## Outputs

- Console coverage summary
- HTML reports with line-by-line coverage
- Coverage statistics (percentage, missed lines)
- File-level and function-level breakdowns
- Historical coverage tracking

## Steps

1. Generate console summary reports
2. Create detailed HTML reports
3. Provide file and function breakdowns
4. Add uncovered line highlighting
5. Support historical coverage tracking

## Success Criteria

- [ ] Reports clearly show coverage levels
- [ ] Uncovered code is easy to identify
- [ ] Reports work in CI and locally
- [ ] Historical trends are trackable

## Notes

Console reports should be concise for quick feedback. HTML reports should allow drilling down to specific lines. Highlight uncovered code clearly. Track coverage over time to prevent regressions.
