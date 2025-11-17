# PR Review

## Overview

Create a workflow that performs comprehensive pull request reviews by combining correctness, performance, and style checks. The workflow produces a unified review report with prioritized findings.

## Parent Plan

[../plan.md](../plan.md)

## Child Plans

None (implementation level)

## Inputs

- Pull request diff or changed files
- All review prompt templates
- Code analysis tools
- Review criteria

## Outputs

- Comprehensive review report
- Findings organized by category and severity
- Line-specific comments for issues
- Overall assessment and recommendations
- Approval or changes requested status

## Steps

1. Analyze changed files in the PR
2. Run correctness, performance, and style reviews
3. Aggregate findings and prioritize issues
4. Generate unified review report

## Success Criteria

- [ ] Workflow reviews all changed files
- [ ] All review types are performed
- [ ] Issues are properly categorized
- [ ] Severity levels are assigned correctly
- [ ] Report is clear and actionable
- [ ] Workflow handles large PRs efficiently

## Notes

Run different review types in parallel when possible. Aggregate results into a single coherent report. Prioritize critical issues. Include positive feedback for good code. Provide specific line-level comments.
