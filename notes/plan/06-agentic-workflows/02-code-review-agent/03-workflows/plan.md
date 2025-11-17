# Workflows

## Overview

Create end-to-end workflows that chain together review templates and tools to perform comprehensive code reviews. Workflows include PR review, code quality checks, and improvement suggestions.

## Parent Plan

[../plan.md](../plan.md)

## Child Plans

- [01-pr-review/plan.md](01-pr-review/plan.md)
- [02-code-quality-check/plan.md](02-code-quality-check/plan.md)
- [03-suggest-improvements/plan.md](03-suggest-improvements/plan.md)

## Inputs

- Prompt templates (correctness, performance, style reviewers)
- Configured code analysis tools
- Understanding of review workflow patterns
- Examples of effective review processes

## Outputs

- PR review workflow implementation
- Code quality check workflow implementation
- Improvement suggestion workflow implementation
- Workflow orchestration logic
- Error handling and reporting

## Steps

1. Create PR review workflow combining all review types
2. Create code quality check workflow with automated tools
3. Create improvement suggestion workflow with prioritization

## Success Criteria

- [ ] Workflows integrate all review types
- [ ] Each workflow handles errors gracefully
- [ ] Workflows produce actionable reports
- [ ] Reviews are thorough and consistent
- [ ] Workflows can run independently or together

## Notes

Design workflows to be composable. Run different review types in parallel when possible. Aggregate results into unified reports. Prioritize issues by severity. Provide clear, actionable feedback.
