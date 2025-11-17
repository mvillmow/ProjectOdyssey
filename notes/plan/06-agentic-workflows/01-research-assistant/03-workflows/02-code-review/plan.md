# Code Review

## Overview

Create a workflow that reviews implementation code against paper specifications. The workflow reads the paper and code, compares them, and provides detailed feedback on correctness and completeness.

## Parent Plan

[../plan.md](../plan.md)

## Child Plans

None (implementation level)

## Inputs

- Research paper specifications
- Implementation code
- Paper analyzer prompt template
- Implementation reviewer prompt template

## Outputs

- Comparison of paper and implementation
- Identified issues and discrepancies
- Suggestions for improvements
- Review report with structured feedback
- Priority ratings for issues

## Steps

1. Analyze paper to extract specifications
2. Review code implementation
3. Compare implementation to specifications
4. Generate structured feedback

## Success Criteria

- [ ] Workflow identifies all major discrepancies
- [ ] Feedback is specific and actionable
- [ ] Issues are prioritized appropriately
- [ ] Suggestions include code examples
- [ ] Review covers correctness and completeness
- [ ] Workflow handles partial implementations

## Notes

Use tools to read both paper and code files. Chain the paper analyzer and implementation reviewer templates. Focus on algorithmic correctness - does the code implement what the paper describes? Provide constructive feedback with examples.
