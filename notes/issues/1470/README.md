# Issue #1470: [Package] Implementation Reviewer - Integration and Packaging

## Objective

Package phase for Implementation Reviewer.

## Phase

Package

## Labels

- `packaging`
- `integration`

## Deliverables

As specified in the issue description.

## Success Criteria

- [ ] All deliverables completed as specified
- [ ] Documentation updated
- [ ] Tests passing (if applicable)
- [ ] Code review completed
- [ ] Changes merged to main

## References

- Issue: <https://github.com/mvillmow/ml-odyssey/issues/1470>
- Related planning documentation in `/notes/plan/`
- Agent hierarchy: `/agents/hierarchy.md`

## Implementation Notes

## Overview

Create a prompt template for reviewing code implementations against paper specifications. The template helps identify correctness issues, missing components, and discrepancies between paper and code.

## Packaging Objectives

- Integrate the implementation with existing codebase
- Ensure all dependencies are properly configured
- Verify compatibility with other components
- Package for deployment/distribution

## Integration Requirements

Based on outputs:

- Implementation reviewer prompt template
- Review checklist with criteria
- Few-shot examples of reviews
- Structured feedback format
- Suggestion templates for improvements

## Integration Steps

1. Define review criteria and checklist
1. Create structured feedback format
1. Add few-shot review examples
1. Include improvement suggestion patterns

## Success Criteria

- [ ] Template covers all review criteria
- [ ] Feedback format is clear and actionable
- [ ] Checklist ensures comprehensive review
- [ ] Few-shot examples show thorough reviews
- [ ] Suggestions are specific and helpful
- [ ] Template identifies both issues and strengths

## Notes

Review criteria should include: correctness of algorithm implementation, completeness of required components, adherence to paper specifications, code quality, and potential bugs. Use XML tags like <correctness>, <completeness>, <issues>, <suggestions>.

## Status

Created: 2025-11-16
Status: Pending
