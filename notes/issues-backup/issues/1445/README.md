# Issue #1445: [Package] Define Role Constraints - Integration and Packaging

## Objective

Package phase for Define Role Constraints.

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

- Issue: <https://github.com/mvillmow/ml-odyssey/issues/1445>
- Related planning documentation in `/notes/plan/`
- Agent hierarchy: `/agents/hierarchy.md`

## Implementation Notes

## Overview

Define the research assistant's role, purpose, and behavioral constraints. This includes the agent's primary responsibilities, limitations, and guidelines for interaction following Claude best practices.

## Packaging Objectives

- Integrate the implementation with existing codebase
- Ensure all dependencies are properly configured
- Verify compatibility with other components
- Package for deployment/distribution

## Integration Requirements

Based on outputs:

- Clear role definition with purpose statement
- Behavioral constraints and limitations
- Interaction guidelines
- Examples of appropriate/inappropriate behavior

## Integration Steps

1. Write clear role definition describing agent purpose
1. Define behavioral constraints to prevent issues
1. Add guidelines for appropriate interactions
1. Include examples of good and bad behavior

## Success Criteria

- [ ] Role definition is clear and specific
- [ ] Constraints prevent hallucination and inappropriate responses
- [ ] Guidelines cover common interaction scenarios
- [ ] Examples illustrate expected behavior
- [ ] Definition follows Claude best practices

## Notes

Use structured prompts with XML tags. Be specific about what the agent should and should not do. Focus on being helpful for research tasks while maintaining accuracy and not hallucinating.

## Status

Created: 2025-11-16
Status: Pending
