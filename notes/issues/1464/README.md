# Issue #1464: [Impl] Architecture Suggester - Implementation

## Objective

Implementation phase for Architecture Suggester.

## Phase

Implementation

## Labels

- `implementation`

## Deliverables

As specified in the issue description.

## Success Criteria

- [ ] All deliverables completed as specified
- [ ] Documentation updated
- [ ] Tests passing (if applicable)
- [ ] Code review completed
- [ ] Changes merged to main

## References

- Issue: <https://github.com/mvillmow/ml-odyssey/issues/1464>
- Related planning documentation in `/notes/plan/`
- Agent hierarchy: `/agents/hierarchy.md`

## Implementation Notes

## Overview

Create a prompt template for suggesting implementation architectures based on paper analysis. The template guides the agent through chain-of-thought reasoning to recommend appropriate designs, data structures, and module organization.

## Implementation Goals

- Implement the functionality to pass all tests
- Follow Mojo best practices and coding standards
- Ensure code is clean, documented, and maintainable
- Meet all requirements specified in the plan

## Required Inputs

- Paper analysis results
- Understanding of architecture design patterns
- Knowledge of Mojo/Python best practices
- Examples of good architecture suggestions

## Expected Outputs

- Architecture suggester prompt template
- Chain-of-thought reasoning structure
- Few-shot examples of architecture suggestions
- Module organization recommendations
- Data structure suggestions

## Implementation Steps

1. Create prompt with chain-of-thought structure
1. Define architecture reasoning steps
1. Add few-shot examples with rationale
1. Include Mojo-specific considerations

## Success Criteria

- [ ] Template uses chain-of-thought reasoning
- [ ] Suggestions consider paper requirements
- [ ] Module organization is clear and logical
- [ ] Data structures match problem needs
- [ ] Recommendations are Mojo-appropriate
- [ ] Few-shot examples show good designs

## Notes

Guide the agent through: understanding requirements, identifying key components, suggesting module structure, recommending data structures, and explaining design rationale. Use XML tags like <requirements>, <components>, <architecture>, <rationale>.

## Status

Created: 2025-11-16
Status: Pending
