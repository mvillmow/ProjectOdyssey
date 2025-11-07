# Correctness Reviewer

## Overview
Create a prompt template for reviewing code correctness, focusing on identifying bugs, logic errors, edge cases, and algorithmic issues. The template guides systematic analysis of code behavior.

## Parent Plan
[../plan.md](../plan.md)

## Child Plans
None (implementation level)

## Inputs
- Correctness review criteria
- Understanding of common bug patterns
- Knowledge of edge case testing
- Examples of good correctness reviews

## Outputs
- Correctness reviewer prompt template
- Bug detection checklist
- Structured output format with XML tags
- Severity rating guidelines
- Fix suggestion patterns

## Steps
1. Design structured output format for issues
2. Create checklist for systematic review
3. Add bug pattern examples
4. Define severity rating criteria

## Success Criteria
- [ ] Template detects common bug types
- [ ] Edge cases are systematically checked
- [ ] Logic errors are identified
- [ ] Output includes severity ratings
- [ ] Suggestions include fix examples
- [ ] Template produces consistent results

## Notes
Focus on: null/undefined handling, boundary conditions, type errors, logic bugs, resource leaks, concurrency issues. Use XML tags like <bug>, <location>, <severity>, <fix>. Include examples of bugs to catch.
