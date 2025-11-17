# Architecture Suggester

## Overview

Create a prompt template for suggesting implementation architectures based on paper analysis. The template guides the agent through chain-of-thought reasoning to recommend appropriate designs, data structures, and module organization.

## Parent Plan

[../plan.md](../plan.md)

## Child Plans

None (implementation level)

## Inputs

- Paper analysis results
- Understanding of architecture design patterns
- Knowledge of Mojo/Python best practices
- Examples of good architecture suggestions

## Outputs

- Architecture suggester prompt template
- Chain-of-thought reasoning structure
- Few-shot examples of architecture suggestions
- Module organization recommendations
- Data structure suggestions

## Steps

1. Create prompt with chain-of-thought structure
2. Define architecture reasoning steps
3. Add few-shot examples with rationale
4. Include Mojo-specific considerations

## Success Criteria

- [ ] Template uses chain-of-thought reasoning
- [ ] Suggestions consider paper requirements
- [ ] Module organization is clear and logical
- [ ] Data structures match problem needs
- [ ] Recommendations are Mojo-appropriate
- [ ] Few-shot examples show good designs

## Notes

Guide the agent through: understanding requirements, identifying key components, suggesting module structure, recommending data structures, and explaining design rationale. Use XML tags like <requirements>, <components>, <architecture>, <rationale>.
