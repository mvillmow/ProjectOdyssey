# Paper Analyzer

## Overview
Create a prompt template for analyzing research papers and extracting key information like problem statement, methodology, architecture, results, and implementation details. The template uses structured XML tags and few-shot examples.

## Parent Plan
[../plan.md](../plan.md)

## Child Plans
None (implementation level)

## Inputs
- Understanding of paper analysis requirements
- Knowledge of typical paper structure
- XML tag patterns for structured output
- Examples of good paper analyses

## Outputs
- Paper analyzer prompt template
- Few-shot examples of paper analysis
- Structured output format with XML tags
- Guidelines for extracting key information

## Steps
1. Design structured output format with XML tags
2. Create prompt with clear analysis instructions
3. Add 2-3 few-shot examples
4. Define extraction guidelines for each section

## Success Criteria
- [ ] Template extracts all key paper sections
- [ ] Output uses consistent XML structure
- [ ] Few-shot examples demonstrate expected quality
- [ ] Template works for various paper types
- [ ] Analysis includes implementation-relevant details

## Notes
Focus on extracting information useful for implementation: problem formulation, mathematical foundations, architecture details, hyperparameters, and training procedures. Use XML tags like <problem>, <method>, <architecture>, <results>.
