# API Documenter

## Overview
Create a prompt template for generating API documentation and docstrings. The template analyzes code to produce comprehensive, accurate documentation following standard formats.

## Parent Plan
[../plan.md](../plan.md)

## Child Plans
None (implementation level)

## Inputs
- Code to document (functions, classes, modules)
- Documentation standards and format
- Understanding of docstring conventions
- Examples of good API documentation

## Outputs
- API documenter prompt template
- Docstring generation guidelines
- Structured output format with XML tags
- Few-shot examples
- Parameter and return type documentation

## Steps
1. Design structured output format for docstrings
2. Create prompt with documentation guidelines
3. Add few-shot examples of good docstrings
4. Define extraction rules for signatures

## Success Criteria
- [ ] Template generates complete docstrings
- [ ] Output follows documentation standards
- [ ] Parameters and returns are documented
- [ ] Examples are included when appropriate
- [ ] Type information is accurate
- [ ] Template handles various code patterns

## Notes
Follow Google or NumPy docstring style. Include sections: description, parameters (with types), returns (with type), raises, examples. Extract function signatures automatically. Use XML tags like <docstring>, <description>, <parameters>, <returns>.
