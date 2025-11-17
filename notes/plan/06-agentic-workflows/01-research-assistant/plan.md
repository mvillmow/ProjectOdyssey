# Research Assistant

## Overview

Create an AI research assistant that helps analyze academic papers, suggest architectures, and review implementations. The assistant follows Claude best practices with structured prompts, clear role definitions, chain-of-thought reasoning, and appropriate tool use for code analysis.

## Parent Plan

[../plan.md](../plan.md)

## Child Plans

- [01-agent-configuration/plan.md](01-agent-configuration/plan.md)
- [02-prompt-templates/plan.md](02-prompt-templates/plan.md)
- [03-workflows/plan.md](03-workflows/plan.md)
- [04-testing/plan.md](04-testing/plan.md)

## Inputs

- Claude best practices for prompt engineering
- Understanding of research paper analysis requirements
- Knowledge of code generation and architecture design patterns
- Existing repository structure for paper implementations

## Outputs

- Agent configuration file with role definition and tool setup
- Prompt templates for paper analysis, architecture suggestions, and implementation review
- Workflows for paper-to-code, code review, and debugging assistance
- Test suite covering all agent capabilities
- Documentation for using the research assistant

## Steps

1. Configure agent with clear role, constraints, and available tools
2. Create prompt templates for different research tasks
3. Implement workflows that chain prompts and tools together
4. Write comprehensive tests for all agent capabilities

## Success Criteria

- [ ] Agent configuration defines clear role and constraints
- [ ] Prompt templates use XML tags and few-shot examples
- [ ] Workflows implement chain-of-thought reasoning
- [ ] Agent can analyze papers and extract key information
- [ ] Agent can suggest appropriate architectures
- [ ] Agent can review implementations for correctness
- [ ] All tests pass with good coverage

## Notes

Follow Claude best practices: use clear role definitions, structured prompts with XML tags, few-shot examples for consistency, chain-of-thought for complex reasoning, and appropriate tool use for code analysis. Keep implementations simple and focused.
