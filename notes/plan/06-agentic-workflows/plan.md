# Agentic Workflows

## Overview
Build AI assistants that follow Claude best practices for prompt engineering to support research, code review, and documentation tasks. This section implements three specialized agents: a research assistant for paper analysis and code generation, a code review agent for quality assurance, and a documentation agent for maintaining comprehensive project documentation.

## Parent Plan
None (top-level)

## Child Plans
- [01-research-assistant/plan.md](01-research-assistant/plan.md)
- [02-code-review-agent/plan.md](02-code-review-agent/plan.md)
- [03-documentation-agent/plan.md](03-documentation-agent/plan.md)

## Inputs
- Understanding of Claude best practices for prompt engineering
- Knowledge of agentic workflow patterns
- Familiarity with tool use and structured prompts
- Existing repository structure and codebase

## Outputs
- Research assistant agent with paper analysis and code generation capabilities
- Code review agent with correctness, performance, and style review tools
- Documentation agent with API documentation, README, and tutorial generation
- Comprehensive test suites for all agents
- Configuration files and prompt templates for each agent

## Steps
1. Build research assistant with agent configuration, prompt templates, workflows, and tests
2. Build code review agent with review criteria, prompt templates, workflows, and tests
3. Build documentation agent with documentation standards, prompt templates, workflows, and tests

## Success Criteria
- [ ] All agents follow Claude best practices (clear roles, chain-of-thought, few-shot examples, XML tags, tool use)
- [ ] Research assistant can analyze papers and suggest implementations
- [ ] Code review agent can review code for correctness, performance, and style
- [ ] Documentation agent can generate API docs, READMEs, and tutorials
- [ ] All agents have comprehensive test coverage
- [ ] All child plans are completed successfully

## Notes
Focus on simplicity and following Claude best practices. Use structured prompts with clear role definitions, chain-of-thought for complex reasoning, few-shot examples for consistency, XML tags for structure, and appropriate tool use patterns. Keep implementations straightforward without over-engineering.
