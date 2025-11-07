# Code Review Agent

## Overview
Create an AI code review agent that analyzes implementations for correctness, performance, and style. The agent follows Claude best practices with structured prompts, clear review criteria, and systematic evaluation.

## Parent Plan
[../plan.md](../plan.md)

## Child Plans
- [01-agent-configuration/plan.md](01-agent-configuration/plan.md)
- [02-prompt-templates/plan.md](02-prompt-templates/plan.md)
- [03-workflows/plan.md](03-workflows/plan.md)
- [04-testing/plan.md](04-testing/plan.md)

## Inputs
- Claude best practices for code review
- Understanding of review criteria (correctness, performance, style)
- Knowledge of Mojo/Python code patterns
- Existing codebase to review

## Outputs
- Agent configuration with review criteria and tools
- Prompt templates for different review types
- Workflows for PR review, quality checks, and improvement suggestions
- Test suite covering all review capabilities
- Documentation for using the code review agent

## Steps
1. Configure agent with review criteria and tools
2. Create prompt templates for correctness, performance, and style reviews
3. Implement workflows for PR review and quality checks
4. Write comprehensive tests for all review capabilities

## Success Criteria
- [ ] Agent configuration defines clear review criteria
- [ ] Prompt templates cover all review aspects
- [ ] Workflows provide actionable feedback
- [ ] Agent identifies correctness issues
- [ ] Agent suggests performance improvements
- [ ] Agent enforces style guidelines
- [ ] All tests pass with good coverage

## Notes
Follow Claude best practices for structured reviews. Use checklists to ensure comprehensive coverage. Provide specific, actionable feedback with code examples. Keep reviews constructive and helpful.
