# Prompt Templates

## Overview
Create structured prompt templates for the code review agent's different review types: correctness review, performance review, and style review. Each template uses Claude best practices with clear criteria and structured output.

## Parent Plan
[../plan.md](../plan.md)

## Child Plans
- [01-correctness-reviewer/plan.md](01-correctness-reviewer/plan.md)
- [02-performance-reviewer/plan.md](02-performance-reviewer/plan.md)
- [03-style-reviewer/plan.md](03-style-reviewer/plan.md)

## Inputs
- Review criteria definitions
- Claude best practices for code review prompts
- Examples of good code reviews
- Knowledge of XML tag patterns

## Outputs
- Correctness reviewer prompt template
- Performance reviewer prompt template
- Style reviewer prompt template
- Structured feedback formats
- Documentation for using templates

## Steps
1. Create correctness reviewer template with bug detection focus
2. Create performance reviewer template with efficiency focus
3. Create style reviewer template with readability focus

## Success Criteria
- [ ] All templates use structured output formats
- [ ] Templates follow review criteria
- [ ] Feedback is specific and actionable
- [ ] Templates include severity ratings
- [ ] Output format is consistent across templates

## Notes
Use XML tags to structure findings: <issue>, <location>, <severity>, <suggestion>. Include examples of the type of issues to look for. Guide the agent through systematic review processes. Keep feedback constructive.
