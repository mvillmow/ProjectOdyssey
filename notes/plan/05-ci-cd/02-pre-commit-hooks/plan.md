# Pre-commit Hooks

## Overview

Set up pre-commit hooks that run locally before commits to catch formatting, linting, and style issues early, improving code quality and reducing CI failures.

## Parent Plan
[Parent](../plan.md)

## Child Plans
- [01-hook-configuration](./01-hook-configuration/plan.md)
- [02-format-checker](./02-format-checker/plan.md)
- [03-linting](./03-linting/plan.md)

## Inputs
- Configure pre-commit hook framework
- Set up code formatters for Mojo, Markdown, and YAML
- Enable linting and type checking
- Provide fast local feedback before pushing
- Make hooks easy to install and use

## Outputs
- Completed pre-commit hooks
- Configure pre-commit hook framework (completed)

## Steps
1. Hook Configuration
2. Format Checker
3. Linting

## Success Criteria
- [ ] Pre-commit hooks installed and configured
- [ ] All file types formatted consistently
- [ ] Linting catches common issues
- [ ] Hooks run in under 10 seconds
- [ ] Clear documentation for installation
- [ ] Hooks are optional but encouraged

## Notes
- Use pre-commit framework for Python-based hooks
- Keep hooks fast to avoid slowing down commits
- Provide easy installation instructions
- Allow developers to skip hooks when needed (--no-verify)
- Focus on auto-fixable issues when possible