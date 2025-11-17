# Format Checker

## Overview

Configure pre-commit hooks to automatically format code files, ensuring consistent style across Mojo, Markdown, and YAML files.

## Parent Plan

[Parent](../plan.md)

## Child Plans

- [01-mojo-formatter](./01-mojo-formatter/plan.md)
- [02-markdown-formatter](./02-markdown-formatter/plan.md)
- [03-yaml-formatter](./03-yaml-formatter/plan.md)

## Inputs

- Set up Mojo code formatter
- Configure Markdown formatter
- Enable YAML formatting
- Auto-fix formatting issues when possible
- Ensure consistent code style

## Outputs

- Completed format checker
- Set up Mojo code formatter (completed)

## Steps

1. Mojo Formatter
2. Markdown Formatter
3. YAML Formatter

## Success Criteria

- [ ] All Mojo files formatted consistently
- [ ] Markdown files follow standard format
- [ ] YAML files properly formatted
- [ ] Formatting happens automatically on commit
- [ ] Formatting is fast (under 5 seconds)

## Notes

- Use official Mojo formatter if available
- For Markdown, use prettier or markdownlint
- For YAML, use yamllint or prettier
- Configure formatters to auto-fix when possible
- Keep formatter rules reasonable and not overly strict
