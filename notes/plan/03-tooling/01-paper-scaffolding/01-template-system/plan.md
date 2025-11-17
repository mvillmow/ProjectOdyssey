# Template System

## Overview

Build a template system for generating paper files with customizable content. Templates use variable substitution to insert paper-specific information like title, author, and date into boilerplate files.

## Parent Plan

[../plan.md](../plan.md)

## Child Plans

- [01-create-templates/plan.md](01-create-templates/plan.md)
- [02-template-variables/plan.md](02-template-variables/plan.md)
- [03-template-rendering/plan.md](03-template-rendering/plan.md)

## Inputs

- Template file definitions
- Variable substitution rules
- Paper metadata (title, author, etc.)

## Outputs

- Template files for README, Mojo code, tests
- Variable system for content customization
- Template rendering engine

## Steps

1. Create template files for common paper components
2. Define variable system for customization
3. Build rendering engine to generate files from templates

## Success Criteria

- [ ] Templates cover all required paper files
- [ ] Variables can be substituted in templates
- [ ] Rendering produces valid, formatted files
- [ ] All child plans are completed successfully

## Notes

Use simple string substitution for templates - no need for complex templating engine. Focus on making templates easy to create and modify.
