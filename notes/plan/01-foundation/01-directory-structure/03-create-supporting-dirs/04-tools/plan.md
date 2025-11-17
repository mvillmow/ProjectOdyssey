# Tools

## Overview

Create the tools/ directory for development utilities and helper tools that support the development workflow. This directory is for repository-specific development utilities, distinct from both Claude Code's built-in tool ecosystem and the existing scripts/ directory which contains automation scripts like create_issues.py. Tools here will include CLI utilities for common development tasks, code generation helpers, and workflow automation specific to this ML paper implementation repository.

## Parent Plan

[../plan.md](../plan.md)

## Child Plans

None (leaf node)

## Inputs

- Repository root directory exists
- Understanding of development workflow needs
- Knowledge of planned tooling from section 03-tooling (paper scaffolding, testing utilities, benchmarking tools, code generation utilities)
- Distinction between scripts/ (automation scripts) and tools/ (development utilities)

## Outputs

- tools/ directory at repository root
- tools/README.md explaining available development utilities
- Organized structure for different tool types (CLI tools, code generators, development helpers)
- Documentation for using and adding tools
- Foundation for future tools including:
  - Paper scaffolding tools
  - Testing utilities
  - Benchmarking tools
  - Code generation utilities

## Steps

1. Create tools/ directory at repository root
2. Write README explaining tool directory purpose and distinction from scripts/
3. Document expected types of development utilities (CLI tools, code generators, paper implementation helpers, workflow automation)
4. Provide guidelines for adding new development tools

## Success Criteria

- [ ] tools/ directory exists at repository root
- [ ] README clearly explains development utility purpose and how it differs from scripts/
- [ ] Documentation helps developers use and contribute tools
- [ ] Structure supports various tool categories (CLI, generators, helpers)

## Notes

Development tools improve productivity and consistency across paper implementations. Keep tools well-documented and easy to use. Include setup instructions and usage examples. This directory is for repository-specific development utilities, separate from Claude Code's built-in tools and the existing scripts/ directory. Specific tools will be developed in section 03-tooling as part of the broader development workflow.
