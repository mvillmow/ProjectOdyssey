# YAML Formatter

## Overview

Configure a formatter for YAML configuration files to ensure proper indentation, consistent syntax, and valid YAML structure.

## Parent Plan

[../plan.md](../plan.md)

## Child Plans

None (leaf node)

## Inputs

- [To be determined]

## Outputs

- Completed yaml formatter

## Steps

1. [To be determined]

## Success Criteria

- [ ] All YAML files formatted consistently
- [ ] Proper indentation (2 spaces)
- [ ] Valid YAML syntax
- [ ] No trailing whitespace
- [ ] Keys sorted if appropriate

## Notes

Use prettier with YAML plugin for auto-fixing. Configure 2-space indentation, max line length 120. Validate syntax with check-yaml hook before formatting.
