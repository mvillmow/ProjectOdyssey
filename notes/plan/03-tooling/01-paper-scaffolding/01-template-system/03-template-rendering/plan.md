# Template Rendering

## Overview

Implement the rendering engine that processes templates and substitutes variables to generate final output files. The renderer takes template files and variable values, performs substitution, and produces ready-to-use paper files.

## Parent Plan

[../plan.md](../plan.md)

## Child Plans

None (Level 4 - implementation level)

## Inputs

- Template files with placeholders
- Variable values from user input
- Output file paths

## Outputs

- Rendered files with substituted content
- Error messages for missing/invalid variables
- Rendering status and logs

## Steps

1. Load template files from disk
2. Parse template content for variable placeholders
3. Substitute variables with provided values
4. Write rendered content to output files

## Success Criteria

- [ ] Renderer correctly substitutes all variables
- [ ] Missing variables are detected and reported
- [ ] Rendered files are properly formatted
- [ ] Error handling provides clear messages

## Notes

Use simple string replacement for variable substitution. No need for complex templating engines - straightforward find-and-replace will work fine. Handle edge cases like missing variables gracefully.
