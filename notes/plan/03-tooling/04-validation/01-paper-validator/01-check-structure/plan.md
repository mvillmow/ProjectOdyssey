# Check Structure

## Overview

Validate that paper directory structure follows repository conventions with all required directories and files present. This check ensures papers are organized consistently.

## Parent Plan

[../plan.md](../plan.md)

## Child Plans

None (Level 4 - implementation level)

## Inputs

- Paper directory path
- Structure requirements specification
- Repository conventions

## Outputs

- Structure validation results
- List of missing directories/files
- Naming violations
- Recommendations for fixes

## Steps

1. Check required directories exist (src, tests, docs)
2. Verify required files are present (README.md, etc.)
3. Validate naming conventions
4. Generate structure report

## Success Criteria

- [ ] All required directories are checked
- [ ] All required files are verified
- [ ] Naming conventions are validated
- [ ] Clear feedback on violations

## Notes

Check for standard structure: src/ for implementation, tests/ for tests, docs/ for documentation, README.md in root. Flag unexpected directories or files. Suggest corrections for common mistakes.
