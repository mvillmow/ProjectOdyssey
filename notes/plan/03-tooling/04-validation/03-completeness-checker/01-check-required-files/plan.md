# Check Required Files

## Overview

Verify that all required repository files are present including configuration files, documentation, and standard project files. This ensures the repository is complete and properly configured.

## Parent Plan

[../plan.md](../plan.md)

## Child Plans

None (Level 4 - implementation level)

## Inputs

- Required files list
- Repository structure
- File location rules

## Outputs

- File presence check results
- Missing files list
- Incorrect file locations
- Recommendations

## Steps

1. Load list of required files
2. Check each file exists in expected location
3. Verify file is not empty
4. Report missing or misplaced files

## Success Criteria

- [ ] All required files are checked
- [ ] Missing files are identified
- [ ] Misplaced files are detected
- [ ] Clear guidance for fixes

## Notes

Check for: README.md, LICENSE, .gitignore, pyproject.toml/pixi.toml, CONTRIBUTING.md. Verify each paper has required files. Check shared code directories have proper structure. Flag empty or placeholder files.
