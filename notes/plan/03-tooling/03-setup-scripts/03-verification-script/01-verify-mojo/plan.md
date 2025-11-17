# Verify Mojo

## Overview

Check that Mojo is properly installed, accessible from PATH, and the correct version. This verification ensures developers can compile and run Mojo code.

## Parent Plan

[../plan.md](../plan.md)

## Child Plans

None (Level 4 - implementation level)

## Inputs

- Expected Mojo version (optional)
- PATH environment variable
- Installation paths to check

## Outputs

- Mojo installation status
- Installed version information
- PATH configuration status
- Any issues found

## Steps

1. Check if mojo command is available
2. Run mojo --version to get version info
3. Verify version meets requirements
4. Report results and any issues

## Success Criteria

- [ ] Detects if Mojo is installed
- [ ] Reports correct version information
- [ ] Identifies PATH issues
- [ ] Provides helpful error messages

## Notes

Check both system and user installations. Suggest installation steps if Mojo not found. Warn if version is too old or too new. Verify mojo can actually execute, not just that binary exists.
