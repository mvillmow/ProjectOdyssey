# Check Tests

## Overview

Validate that paper has adequate test coverage with properly structured test files. This check ensures implementations are tested and quality is maintained.

## Parent Plan

[../plan.md](../plan.md)

## Child Plans

None (Level 4 - implementation level)

## Inputs

- Paper tests directory
- Test file patterns
- Coverage requirements

## Outputs

- Test validation results
- Test file inventory
- Coverage assessment
- Missing test areas

## Steps

1. Check tests directory exists and has files
2. Verify test files follow naming conventions
3. Count tests and assess coverage
4. Report on test adequacy

## Success Criteria

- [ ] Test files are discovered
- [ ] Naming conventions are checked
- [ ] Test count is assessed
- [ ] Coverage gaps are identified

## Notes

Check for test files in tests/ directory. Verify naming follows conventions (test_*.mojo,*_test.mojo). Don't require specific coverage percentage but flag papers with no tests. Encourage comprehensive testing.
