# Validate Structure

## Overview
Validate that a paper's directory structure and files meet repository requirements. This check ensures all necessary directories and files exist and are properly organized before running tests.

## Parent Plan
[../plan.md](../plan.md)

## Child Plans
None (Level 4 - implementation level)

## Inputs
- Paper directory path
- Structure requirements and rules
- Required file checklist

## Outputs
- Validation report with pass/fail status
- List of missing or incorrect items
- Suggestions for fixing issues

## Steps
1. Check required directories exist
2. Verify required files are present
3. Validate file naming conventions
4. Generate structure validation report

## Success Criteria
- [ ] All required directories are checked
- [ ] All required files are verified
- [ ] Naming violations are detected
- [ ] Clear report helps fix issues

## Notes
Check for standard paper structure: src/, tests/, docs/, README.md. Verify test files follow naming conventions. Provide helpful messages about what's missing and where it should be.
