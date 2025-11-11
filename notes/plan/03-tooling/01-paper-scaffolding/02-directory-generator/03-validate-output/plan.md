# Validate Output

## Overview
Implement validation logic to verify that generated paper structures are complete and correct. This ensures all required directories and files exist, are properly formatted, and follow repository conventions.

## Parent Plan
[../plan.md](../plan.md)

## Child Plans
None (Level 4 - implementation level)

## Inputs
- Generated directory structure
- Generated files
- Validation checklist and rules

## Outputs
- Validation report with pass/fail status
- List of any missing or invalid items
- Suggestions for fixing issues

## Steps
1. Check all required directories exist
2. Verify all required files are present
3. Validate file content (proper format, no syntax errors)
4. Generate validation report with results

## Success Criteria
- [ ] Validation detects missing directories
- [ ] Validation detects missing files
- [ ] Validation checks file format and content
- [ ] Clear report shows any issues found

## Notes
Keep validation simple and focused on essential checks. Provide actionable error messages that tell users exactly what's wrong and how to fix it. Validation should never modify files.
